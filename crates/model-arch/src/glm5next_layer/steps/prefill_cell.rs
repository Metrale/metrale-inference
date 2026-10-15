// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The layer side of `METRALE_GLM_PREFILL_CELL_GRID` (`glm5next_prefill_cell`): a
//! prefill pass's segments planned per cell, each with the path the grid serve runs its rows
//! on, and a KDA layer's SSM state clamp between its cells: `ssm_state_clamp_norm_fused` over
//! this layer's state alone (grid `[heads, 1]`), the per-(head, layer) block the model's
//! between-pass launch (grid `[heads, layers]`) runs for it.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - The clamp touches only this layer's FP32 recurrent state, on the layer's stream after the
//!   segment's launches; a DSA layer clamps nothing.
//! - Each KDA layer owns its own device pointer slot, so no two layers' clamps share a buffer.

use super::forward_sp_pipe::PipeSub;
use super::prefill_capture::{CapturePoint, capture_points};
use super::*;
use crate::glm5next_kda::cut::KdaCut;
use crate::glm5next_prefill_cell as cell;
use metrale_model_layers::layer::MidchunkCapture;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// 2026-10-15: One segment of a layer's prefill pass: rows `start..end`, the capture after it,
/// its KDA cuts (merge mode), whether the state clamp runs after it (a cell end, cell mode
/// only), and its path when cell mode decided it (`None`: the pre-existing segment rule).
pub(in crate::glm5next_layer) struct LayerSeg {
    pub start: usize,
    pub end: usize,
    pub after: Option<CapturePoint>,
    pub cuts: Vec<KdaCut>,
    pub clamp_after: bool,
    pub decode: Option<bool>,
}

/// 2026-10-15: The clamp kernel and each KDA layer's one-pointer device slot, by layer index.
struct ClampRig {
    kernel: KernelHandle,
    slots: HashMap<usize, DevicePtr>,
}

static RIG: OnceLock<Mutex<Option<ClampRig>>> = OnceLock::new();

/// 2026-10-15: The clamp kernel and layer `layer_idx`'s pointer slot, resolved and allocated
/// on first use.
fn rig(gpu: &dyn GpuBackend, layer_idx: usize) -> Result<(KernelHandle, DevicePtr)> {
    let mut g = RIG
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|_| anyhow::anyhow!("{}: the clamp rig lock is poisoned", cell::CELL_GRID_ENV))?;
    if g.is_none() {
        let kernel = gpu
            .kernel("ssm_state_norm", "ssm_state_clamp_norm_fused")
            .with_context(|| format!("{} needs ssm_state_clamp_norm_fused", cell::CELL_GRID_ENV))?;
        *g = Some(ClampRig {
            kernel,
            slots: HashMap::new(),
        });
    }
    let r = g.as_mut().expect("set above");
    let slot = match r.slots.get(&layer_idx) {
        Some(&p) => p,
        None => {
            let p = gpu.alloc(8)?;
            r.slots.insert(layer_idx, p);
            p
        }
    };
    Ok((r.kernel, slot))
}

impl Glm5NextLayer {
    /// 2026-10-15: The segments of a pass of `num_tokens` rows from `seq_len_start` under
    /// `grid` (`glm5next_prefill_cell::plan`), the capture points of `cap` handled per cell in
    /// the pre-existing mode (`merge`), each segment's path by `seg_takes_decode`.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn cell_segments(
        &self,
        grid: cell::CellGrid,
        num_tokens: usize,
        rows: usize,
        merge: bool,
        seq_len_start: usize,
        ctx: &ForwardContext,
        cap: Option<&MidchunkCapture<'_>>,
        ordinal: Option<usize>,
    ) -> Result<Vec<LayerSeg>> {
        let points = capture_points(num_tokens, cap);
        let rank0 = ctx.comm.is_none_or(|c| c.rank() == 0);
        let mut out = Vec::new();
        for g in cell::plan(grid, seq_len_start, num_tokens, rows, merge, &points, rank0) {
            let decode = cell::seg_takes_decode(
                &g,
                seq_len_start + g.start,
                ctx.levers.one_row_tail_decode,
                ctx.levers.one_row_prefill,
            );
            let cuts = g
                .cuts
                .iter()
                .map(|&(row, which)| {
                    let dst = match (ordinal, cap) {
                        (Some(idx), Some(c)) => self.capture_dsts(c, which, idx)?,
                        _ => None,
                    };
                    Ok(KdaCut { row, dst })
                })
                .collect::<Result<Vec<KdaCut>>>()?;
            out.push(LayerSeg {
                start: g.start,
                end: g.end,
                after: g.after,
                cuts,
                clamp_after: g.clamp_after,
                decode: Some(decode),
            });
        }
        Ok(out)
    }

    /// 2026-10-15: What runs on this layer's mixer state after a segment: the capture at its end
    /// (point `after`, SSM ordinal `ordinal` of `cap`), then the cell clamp when `clamp_after`.
    /// The serial loop runs it after the segment, the cell pipeline after the segment's mixer
    /// (`Prim::AfterMixer`), the last launch before it that touches the state.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn seg_after(
        &self,
        state: &mut dyn LayerState,
        after: Option<CapturePoint>,
        clamp_after: bool,
        ordinal: Option<usize>,
        cap: Option<&MidchunkCapture<'_>>,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if let (Some(which), Some(idx), Some(c)) = (after, ordinal, cap) {
            self.capture_kda_state(state, c, which, idx, gpu, stream)?;
        }
        if clamp_after {
            self.cell_clamp(state, gpu, stream)?;
        }
        Ok(())
    }

    /// 2026-10-15: `METRALE_GLM_PREFILL_CELL_PIPELINE=1`: the leading run of `segs` (consecutive,
    /// rows of the pass from `seq_len_start` at `hidden`, highway slot `slot0`) that each are one
    /// SP-mHC sub-chunk of at most `rows` off the decode path, pipelined through this layer
    /// (`steps/forward_sp_pipe.rs`) when it holds two or more. Returns how many segments it ran
    /// (0: none, the caller runs them serially). Every rank plans the same segment boundaries
    /// (`glm5next_prefill_cell::plan`; only the clamps differ by rank), so the programs pair.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn cell_pipeline_run(
        &self,
        hidden: DevicePtr,
        segs: &[LayerSeg],
        rows: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
        slot0: usize,
        ordinal: Option<usize>,
        cap: Option<&MidchunkCapture<'_>>,
    ) -> Result<usize> {
        let mut takes = Vec::with_capacity(segs.len());
        for (i, g) in segs.iter().enumerate() {
            let k = g.end - g.start;
            anyhow::ensure!(
                i == 0 || segs[i - 1].end == g.start,
                "GLM layer {}: cell segments are not consecutive",
                self.layer_idx
            );
            if rows < 2 || g.decode != Some(false) || k > rows || !self.sp_takes(ctx, k, true)? {
                break;
            }
            takes.push(true);
        }
        let p = crate::glm5next_sp_pipeline::pipelined(&takes);
        if p == 0 {
            return Ok(0);
        }
        let subs: Vec<PipeSub<'_>> = segs[..p]
            .iter()
            .map(|g| PipeSub {
                k: g.end - g.start,
                cuts: &g.cuts,
                after: g.after,
                clamp_after: g.clamp_after,
            })
            .collect();
        let t = segs[0].start;
        self.forward_sp_pipeline(
            hidden.offset(t * self.hidden * 2),
            &subs,
            state,
            kv_cache,
            seq_len_start + t,
            block_table,
            ctx,
            stream,
            slot0 + t,
            ordinal,
            cap,
        )?;
        Ok(p)
    }

    /// 2026-10-15: Clamp this KDA layer's recurrent state on `stream`, as the model's
    /// between-pass clamp does for its block of this layer; nothing for a DSA layer.
    pub(in crate::glm5next_layer) fn cell_clamp(
        &self,
        state: &mut dyn LayerState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let Glm5NextMixer::Kda { cfg, .. } = &self.mixer else {
            return Ok(());
        };
        let (heads, head_dim) = (cfg.heads as u32, cfg.head_dim as u32);
        let h = self.kda_state(state)?.h_state;
        let (kernel, slot) = rig(gpu, self.layer_idx)?;
        gpu.copy_h2d_async(&h.0.to_le_bytes(), slot, stream)?;
        KernelLaunch::new(gpu, kernel)
            .grid([heads, 1, 1])
            .block([head_dim, 1, 1])
            .arg_ptr(slot)
            .arg_u32(heads)
            .arg_u32(head_dim)
            .arg_u32(head_dim)
            .launch(stream)
    }
}
