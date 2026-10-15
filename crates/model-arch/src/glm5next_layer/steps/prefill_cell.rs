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
