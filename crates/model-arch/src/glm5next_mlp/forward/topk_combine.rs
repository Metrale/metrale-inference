// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The router top-k and the combine launches of a routed MoE site
//! (`forward_moe_pieces`), and the two launch-shape levers that make them cheaper at a few rows.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - `METRALE_GLM_ROUTER_TOPK_RANK=1` swaps `glm5next_router_topk` for
//!   `glm5next_router_topk_rank`, which writes the same ids and weights byte for byte
//!   (`glm5next_ffn.cu`); a lever that cannot run (kernel absent, too many experts) is an
//!   error, never a quiet fallback.
//! - `METRALE_GLM_MOE_COMBINE_SPLIT=1` only widens the combine's grid: each element is the same
//!   expressions in the same order whatever the grid.
//! - 2026-10-14: `METRALE_GLM_MOE_COMBINE_VEC8=1` swaps in the `_vec8` combine, eight elements
//!   per thread with the same per-element expressions (`glm5next_ffn.cu`); it errors where it
//!   cannot run.
//! - 2026-10-14: [`CombineRing`] (`METRALE_GLM_MOE_COMBINE_RING=1`, read by the SP-mHC MLP
//!   section) cuts the combine into one launch per row shard: a row is computed from that row's
//!   bytes alone (grid x is the row), so the shards' launches write the one launch's bytes.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::super::weights::Glm5NextMoeWeights;
use super::super::{Glm5NextMlpConfig, Glm5NextMlpKernels};
use super::{ACT_BLOCK, Glm5NextMlpWorkspace};

/// 2026-10-10: The most experts `glm5next_router_topk_rank` ranks (`ROUTER_RANK_MAX_EXPERTS` in
/// `glm5next_ffn.cu`): one thread per expert, at most 1024 threads.
pub const ROUTER_RANK_MAX_EXPERTS: usize = 1024;

/// 2026-10-10: The top-k kernel and its block width: the base kernel on `ACT_BLOCK` threads, or
/// (`rank`) the rank kernel on one thread per expert rounded up to whole warps.
pub(super) fn topk_launch(
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    rank: bool,
) -> Result<(KernelHandle, u32)> {
    if !rank {
        return Ok((k.router, ACT_BLOCK));
    }
    if k.router_rank.0 == 0 {
        bail!("METRALE_GLM_ROUTER_TOPK_RANK=1: glm5next_router_topk_rank is not in this target");
    }
    if cfg.num_experts == 0 || cfg.num_experts > ROUTER_RANK_MAX_EXPERTS {
        bail!(
            "METRALE_GLM_ROUTER_TOPK_RANK=1: {} experts; the rank kernel takes 1..={}",
            cfg.num_experts,
            ROUTER_RANK_MAX_EXPERTS
        );
    }
    Ok((k.router_rank, cfg.num_experts.div_ceil(32) as u32 * 32))
}

/// 2026-10-10: Blocks per row of the combine: one, or (`split`) one per `ACT_BLOCK` elements of
/// `hidden`, so each thread owns one element.
pub(super) fn combine_blocks_per_row(split: bool, hidden: usize) -> u32 {
    if split {
        hidden.div_ceil(ACT_BLOCK as usize).max(1) as u32
    } else {
        1
    }
}

/// 2026-10-10: The ids and weights of `rows` rows from `ws.logits` into `ws.ids` / `ws.wts`, one
/// block per row.
pub(super) fn router_topk(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextMoeWeights,
    ws: &Glm5NextMlpWorkspace,
    rows: usize,
    stream: u64,
) -> Result<()> {
    let (func, block) = topk_launch(k, cfg, crate::glm5next_layer::router_topk_rank())?;
    KernelLaunch::new(gpu, func)
        .grid([rows as u32, 1, 1])
        .block([block, 1, 1])
        .arg_ptr(ws.logits)
        .arg_ptr(w.router_bias)
        .arg_ptr(ws.ids)
        .arg_ptr(ws.wts)
        .arg_u32(cfg.num_experts as u32)
        .arg_u32(cfg.top_k as u32)
        .arg_u32(1)
        .arg_f32(cfg.routed_scale)
        .arg_u32(u32::from(cfg.renormalize))
        .arg_u32(u32::from(cfg.router_bf16_ladder))
        .launch(stream)
}

/// 2026-10-14: The most slots `glm5next_moe_combine[_indexed]_vec8` sums
/// (`GLM_COMBINE_VEC8_MAX_K` in `glm5next_ffn.cu`): one 16-byte register per slot.
pub const COMBINE_VEC8_MAX_K: usize = 16;

/// 2026-10-14: The combine kernel of one launch and its blocks per row. Off (`vec8` false), the
/// scalar kernel at [`combine_blocks_per_row`]; on, the `_vec8` twin at one thread per eight
/// elements of `hidden` (`split` is then moot: the grid always spreads a row). A `vec8` that
/// cannot run (kernel absent, `hidden` not a multiple of 8, more than [`COMBINE_VEC8_MAX_K`]
/// slots, a buffer of `bufs` not 16-byte aligned) is an error, never the scalar kernel.
pub(super) fn combine_launch(
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    indexed: bool,
    split: bool,
    vec8: bool,
    bufs: [DevicePtr; 3],
) -> Result<(KernelHandle, u32)> {
    if !vec8 {
        let func = if indexed {
            k.combine_indexed
        } else {
            k.combine
        };
        return Ok((func, combine_blocks_per_row(split, cfg.hidden)));
    }
    let func = k.combine_vec8[usize::from(indexed)];
    if func.0 == 0 {
        bail!("METRALE_GLM_MOE_COMBINE_VEC8=1: the _vec8 combine is not in this target");
    }
    if !cfg.hidden.is_multiple_of(8) || cfg.top_k > COMBINE_VEC8_MAX_K {
        bail!(
            "METRALE_GLM_MOE_COMBINE_VEC8=1: hidden {} is not a multiple of 8 or top_k {} > {}",
            cfg.hidden,
            cfg.top_k,
            COMBINE_VEC8_MAX_K
        );
    }
    if let Some(p) = bufs.iter().find(|p| !p.0.is_multiple_of(16)) {
        bail!(
            "METRALE_GLM_MOE_COMBINE_VEC8=1: buffer {:#x} is not 16-byte aligned",
            p.0
        );
    }
    Ok((
        func,
        (cfg.hidden / 8).div_ceil(ACT_BLOCK as usize).max(1) as u32,
    ))
}

/// 2026-10-14: The buffers one combine launch reads and writes, at its first row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CombinePtrs {
    pub expert_out: DevicePtr,
    pub token_to_perm: DevicePtr,
    pub wts: DevicePtr,
    pub shared: DevicePtr,
    pub out: DevicePtr,
}

impl CombinePtrs {
    /// 2026-10-14: The buffers of a launch whose row 0 is row `lo` of the launch at `self`:
    /// every per-row buffer moves `lo` rows, except the indexed kernel's `expert_out`, whose rows
    /// `token_to_perm` names absolutely. Row `t` of that launch is then row `lo + t` of this one,
    /// the same expressions on the same bytes.
    pub fn from_row(self, lo: usize, hidden: usize, top_k: usize, indexed: bool) -> Self {
        let (row, slots) = (hidden * 2, lo * top_k);
        Self {
            expert_out: if indexed {
                self.expert_out
            } else {
                self.expert_out.offset(slots * row)
            },
            token_to_perm: self.token_to_perm.offset(slots * 4),
            wts: self.wts.offset(slots * 4),
            shared: self.shared.offset(lo * row),
            out: self.out.offset(lo * row),
        }
    }
}

/// 2026-10-14: `METRALE_GLM_MOE_COMBINE_RING=1`: the combine as one launch per row shard of
/// `shards`, in that order, each followed by `events[i]` recorded on the compute stream, so a
/// ring reduce-scatter can send shard `i` while the later shards combine.
#[derive(Clone, Copy, Debug)]
pub struct CombineRing<'a> {
    pub shards: &'a [(usize, usize)],
    pub events: &'a [u64],
}

impl CombineRing<'_> {
    /// 2026-10-14: Errors unless there is one event per shard and the shards, in any order,
    /// tile `0..rows` (every row combined exactly once, as by the one launch).
    pub fn check(&self, rows: usize) -> Result<()> {
        if self.shards.len() != self.events.len() {
            bail!(
                "MoE combine ring: {} shards but {} events",
                self.shards.len(),
                self.events.len()
            );
        }
        let mut sorted = self.shards.to_vec();
        sorted.sort_unstable();
        let mut at = 0;
        for (lo, hi) in sorted {
            if lo != at || hi < lo {
                bail!(
                    "MoE combine ring: shards {:?} do not tile 0..{rows}",
                    self.shards
                );
            }
            at = hi;
        }
        if at != rows {
            bail!(
                "MoE combine ring: shards {:?} do not tile 0..{rows}",
                self.shards
            );
        }
        Ok(())
    }
}

/// 2026-10-10: `out = shared + sum_k wts[k] * routed[k]` for `rows` rows. After the grouped GEMM
/// (`indexed`) the routed rows are in expert-sorted order and `glm5next_moe_combine_indexed`
/// finds each slot's row through `token_to_perm`. 2026-10-14: the kernel and grid from
/// [`combine_launch`] under the two combine levers; with `ring`, one launch per shard
/// ([`CombineRing`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn combine(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    ws: &Glm5NextMlpWorkspace,
    out: DevicePtr,
    rows: usize,
    indexed: bool,
    ring: Option<&CombineRing<'_>>,
    stream: u64,
) -> Result<()> {
    let levers = (
        crate::glm5next_layer::moe_combine_split(),
        crate::glm5next_layer::moe_combine_vec8(),
    );
    let Some(ring) = ring else {
        return combine_with(gpu, k, cfg, ws, out, (0, rows), indexed, levers, stream);
    };
    ring.check(rows)?;
    for (&shard, &ev) in ring.shards.iter().zip(ring.events) {
        combine_with(gpu, k, cfg, ws, out, shard, indexed, levers, stream)?;
        gpu.record_event(ev, stream)?;
    }
    Ok(())
}

/// 2026-10-14: [`combine`] of rows `lo..hi` with its `(split, vec8)` levers passed in (no launch
/// when the range is empty). Every kernel it picks takes the same arguments.
#[allow(clippy::too_many_arguments)]
pub(super) fn combine_with(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    ws: &Glm5NextMlpWorkspace,
    out: DevicePtr,
    (lo, hi): (usize, usize),
    indexed: bool,
    (split, vec8): (bool, bool),
    stream: u64,
) -> Result<()> {
    if hi <= lo {
        return Ok(());
    }
    let p = CombinePtrs {
        expert_out: ws.expert_out,
        token_to_perm: ws.token_to_perm,
        wts: ws.wts,
        shared: ws.shared_out,
        out,
    }
    .from_row(lo, cfg.hidden, cfg.top_k, indexed);
    let bufs = [p.expert_out, p.shared, p.out];
    let (func, y) = combine_launch(k, cfg, indexed, split, vec8, bufs)?;
    let launch = KernelLaunch::new(gpu, func)
        .grid([(hi - lo) as u32, y, 1])
        .block([ACT_BLOCK, 1, 1])
        .arg_ptr(p.expert_out);
    let launch = if indexed {
        launch.arg_ptr(p.token_to_perm)
    } else {
        launch
    };
    launch
        .arg_ptr(p.wts)
        .arg_ptr(p.shared)
        .arg_ptr(p.out)
        .arg_u32(cfg.hidden as u32)
        .arg_u32(cfg.top_k as u32)
        .launch(stream)
}

#[cfg(test)]
#[path = "topk_combine_tests.rs"]
mod tests;
