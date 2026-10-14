// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The expert sort of the grouped W4A4 routed path: `moe_sort_by_expert` on one
//! block, or under `METRALE_GLM_MOE_SORT_MULTIBLOCK=1` the three `moe_sort_mb_*` launches
//! (`moe_permute.cu`) over one block per [`SORT_MB_CHUNK`] slots.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Both write `expert_offsets`, `sorted_token_ids`, `sorted_expert_ids` and `token_to_perm` with
//!   the same offsets; the multi-block sort keeps each expert's slots in slot order, which is one
//!   of the orders the one-block sort's shared atomics may leave. Each routed row's GEMM output
//!   depends only on that row, and the combine finds it through `token_to_perm`.
//! - A lever that cannot run (kernels absent, too many experts) is an error, never the one-block
//!   sort.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::Glm5NextMlpWorkspace;
use crate::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};

/// 2026-10-14: Slots one `moe_sort_mb_hist` / `moe_sort_mb_scatter` block owns
/// (`MOE_SORT_MB_CHUNK` in `moe_permute.cu`).
pub const SORT_MB_CHUNK: usize = 1024;

/// 2026-10-14: The most experts the multi-block sort counts (`MOE_SORT_MB_MAX_EXPERTS`): its
/// shared counters, and one scan thread per expert.
pub const SORT_MB_MAX_EXPERTS: usize = 1024;

/// 2026-10-14: The block of `moe_sort_by_expert`, `moe_sort_mb_hist` and `moe_sort_mb_scatter`;
/// the scatter's eight warps (`MOE_SORT_MB_WARPS`) need exactly this.
const SORT_BLOCK: u32 = 256;

/// 2026-10-14: The `sort_hist` bytes of a workspace of `total_expanded` slots: one count per
/// expert per block.
pub fn sort_hist_bytes(cfg: &Glm5NextMlpConfig, total_expanded: usize) -> usize {
    total_expanded.div_ceil(SORT_MB_CHUNK).max(1) * cfg.num_experts.max(1) * 4
}

/// 2026-10-14: [`sort_by_expert_with`] under `METRALE_GLM_MOE_SORT_MULTIBLOCK`.
pub(super) fn sort_by_expert(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    ws: &Glm5NextMlpWorkspace,
    slots: usize,
    stream: u64,
) -> Result<()> {
    let multiblock = crate::glm5next_layer::moe_sort_multiblock();
    sort_by_expert_with(gpu, k, cfg, ws, slots, multiblock, stream)
}

/// 2026-10-14: The `slots` ids of `ws.ids` sorted by expert into the workspace's routing tables:
/// the one-block sort, or (`multiblock`) the three multi-block launches.
pub(super) fn sort_by_expert_with(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    ws: &Glm5NextMlpWorkspace,
    slots: usize,
    multiblock: bool,
    stream: u64,
) -> Result<()> {
    let (e, topk) = (cfg.num_experts as u32, cfg.top_k as u32);
    if !multiblock {
        return KernelLaunch::new(gpu, k.moe_sort_by_expert)
            .grid([1, 1, 1])
            .block([SORT_BLOCK, 1, 1])
            .arg_ptr(ws.ids)
            .arg_ptr(ws.sorted_token_ids)
            .arg_ptr(ws.sorted_expert_ids)
            .arg_ptr(ws.expert_offsets)
            .arg_ptr(ws.token_to_perm)
            .arg_u32(slots as u32)
            .arg_u32(e)
            .arg_u32(topk)
            .launch(stream);
    }
    let [hist, scan, scatter] = k.sort_mb;
    ensure!(
        [hist, scan, scatter].iter().all(|h| h.0 != 0),
        "METRALE_GLM_MOE_SORT_MULTIBLOCK=1: moe_sort_mb_* is not in this target"
    );
    ensure!(
        (1..=SORT_MB_MAX_EXPERTS).contains(&cfg.num_experts),
        "METRALE_GLM_MOE_SORT_MULTIBLOCK=1: {} experts; the multi-block sort takes 1..={}",
        cfg.num_experts,
        SORT_MB_MAX_EXPERTS
    );
    ensure!(
        slots <= ws.max_total_expanded,
        "METRALE_GLM_MOE_SORT_MULTIBLOCK=1: {slots} slots past the workspace's {}",
        ws.max_total_expanded
    );
    let nb = slots.div_ceil(SORT_MB_CHUNK).max(1) as u32;
    KernelLaunch::new(gpu, hist)
        .grid([nb, 1, 1])
        .block([SORT_BLOCK, 1, 1])
        .arg_ptr(ws.ids)
        .arg_ptr(ws.sort_hist)
        .arg_u32(slots as u32)
        .arg_u32(e)
        .launch(stream)?;
    KernelLaunch::new(gpu, scan)
        .grid([1, 1, 1])
        .block([e.div_ceil(32) * 32, 1, 1])
        .arg_ptr(ws.sort_hist)
        .arg_ptr(ws.expert_offsets)
        .arg_u32(nb)
        .arg_u32(e)
        .launch(stream)?;
    KernelLaunch::new(gpu, scatter)
        .grid([nb, 1, 1])
        .block([SORT_BLOCK, 1, 1])
        .arg_ptr(ws.ids)
        .arg_ptr(ws.sort_hist)
        .arg_ptr(ws.expert_offsets)
        .arg_ptr(ws.sorted_token_ids)
        .arg_ptr(ws.sorted_expert_ids)
        .arg_ptr(ws.token_to_perm)
        .arg_u32(slots as u32)
        .arg_u32(e)
        .arg_u32(topk)
        .launch(stream)
}

#[cfg(test)]
#[path = "sort_tests.rs"]
mod tests;
