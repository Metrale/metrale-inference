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

/// 2026-10-10: `out = shared + sum_k wts[k] * routed[k]` for `rows` rows. After the grouped GEMM
/// (`indexed`) the routed rows are in expert-sorted order and `glm5next_moe_combine_indexed`
/// finds each slot's row through `token_to_perm`.
#[allow(clippy::too_many_arguments)]
pub(super) fn combine(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    ws: &Glm5NextMlpWorkspace,
    out: DevicePtr,
    rows: usize,
    indexed: bool,
    stream: u64,
) -> Result<()> {
    let y = combine_blocks_per_row(crate::glm5next_layer::moe_combine_split(), cfg.hidden);
    let launch = KernelLaunch::new(
        gpu,
        if indexed {
            k.combine_indexed
        } else {
            k.combine
        },
    )
    .grid([rows as u32, y, 1])
    .block([ACT_BLOCK, 1, 1])
    .arg_ptr(ws.expert_out);
    let launch = if indexed {
        launch.arg_ptr(ws.token_to_perm)
    } else {
        launch
    };
    launch
        .arg_ptr(ws.wts)
        .arg_ptr(ws.shared_out)
        .arg_ptr(out)
        .arg_u32(cfg.hidden as u32)
        .arg_u32(cfg.top_k as u32)
        .launch(stream)
}

#[cfg(test)]
#[path = "topk_combine_tests.rs"]
mod tests;
