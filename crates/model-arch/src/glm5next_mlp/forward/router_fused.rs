// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: `--glm-router-fused` (R2): the routed MoE site's router logits and top-k as one
//! launch, `glm5next_router_gemv_topk_few` (`glm5next_router_fused.cu`, an instantiation of
//! `glm_router_gemv_topk.cuh`), in place of `dense_gemv_bf16_batchm_few_fp32out` then
//! `glm5next_router_topk_rank`: same logits, ids and weights (`glm5next_router_fused_gate`).
//!
//! Owner: model-arch (GLM-5.3 MLP).
//! Invariants:
//! - Taken only where the pair would be exactly those two kernels ([`takes`]): one piece of
//!   2..=8 rows, the batched router rows (`METRALE_GLM_ROUTER_ROWS`) on the few-row entry at four
//!   outputs a block, the rank top-k (`METRALE_GLM_ROUTER_TOPK_RANK`), and a shape the kernel
//!   holds; otherwise the caller runs the pair. The lever on where it applies without the kernel
//!   in the build is an error, never a quiet fallback.
//! - Never under an installed MoE plan: installation refuses the lever (`glm5next_plan`).

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::super::{Glm5NextMlpConfig, Glm5NextMlpKernels, Glm5NextMoeWeights};
use super::Glm5NextMlpWorkspace;
use crate::glm5next_layer::wide_gemv::{FEW_NPB, fixed_entry};

/// 2026-10-24: The most experts and slots the fused kernel ranks (`GLM_RGT_MAX_EXPERTS`, the
/// rank body's 16 slots).
pub const ROUTER_FUSED_MAX_EXPERTS: usize = 512;
pub const ROUTER_FUSED_MAX_TOP_K: usize = 16;

/// 2026-10-24: Whether a site of `row_pieces` runs the few-row router GEMV then the rank top-k,
/// the pair the fused kernel replaces (the lever aside).
pub(super) fn takes(k: &Glm5NextMlpKernels, cfg: &Glm5NextMlpConfig, row_pieces: &[usize]) -> bool {
    let &[rows] = row_pieces else {
        return false;
    };
    let l = &k.levers;
    let few = k.gemv_batchm_few;
    (2..=8).contains(&rows)
        && l.router_rows
        && k.gemv_f32.0 != 0
        && k.gemv_batchm_f32.0 != 0
        && k.gemv_batchm_wide_f32.0 != 0
        && l.router_topk_rank
        && k.router_rank.0 != 0
        && few.f32.0 != 0
        && few.f32_npb == FEW_NPB
        && fixed_entry(&k.batchm_f32(), rows, &l.dense).map(|h| h.0) == Some(few.f32.0)
        && (1..=ROUTER_FUSED_MAX_EXPERTS).contains(&cfg.num_experts)
        && (1..=ROUTER_FUSED_MAX_TOP_K).contains(&cfg.top_k)
}

/// 2026-10-24: Launch the fused router over `x` when the lever is on and [`takes`] holds;
/// `false` (nothing launched) otherwise.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_launch(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextMoeWeights,
    x: DevicePtr,
    ws: &Glm5NextMlpWorkspace,
    row_pieces: &[usize],
    stream: u64,
) -> Result<bool> {
    if !k.levers.decode.router_fused || !takes(k, cfg, row_pieces) {
        return Ok(false);
    }
    ensure!(
        k.router_fused.0 != 0,
        "--glm-router-fused: glm5next_router_gemv_topk_few is not in this build"
    );
    let rows = row_pieces[0];
    KernelLaunch::new(gpu, k.router_fused)
        .grid([(cfg.num_experts as u32).div_ceil(FEW_NPB), 1, 1])
        .block([64 * FEW_NPB, 1, 1])
        .arg_ptr(x)
        .arg_ptr(w.router)
        .arg_ptr(ws.logits)
        .arg_ptr(w.router_bias)
        .arg_ptr(ws.ids)
        .arg_ptr(ws.wts)
        .arg_ptr(ws.route_ctl)
        .arg_u32(rows as u32)
        .arg_u32(cfg.num_experts as u32)
        .arg_u32(cfg.hidden as u32)
        .arg_u32(cfg.top_k as u32)
        .arg_f32(cfg.routed_scale)
        .arg_u32(u32::from(cfg.renormalize))
        .arg_u32(u32::from(cfg.router_bf16_ladder))
        .launch(stream)?;
    // 2026-10-24: Once per process: the serve log's proof that R2 reached the router.
    static TAKEN: std::sync::Once = std::sync::Once::new();
    TAKEN.call_once(|| tracing::info!("GLM MoE: fused router + top-k taken ({rows} rows)"));
    Ok(true)
}

#[cfg(test)]
#[path = "router_fused_tests.rs"]
mod tests;
