// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The layer types (`FfnComponent` and the modules below) and the
//! kernel-handle resolvers several layers share.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.

pub mod dense_ffn;
pub mod ep_dispatch;
pub mod fp8_calibration;
mod gemv_tier;
pub mod glm_vit;
pub mod moe;
pub mod mtp_head;
pub mod mtp_meta;
pub mod mtp_multi;
pub mod ngram_embed;
pub mod ops;
pub mod ple;
pub mod qsa;
pub mod qwen3_attention;
pub mod qwen3_ssm;
pub mod vision_encoder;
pub mod vision_tower;
pub mod w4a16_gemv_tiers;

/// 2026-09-25: Smallest K at which the tile-GEMM sites (`dense_ffn.rs`,
/// `qwen3_attention/trait_impl/multi_seq/qkv.rs`, `qwen3_ssm/kernel_select.rs`)
/// take the deep-K `w4a16_gemm_t_k64` handle.
///
/// Resolved once per process: `METRALE_W4A16_K64_MIN_K` (a parsed `u32`) if
/// set, else 8192 when `METRALE_NO_W4A16_K64=1`, else 6144.
pub(crate) fn w4a16_k64_min_k() -> u32 {
    static MIN_K: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *MIN_K.get_or_init(|| {
        if let Some(n) = std::env::var("METRALE_W4A16_K64_MIN_K")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
        {
            return n;
        }
        if std::env::var("METRALE_NO_W4A16_K64").ok().as_deref() == Some("1") {
            8192
        } else {
            6144
        }
    })
}

pub use dense_ffn::{DenseFfnLayer, DenseFfnWeights, FfnActivation};

pub use glm_vit::{GlmVit, GlmVitBlock, GlmVitMerger};

pub use moe::MoeLayer;
// 2026-09-28: The MoE experts' decode activation cell (`moe/fp8_grouped_tc_w8a8.rs`).
pub use moe::{moe_expert_fp8_act, set_moe_expert_fp8_act};
pub use mtp_head::{
    MtpHead, MtpQuantization, mtp_drafter_prefill_enabled, set_mtp_experts_nvfp4_from_cli,
};

pub use qwen3_attention::Qwen3AttentionLayer;
pub use qwen3_ssm::Qwen3SsmLayer;
pub use vision_encoder::{MergerLayer, ViTBlock, VisionEncoder};
pub use vision_tower::VisionTower;

use crate::layer::ForwardContext;
use anyhow::Result;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

/// 2026-09-26: The `w4a16_gemm_t_m128_v2` handle for the layer builders in
/// `dense_ffn_init.rs`, `qwen3_attention/init_prefill_kernels.rs` and
/// `qwen3_ssm/init.rs`.
///
/// `KernelHandle(0)` unless `METRALE_W4A16_VARIANT` is `v2` or `v3`, with no
/// lookup. For `v2`/`v3` it panics when the target lacks the kernel.
#[track_caller]
pub fn w4a16_v2_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    let variant = std::env::var("METRALE_W4A16_VARIANT").ok();
    // 2026-09-25: Off unless requested. Measured 2026-07-30 with
    // `w4a16_bf16_v2_bench` on the 27B FFN shapes: v2 ran at 0.78-0.82x of v1.
    if !matches!(variant.as_deref(), Some("v2") | Some("v3")) {
        if variant.as_deref() == Some("v1") {
            tracing::debug!("METRALE_W4A16_VARIANT=v1: w4a16 m128 v2 suppressed (explicit)");
        }
        return KernelHandle(0);
    }
    let h = try_kernel(gpu, "w4a16_v2", "w4a16_gemm_t_m128_v2");
    if h.0 == 0 {
        panic!(
            "METRALE_W4A16_VARIANT={} requested but w4a16_v2::w4a16_gemm_t_m128_v2 is not in this \
             target's kernel set — refusing to start with a silently-degraded config",
            variant.unwrap()
        );
    }
    tracing::debug!(
        handle = h.0,
        "w4a16_gemm_t_m128_v2 resolution (explicit opt-in)"
    );
    h
}

/// 2026-09-25: The `w4a16_gemm_t_m128_v3` handle. `KernelHandle(0)`, with no
/// lookup and so no boot-audit row, unless `METRALE_W4A16_VARIANT=v3`; then it
/// panics when the target lacks the kernel. Its user
/// (`qwen3_attention/prefill_weights.rs`) launches it only for `v == 3` and a
/// nonzero handle.
#[track_caller]
pub fn w4a16_v3_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    if std::env::var("METRALE_W4A16_VARIANT").as_deref() != Ok("v3") {
        return KernelHandle(0);
    }
    let h = try_kernel(gpu, "w4a16_v3", "w4a16_gemm_t_m128_v3");
    if h.0 == 0 {
        panic!(
            "METRALE_W4A16_VARIANT=v3 requested but w4a16_v3::w4a16_gemm_t_m128_v3 is not in this \
             target's kernel set — refusing to start with a silently-degraded config"
        );
    }
    h
}

/// 2026-09-25: The N128/M64 tile GEMM. Prefers `w4a16_gemm_t_p3`; takes
/// `w4a16_gemm_t` when `METRALE_NO_TGEMM_PIPELINE3` is set to any value
/// (`0` included) or the target lacks `_p3`.
#[track_caller]
pub fn tgemm_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    if std::env::var("METRALE_NO_TGEMM_PIPELINE3").is_err() {
        let h = try_kernel(gpu, "w4a16", "w4a16_gemm_t_p3");
        if h.0 != 0 {
            return h;
        }
    }
    try_kernel(gpu, "w4a16", "w4a16_gemm_t")
}

/// 2026-09-25: False for `kimi_k3` and `kimi_linear`. For those the model
/// builder (`model-engine` `model/impl_a1.rs`) skips the `tgemm_kernel` lookup
/// and keeps a zero handle.
pub fn tgemm_probe_ok(model_type: &str) -> bool {
    !matches!(model_type, "kimi_k3" | "kimi_linear")
}

/// 2026-09-25: The k64 deep-K tile GEMM. Prefers `w4a16_gemm_t_k64_p3`; takes
/// `w4a16_gemm_t_k64` when `METRALE_NO_K64_PIPELINE3` is set to any value
/// (`0` included) or the target lacks `_p3`, and errors when that is missing too.
#[track_caller]
pub fn k64_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    let want_p3 = std::env::var("METRALE_NO_K64_PIPELINE3").is_err();
    if want_p3 {
        let h = try_kernel(gpu, "w4a16", "w4a16_gemm_t_k64_p3");
        if h.0 != 0 {
            return Ok(h);
        }
    }
    gpu.kernel("w4a16", "w4a16_gemm_t_k64")
}

/// 2026-09-25: The N_TILE=64 deep-K twin `w4a16_gemm_t_k64_n64_p3`.
/// `KernelHandle(0)` when the target lacks it, or, with no lookup, when
/// `METRALE_NO_K64_N64` is set to any value (`0` included).
#[track_caller]
pub fn k64_n64_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    if std::env::var("METRALE_NO_K64_N64").is_ok() {
        return KernelHandle(0);
    }
    try_kernel(gpu, "w4a16", "w4a16_gemm_t_k64_n64_p3")
}

/// 2026-09-25: Largest wide-tile grid at which the N_TILE=64 twin serves instead.
///
/// The wide tile launches `ceil(n/128) * ceil(m/64)` CTAs of 128 threads
/// (`ops::w4a16_gemm_n128_ldb`); the twin launches `ceil(n/64) * ceil(m/64)`
/// (`ops::w4a16_gemm`), about twice the grid.
const K64_N64_MAX_WIDE_CTAS: u32 = 64;

/// 2026-09-25: Whether the N_TILE=64 twin serves an `m × n` launch
/// (`K64_N64_MAX_WIDE_CTAS`).
pub fn k64_n64_wins(m: u32, n: u32) -> bool {
    n.div_ceil(128) * m.div_ceil(64) <= K64_N64_MAX_WIDE_CTAS
}

mod moe_grouped_decode;
pub use moe_grouped_decode::*;

mod expert_quantization;
mod moe_expert_tables;
pub use expert_quantization::{
    ExpertQuantization, expert_quantization, set_expert_quantization_from_cli,
};
pub use moe_expert_tables::{MoeExpertTables, moe_expert_tables, set_moe_expert_tables_from_plan};

mod activation_quantization;
pub use activation_quantization::{
    act_route, activation_quantization, any_fixed, family_fixed, fixed_act,
    set_activation_quantization_from_cli,
};
mod weight_quantization;
pub use weight_quantization::{kernel_caps, set_weight_quantization_from_cli, weight_quantization};

mod row_tiers;
pub use row_tiers::{
    RowTiers, prefill_row_invariant, publish_row_tiers, resolve_row_tiers, row_invariant,
    row_tiers, row_tiers_from,
};

mod kernel_probe;
pub use kernel_probe::{try_kernel, try_target_kernel};

mod w8a8_layer;
pub use w8a8_layer::{W8a8Ctx, W8a8Ffn, W8a8Mixer};

mod ffn_component;
pub use ffn_component::FfnComponent;

pub(crate) use gemv_tier::batch8_kernel;
