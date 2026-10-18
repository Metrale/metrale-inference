// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The Qwen3.5 MoE loader levers, resolved from the startup settings once per load.
//!
//! Owner: model-arch weight loader (Qwen3.5).
//! Invariants: `load_layers` resolves this once before its layer loop; each field reads one
//! settings field with the parse its old reader applied. `METRALE_HOLO_FAST_MOE_MODE` is not
//! here: its settings row records presence only, and the mode is its text.

use metrale_settings::InferenceServerSettings;

use super::super::super::shared_levers::{grouped_cutlass, unified_moe_layout};

/// 2026-10-17: The fast-MoE layer list when `METRALE_HOLO_FAST_MOE_LAYERS` is unset: every
/// layer (`parse_layer_ranges` reads it as one inclusive range).
pub(super) const ALL_LAYERS: &str = "0-99999";

/// 2026-10-17: The levers `load_layers` reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Qwen35LoadLevers {
    /// 2026-10-17: `METRALE_HOLO_LOW_MEMORY_MOE`: the low-memory expert layout with a fast-MoE
    /// mode, for a qualifying MoE.
    pub(super) low_memory_moe: bool,
    /// 2026-10-17: `METRALE_HOLO_FAST_MOE_LAYERS`, else `ALL_LAYERS`.
    pub(super) fast_moe_layers: String,
    /// 2026-10-17: `METRALE_HOLO_NATIVE_FP8_SSM`: Holo ModelOpt SSM projections in native FP8.
    pub(super) native_fp8_ssm: bool,
    /// 2026-10-17: `METRALE_HOLO_NATIVE_FP8_ATTN=1`: Holo ModelOpt attention in native FP8.
    pub(super) native_fp8_attn: bool,
    /// 2026-10-17: `METRALE_FORCE_NVFP4_ALL`.
    pub(super) force_nvfp4_all: bool,
    /// 2026-10-17: `METRALE_FORCE_NVFP4_MOE`.
    pub(super) force_nvfp4_moe: bool,
    /// 2026-10-17: `METRALE_HOLO_FP4_PROJ_DECODE=1`.
    pub(super) fp4_proj_decode: bool,
    /// 2026-10-17: `METRALE_HOLO_MOE_GROUPED_CUTLASS` (`shared_levers::grouped_cutlass`).
    pub(super) grouped_cutlass: bool,
    /// 2026-10-17: `METRALE_FP8_DEQUANT_MOE_TO_BF16=1`.
    pub(super) fp8_dequant_moe_to_bf16: bool,
    /// 2026-10-17: `METRALE_FP8_DEQUANT_ATTN_TO_BF16=1`.
    pub(super) fp8_dequant_attn_to_bf16: bool,
    /// 2026-10-17: `METRALE_FP8_DEQUANT_LAYERS`, the layer list both dequants are limited to.
    pub(super) fp8_dequant_layers: Option<String>,
    /// 2026-10-17: `METRALE_UNIFIED_MOE_LAYOUT` (`shared_levers::unified_moe_layout`), which
    /// the fast-MoE `unified` mode requires.
    pub(super) unified_moe_layout: bool,
}

impl Qwen35LoadLevers {
    /// 2026-10-17: Resolve from the startup settings.
    pub(super) fn from_settings(s: &InferenceServerSettings) -> Self {
        let k = &s.kernels;
        let is_one = |v: &Option<String>| v.as_deref() == Some("1");
        Self {
            low_memory_moe: k.holo_low_memory_moe,
            fast_moe_layers: k
                .holo_fast_moe_layers
                .clone()
                .unwrap_or_else(|| ALL_LAYERS.to_string()),
            native_fp8_ssm: k.holo_native_fp8_ssm,
            native_fp8_attn: is_one(&k.holo_native_fp8_attn),
            force_nvfp4_all: k.force_nvfp4_all,
            force_nvfp4_moe: k.force_nvfp4_moe,
            fp4_proj_decode: is_one(&k.holo_fp4_proj_decode),
            grouped_cutlass: grouped_cutlass(s),
            fp8_dequant_moe_to_bf16: is_one(&k.fp8_dequant_moe_to_bf16),
            fp8_dequant_attn_to_bf16: is_one(&k.fp8_dequant_attn_to_bf16),
            fp8_dequant_layers: k.fp8_dequant_layers.clone(),
            unified_moe_layout: unified_moe_layout(s),
        }
    }
}

#[cfg(test)]
#[path = "levers_tests.rs"]
mod tests;
