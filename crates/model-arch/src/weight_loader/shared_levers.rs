// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The loader levers that more than one model family reads, each with one predicate
//! over its settings field, so two families cannot read the same variable two ways.
//!
//! Owner: model-arch weight loaders.
//! Invariants: each function reads one settings field and nothing else; a loader calls it once
//! per load and keeps the answer.

use metrale_settings::InferenceServerSettings;

/// 2026-10-17: `METRALE_HOLO_MOE_GROUPED_CUTLASS` is `1` or `true`: build the CUTLASS grouped
/// NVFP4 scale tables at load (`build_cutlass_grouped_sfb`). Read by the qwen35 fast-MoE layers,
/// qwen4_exp and laguna.
pub(crate) fn grouped_cutlass(s: &InferenceServerSettings) -> bool {
    matches!(
        s.kernels.holo_moe_grouped_cutlass.as_deref(),
        Some("1" | "true")
    )
}

/// 2026-10-17: `METRALE_UNIFIED_MOE_LAYOUT` is `1` or `true` in any case: the unified transposed
/// MoE layout. Read by the qwen35 fast-MoE `unified` mode and laguna.
pub(crate) fn unified_moe_layout(s: &InferenceServerSettings) -> bool {
    s.kernels
        .unified_moe_layout
        .as_deref()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// 2026-10-17: `METRALE_GDN_BF16_WEIGHTS=1`: GDN keeps BF16 projections. Read by the qwen35
/// NVFP4 GDN arm (`out_proj_dense`) and qwen35_dense (`gdn_dequant`).
pub(crate) fn gdn_bf16_weights(s: &InferenceServerSettings) -> bool {
    s.kernels.gdn_bf16_weights.as_deref() == Some("1")
}

/// 2026-10-17: `METRALE_NVFP4_MLA` is `0`, `false`, `no` or `off` (trimmed, any case): the MLA
/// projections stay BF16. Read by the Mistral MLA and LongCat loaders.
pub(crate) fn nvfp4_mla_disabled(s: &InferenceServerSettings) -> bool {
    s.kernels.nvfp4_mla.as_deref().is_some_and(|v| {
        let v = v.trim().to_ascii_lowercase();
        matches!(v.as_str(), "0" | "false" | "no" | "off")
    })
}

#[cfg(test)]
#[path = "shared_levers_tests.rs"]
mod tests;
