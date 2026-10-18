// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Row threshold and switches for the grouped-GEMM MoE decode arm.
//!
//! Owner: model-layers (MoE).
//! Invariants:
//! - `moe_grouped_decode_decide` is false at every width while the arm is
//!   disabled, even when it is forced. 2026-10-17: The two switches are
//!   `MoeLevers::grouped_decode` and `MoeLevers::grouped_decode_forced`, and
//!   `MoeLevers::grouped_decode_for` decides through this function.

/// 2026-09-25: Minimum rows for the grouped-GEMM MoE decode arm. The SSM stack
/// (`qwen3_ssm::trait_decode_multi_seq`) and the attention layers' multi-seq FFN
/// both decide through `MoeLevers::grouped_decode_for`, so they share this width. The
/// arm reads each routed expert once for all rows but first sorts and permutes
/// the rows, a fixed cost that the threshold keeps off narrow batches.
pub fn moe_grouped_decode_min_rows() -> usize {
    16
}

/// 2026-09-25: Whether the arm runs for `n` rows, given the two switches
/// (`enabled` off when `METRALE_NO_MOE_GROUPED_DECODE` is set; `forced` by
/// `METRALE_MOE_GROUPED_DECODE`, for measuring narrow widths).
pub fn moe_grouped_decode_decide(n: usize, enabled: bool, forced: bool) -> bool {
    enabled && (n >= moe_grouped_decode_min_rows() || forced)
}

#[cfg(test)]
#[path = "moe_grouped_decode_tests.rs"]
mod moe_grouped_decode_tests;
