// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: Which `FfnArm` a multi-sequence decode step takes, as a pure
//! function of the layer's capabilities at that batch width. Two ladders exist,
//! each pinned by its tests: the GDN layer's (`select_ssm_arm`) and the attention
//! layer's (`select_attn_arm`). They differ in order and in the arms they know,
//! and this module states both; merging them is a routing change for every
//! served model and needs its own measurement.
//!
//! Owner: model-layers.
//! Invariants:
//! - The selectors read nothing but `LadderCaps`: no environment, no levers, no
//!   layer. The layers evaluate the caps (`ssm_ffn_caps`, `attn_ffn_caps`).
//! - A ladder's order is the order its `if`/`match` chain had before this module
//!   existed; a reordering is a route change and fails the tests.

use super::ffn_rows::FfnArm;

/// 2026-10-13: What the FFN can do for `n` rows, evaluated by the layer.
/// Every predicate is already conjoined with its own preconditions (the
/// `!force_seq` guards of the attention ladder included), so a selector only
/// orders them.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LadderCaps {
    pub n: usize,
    /// 2026-10-13: The attention ladder's "every batched arm is skipped" (an MLA layer).
    pub force_seq: bool,
    /// 2026-10-13: The attention ladder's opt-in grouped routed arm is on and eligible.
    pub use_grouped: bool,
    pub dense_fixed: bool,
    pub nvfp4_grouped: bool,
    /// 2026-10-13: Row-invariant tier policy and the FP8 per-row grouped routing serves `n`.
    pub fp8_row_invariant: bool,
    pub is_dense: bool,
    /// 2026-10-13: The GDN ladder's dense `forward_prefill` arm: `n` at or above its
    /// minimum, a dense FFN, the arm not disabled.
    pub ssm_dense_prefill: bool,
    /// 2026-10-13: The `Km` arm's chunk when `can_forward_km` holds for it: 8 on the GDN
    /// ladder (chunks of at most 8), `n` on the attention ladder (one call).
    pub km_chunk: Option<usize>,
    pub fp8_grouped: bool,
    /// 2026-10-13: `moe_grouped_decode_for(n)`.
    pub moe_grouped: bool,
    /// 2026-10-13: `n == 4` and the atomic C4 arm is opted in.
    pub atomic_c4: bool,
    /// 2026-10-13: Token-major decode is the default (the legacy per-token lever is off).
    pub token_major: bool,
    /// 2026-10-13: The pairwise `forward_k2` walk is on.
    pub pairwise: bool,
    /// 2026-10-13: `METRALE_MOE_BATCHED_DECODE=1`.
    pub batched: bool,
}

/// 2026-10-13: The GDN layer's ladder (`qwen3_ssm/trait_decode_multi_seq.rs`).
pub(crate) fn select_ssm_arm(c: &LadderCaps) -> FfnArm {
    if c.dense_fixed {
        FfnArm::DenseFixed
    } else if c.nvfp4_grouped {
        FfnArm::Nvfp4Grouped
    } else if c.n >= 2 && c.fp8_row_invariant {
        FfnArm::Fp8RowInvariant
    } else if c.n == 2 {
        FfnArm::K2
    } else if c.n == 3 {
        FfnArm::K3
    } else if c.ssm_dense_prefill {
        FfnArm::Prefill
    } else if let (true, Some(chunk)) = (c.n >= 4, c.km_chunk) {
        FfnArm::Km { chunk }
    } else if c.fp8_grouped {
        FfnArm::Fp8Grouped
    } else if c.moe_grouped {
        FfnArm::Prefill
    } else if c.atomic_c4 {
        FfnArm::AtomicC4
    } else if c.token_major {
        FfnArm::TokenMajor
    } else if c.batched {
        FfnArm::Batched
    } else {
        FfnArm::PerRow
    }
}

/// 2026-10-13: The attention layer's ladder
/// (`qwen3_attention/trait_impl/multi_seq/ffn.rs`). `force_seq` skips every
/// batched arm but the `batched` opt-in, which the final branch still honours.
pub(crate) fn select_attn_arm(c: &LadderCaps) -> FfnArm {
    let open = !c.force_seq;
    if open && c.dense_fixed {
        FfnArm::DenseFixed
    } else if open && c.nvfp4_grouped {
        FfnArm::Nvfp4Grouped
    } else if open && c.n >= 2 && c.fp8_row_invariant {
        FfnArm::Fp8RowInvariant
    } else if open && !c.use_grouped && c.n == 3 {
        FfnArm::K3
    } else if open && !c.use_grouped && c.n == 2 {
        FfnArm::K2
    } else if let (true, Some(chunk)) = (open, c.km_chunk) {
        FfnArm::Km { chunk }
    } else if open && c.fp8_grouped {
        FfnArm::Fp8Grouped
    } else if open && (c.is_dense || c.moe_grouped) {
        FfnArm::Prefill
    } else if open && c.use_grouped {
        FfnArm::Prefill
    } else if open && c.n > 2 && c.n.is_multiple_of(2) && c.pairwise {
        FfnArm::PairwiseK2
    } else if c.batched {
        FfnArm::Batched
    } else {
        FfnArm::PerRow
    }
}
