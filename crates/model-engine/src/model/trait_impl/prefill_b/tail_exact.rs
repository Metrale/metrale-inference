// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Which prefill passes run the exact GDN recurrence arm
//! (`ForwardContext::gdn_exact_replay`) instead of the FLA chunked kernel.
//!
//! A warm repeat of a prompt restores the SSM snapshot saved at the tail split point
//! (`prefill_plan::tail_split_point`) and replays the tokens from there to the end of
//! the prompt on the exact arm. Without an in-pass capture (`inpass_cut_capture_supported`)
//! the cold request computes the same tokens in a pass of its own that starts at that
//! point, and that pass ran the FLA arm: a different reduction tree, so cold and warm
//! gave the last prompt rows different bits (the reference Hopper box, Flash-Next, 32k
//! prompt: max |d| 4e-3 at layer 0's block output, first token differs). The cold pass
//! now takes the arm the warm replay takes.
//!
//! Owner: model-engine prefill.
//! Invariants: a pass over the same tokens from the same SSM state takes the same GDN
//! arm cold and warm.

use std::sync::OnceLock;

use super::super::super::types::TransformerModel;

/// 2026-10-14: `METRALE_GDN_TAIL_EXACT` not equal to `"0"`, read once per process. `0`
/// restores the FLA arm on the cold tail pass.
pub(in crate::model) fn gdn_tail_exact_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_GDN_TAIL_EXACT").as_deref() != Ok("0"))
}

/// 2026-10-14: Whether a pass starting at `chunk_start` runs the exact GDN arm: after a
/// snapshot restore (`marconi_skip`), or, with `tail_exact` on, when it starts at the
/// prompt's tail split point `split`, where a warm repeat restores and replays.
pub(in crate::model) fn pass_takes_exact_gdn(
    marconi_skip: bool,
    tail_exact: bool,
    chunk_start: usize,
    split: Option<usize>,
) -> bool {
    marconi_skip || (tail_exact && split == Some(chunk_start))
}

impl TransformerModel {
    /// 2026-10-14: `pass_takes_exact_gdn` for this model and prompt; `bs` is the KV block
    /// size, passed by a caller holding the `kv_cache` lock.
    pub(in crate::model) fn prefill_gdn_exact_pass(
        &self,
        tokens: &[u32],
        chunk_start: usize,
        bs: usize,
        marconi_skip: bool,
    ) -> bool {
        pass_takes_exact_gdn(
            marconi_skip,
            gdn_tail_exact_enabled(),
            chunk_start,
            self.prefill_tail_split_at(tokens, bs),
        )
    }
}
