// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Whether a prefill pass runs each layer's decode path or its prefill path.
//!
//! A one-row pass after position 0 (a prefix-cache hit that replays one token, the last chunk
//! of a prompt one token past a chunk boundary, the matched-equals-total re-run) has run each
//! layer's decode path. A layer that splits its pass at an in-pass capture
//! (`splits_pass_at_capture`, GLM-5.3) runs the same one-row tail of a cold prompt through its
//! prefill path, so a cold prompt and its warm replay differed in the last row's bits.
//! `METRALE_ONE_ROW_PREFILL=1` (`ModelLevers::one_row_prefill`) sends such a model's one-row
//! passes through the prefill path too; a model whose layers do not split keeps the decode path.
//!
//! Owner: model-engine prefill.
//! Invariants:
//! - With the lever off the answer is the one this replaced: `rows == 1 && start > 0`.
//! - A pass of several rows, or one starting at position 0, never takes the decode path.

/// 2026-10-14: True when a prefill pass of `rows` rows from absolute position `start` runs each
/// layer's decode path. Pure.
pub(super) fn takes_decode_path(
    rows: usize,
    start: usize,
    one_row_prefill: bool,
    splits_pass_at_capture: bool,
) -> bool {
    rows == 1 && start > 0 && !(one_row_prefill && splits_pass_at_capture)
}
