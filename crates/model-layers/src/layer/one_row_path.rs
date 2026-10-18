// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Whether a prefill pass, or a one-row segment of a split pass, runs each layer's
//! decode path or its prefill path.
//!
//! A one-row pass after position 0 (a prefix-cache hit that replays one token, the last chunk
//! of a prompt one token past a chunk boundary, the matched-equals-total re-run) has run each
//! layer's decode path. A layer that splits its pass at an in-pass capture
//! (`splits_pass_at_capture`, GLM-5.3) runs the same one-row tail of a cold prompt as a
//! one-row segment of its pass, through its prefill path, so a cold prompt and its warm replay
//! differed in the last row's bits. Two levers make the two agree:
//! - `METRALE_ONE_ROW_PREFILL=1` (`ModelLevers::one_row_prefill`) sends such a model's one-row
//!   passes through the prefill path ([`takes_decode_path`]).
//! - `METRALE_ONE_ROW_TAIL_DECODE=1` (`ModelLevers::one_row_tail_decode`) sends the one-row
//!   segment through the decode path instead ([`split_segment_takes_decode_path`]), so a
//!   one-row segment and a one-row pass take the same path.
//!
//! Owner: model-layers (the model's pass and the GLM layer's segment loop both ask here).
//! Invariants:
//! - With both levers off each answer is the one it replaced: a pass `rows == 1 && start > 0`,
//!   a segment never.
//! - A pass or segment of several rows, or one starting at position 0, never takes the decode
//!   path.
//! - `one_row_prefill` takes precedence: with it on, neither a one-row pass nor a one-row
//!   segment of a splitting layer takes the decode path, so the two still agree.

/// 2026-10-14: True when a prefill pass of `rows` rows from absolute position `start` runs each
/// layer's decode path. Pure.
pub fn takes_decode_path(
    rows: usize,
    start: usize,
    one_row_prefill: bool,
    splits_pass_at_capture: bool,
) -> bool {
    rows == 1 && start > 0 && !(one_row_prefill && splits_pass_at_capture)
}

/// 2026-10-14: True when a layer that splits its pass at an in-pass capture runs a segment of
/// `rows` rows from absolute position `start` through its decode path. `split` is whether the
/// pass has more than one segment: a pass of one segment is the whole pass, which the model
/// already routed by [`takes_decode_path`]. Under `tail_decode` the segment takes the path a
/// separate pass of the same rows would ([`takes_decode_path`] for a splitting layer). Pure.
pub fn split_segment_takes_decode_path(
    rows: usize,
    start: usize,
    split: bool,
    tail_decode: bool,
    one_row_prefill: bool,
) -> bool {
    tail_decode && split && takes_decode_path(rows, start, one_row_prefill, true)
}
