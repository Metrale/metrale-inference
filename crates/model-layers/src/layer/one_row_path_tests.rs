// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Tests for `one_row_path::takes_decode_path` and
//! `one_row_path::split_segment_takes_decode_path`.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.

use super::one_row_path::{split_segment_takes_decode_path, takes_decode_path};

/// 2026-10-14: The 1k bit probe's shapes: a 1073-token prompt whose tail split falls at 1072.
const PROBE_TOTAL: usize = 1073;
const PROBE_CUT: usize = 1072;

/// 2026-10-14: Lever off: the decision is the replaced `rows == 1 && start > 0` on every input,
/// whatever the model's layers answer, so the default serve's bits are unchanged.
#[test]
fn with_the_lever_off_the_decision_is_the_replaced_one() {
    for rows in 0..4 {
        for start in [0usize, 1, 1072, 8208] {
            for splits in [false, true] {
                assert_eq!(
                    takes_decode_path(rows, start, false, splits),
                    rows == 1 && start > 0,
                    "rows {rows} start {start} splits {splits}"
                );
            }
        }
    }
}

/// 2026-10-14: Lever on: a splitting model's one-row pass after position 0 (the 1k warm
/// replay at 1072) takes the prefill path, as the cold pass's in-pass tail segment does; a
/// non-splitting model keeps the decode path.
#[test]
fn with_the_lever_on_only_a_splitting_models_one_row_pass_leaves_the_decode_path() {
    assert!(!takes_decode_path(1, 1072, true, true));
    assert!(takes_decode_path(1, 1072, true, false));
}

/// 2026-10-14: A pass of several rows, or one at position 0, takes the prefill path under
/// every lever and model.
#[test]
fn several_rows_or_position_zero_never_take_the_decode_path() {
    for (one_row, splits) in [(false, false), (false, true), (true, false), (true, true)] {
        assert!(!takes_decode_path(2, 1072, one_row, splits));
        assert!(!takes_decode_path(1073, 0, one_row, splits));
        assert!(!takes_decode_path(1, 0, one_row, splits));
    }
}

/// 2026-10-14: `ONE_ROW_TAIL_DECODE` off: no segment takes the decode path, on any input, so the
/// GLM segment loop runs every sub-chunk as before and the default serve's bits are unchanged.
#[test]
fn with_the_tail_lever_off_no_segment_takes_the_decode_path() {
    for rows in 0..4 {
        for start in [0usize, 1, PROBE_CUT, 8208] {
            for split in [false, true] {
                for one_row_prefill in [false, true] {
                    assert!(
                        !split_segment_takes_decode_path(
                            rows,
                            start,
                            split,
                            false,
                            one_row_prefill
                        ),
                        "rows {rows} start {start} split {split} one_row_prefill {one_row_prefill}"
                    );
                }
            }
        }
    }
}

/// 2026-10-14: The 1k probe's cold pass is split at 1072 into [0, 1072) and [1072, 1073); its
/// warm replay is one pass of one row at 1072. Under the tail lever the one-row segment takes
/// the path the replay's pass takes (decode), the 1072-row segment does not. Without it they
/// differ: the defect.
#[test]
fn the_probe_tail_segment_and_its_replay_take_the_same_path_only_under_the_tail_lever() {
    let replay = takes_decode_path(1, PROBE_CUT, false, true);
    assert!(replay, "the warm one-row replay runs the decode path");
    let tail_rows = PROBE_TOTAL - PROBE_CUT;
    assert!(split_segment_takes_decode_path(
        tail_rows, PROBE_CUT, true, true, false
    ));
    assert!(!split_segment_takes_decode_path(
        PROBE_CUT, 0, true, true, false
    ));
    // 2026-10-14: The negative control: the default rule leaves the pair on different paths.
    assert_ne!(
        split_segment_takes_decode_path(tail_rows, PROBE_CUT, true, false, false),
        replay
    );
}

/// 2026-10-14: Under the tail lever a split segment takes the decode path exactly when a
/// separate pass of the same rows from the same position would in a splitting layer, with
/// `ONE_ROW_PREFILL` off and on: the "one-row segment and one-row pass agree" rule. With
/// `ONE_ROW_PREFILL` on both take the prefill path.
#[test]
fn under_the_tail_lever_a_split_segment_matches_a_separate_pass_of_its_rows() {
    for rows in 0..4 {
        for start in [0usize, 1, PROBE_CUT, 8208] {
            for one_row_prefill in [false, true] {
                assert_eq!(
                    split_segment_takes_decode_path(rows, start, true, true, one_row_prefill),
                    takes_decode_path(rows, start, one_row_prefill, true),
                    "rows {rows} start {start} one_row_prefill {one_row_prefill}"
                );
            }
        }
    }
    assert!(!split_segment_takes_decode_path(
        1, PROBE_CUT, true, true, true
    ));
}

/// 2026-10-14: A pass of one segment is the whole pass, which the model routes itself: the
/// segment rule never answers for it, even a one-row pass after position 0 under both levers'
/// settings, so a pass the model sent down the prefill path stays there.
#[test]
fn an_unsplit_pass_is_never_rerouted_by_the_segment_rule() {
    for one_row_prefill in [false, true] {
        assert!(!split_segment_takes_decode_path(
            1,
            PROBE_CUT,
            false,
            true,
            one_row_prefill
        ));
    }
}

/// 2026-10-14: A one-row segment at position 0 (a capture point at local row 1 of a pass from
/// position 0) keeps the prefill path: a separate pass of that row at position 0 is a prefill.
#[test]
fn a_one_row_segment_at_position_zero_keeps_the_prefill_path() {
    assert!(!split_segment_takes_decode_path(1, 0, true, true, false));
    assert!(split_segment_takes_decode_path(1, 1, true, true, false));
}
