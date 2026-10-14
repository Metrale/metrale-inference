// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Tests for `one_row_path::takes_decode_path`.
//!
//! Owner: model-engine prefill.
//! Invariants: none beyond the types.

use super::one_row_path::takes_decode_path;

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
