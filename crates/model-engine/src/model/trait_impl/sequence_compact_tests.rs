// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The decode-launch slot guard (`first_shared_slot`) on the slot vectors
//! the fn-b24y serve captured.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use super::first_shared_slot;

#[test]
fn the_guard_names_the_slot_two_rows_share() {
    // 2026-10-14: fn-b24y serve log, n=8 at max_slots 8: rows 5 and 6 on slot 5.
    assert_eq!(first_shared_slot([0, 1, 2, 3, 4, 5, 5, 6], 8), Some(5));
    assert_eq!(first_shared_slot([3, 1, 3], 8), Some(3));
}

#[test]
fn the_guard_passes_distinct_slots_and_repeated_padding() {
    // 2026-10-14: n=5 padded to 8 with the dummy slot (index max_slots), as captured.
    assert_eq!(first_shared_slot([0, 1, 2, 3, 4, 8, 8, 8], 8), None);
    assert_eq!(first_shared_slot([0, 3], 8), None);
    assert_eq!(first_shared_slot([usize::MAX, usize::MAX], 8), None);
}
