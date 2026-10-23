// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Wrap and reset of a cumulative counter, at 64 bits and a narrower width.

use super::*;

#[test]
fn a_64_bit_wrap_is_a_small_delta_and_a_drop_is_a_reset() {
    let mut t = CounterTracker::default();
    assert_eq!(t.advance(u64::MAX - 5), Advance::First);
    assert_eq!(t.advance(4), Advance::Wrapped(10));
    assert_eq!(t.advance(10), Advance::Delta(6));
    // 2026-10-22: Backwards from the middle of the range: the driver restarted.
    let mut t = CounterTracker::default();
    t.advance(1 << 40);
    assert_eq!(t.advance(7), Advance::Reset(7));
}

#[test]
fn a_32_bit_counter_wraps_at_its_own_width() {
    let mut t = CounterTracker::with_width(32);
    t.advance(u64::from(u32::MAX) - 1);
    assert_eq!(t.advance(3), Advance::Wrapped(5));
    // 2026-10-22: The same step on a 64-bit tracker is a reset, not a wrap.
    let mut t = CounterTracker::default();
    t.advance(u64::from(u32::MAX) - 1);
    assert_eq!(t.advance(3), Advance::Reset(3));
}
