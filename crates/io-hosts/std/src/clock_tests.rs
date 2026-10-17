// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `SystemClock` monotonic ordering, one process origin, and a sane wall clock.

use std::time::Duration;

use metrale_abstract_io::Clock;

use super::SystemClock;

#[test]
fn monotonic_ns_never_decreases() {
    let clock = SystemClock::new();
    let mut last = clock.monotonic_ns();
    for _ in 0..1_000 {
        let now = clock.monotonic_ns();
        assert!(now >= last, "monotonic went back: {now:?} < {last:?}");
        last = now;
    }
}

/// Catches: a per-instance origin. The later clock would then read about 0 while the earlier
/// one reads at least 2 ms, so a reading taken after the other would compare smaller.
#[test]
fn two_clocks_share_one_process_origin() {
    let early = SystemClock::new();
    std::thread::sleep(Duration::from_millis(2));
    let late = SystemClock::new();
    let first = early.monotonic_ns();
    let second = late.monotonic_ns();
    assert!(
        second >= first,
        "clocks disagree: {second:?} read after {first:?}"
    );
}

#[test]
fn wall_clock_is_after_the_epoch() {
    assert!(SystemClock::new().now_unix_ms().0 > 0);
}
