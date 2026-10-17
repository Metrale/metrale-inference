// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `SystemClock` monotonic ordering and a sane wall clock.

use metrale_abstract_io::Clock;

use super::SystemClock;

#[test]
fn monotonic_ns_never_decreases() {
    let clock = SystemClock::new();
    let mut last = clock.monotonic_ns();
    for _ in 0..1_000 {
        let now = clock.monotonic_ns();
        assert!(now >= last, "monotonic went back: {now} < {last}");
        last = now;
    }
}

#[test]
fn wall_clock_is_after_the_epoch() {
    assert!(SystemClock::new().now_unix_ms() > 0);
}
