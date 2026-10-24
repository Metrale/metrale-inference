// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `TokioTimer` deadlines against `SystemClock` on a current-thread runtime.

use std::time::Duration;

use metrale_abstract_io::{Clock, DurNs, MonoNs, Timer};

use super::TokioTimer;
use crate::SystemClock;

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("current-thread runtime with time")
}

/// Catches: a deadline read from a different origin than the clock's, or a relative sleep that
/// ends before the absolute deadline. The 5 s timeout catches a unit slip (ns read as ms makes
/// the 20 ms wait last 5.5 hours).
#[test]
fn sleep_until_ends_at_or_after_the_deadline_and_promptly() {
    let clock = SystemClock::new();
    let deadline = clock.monotonic_ns().plus(DurNs::from_ms(20));
    runtime().block_on(async {
        tokio::time::timeout(Duration::from_secs(5), TokioTimer.sleep_until(deadline))
            .await
            .expect("a 20 ms sleep finishes within 5 s");
    });
    assert!(clock.monotonic_ns() >= deadline);
}

/// Catches: a past deadline that waits (e.g. one read as a duration from now). The origin is at
/// least 200 ms old, so such a mutant sleeps at least 100 ms for `half` and overruns the 50 ms
/// timeout; a correct timer is ready at once.
#[test]
fn a_past_deadline_returns_without_waiting_for_it() {
    let clock = SystemClock::new();
    std::thread::sleep(Duration::from_millis(200));
    let half = MonoNs(clock.monotonic_ns().0 / 2);
    runtime().block_on(async {
        tokio::time::timeout(Duration::from_millis(50), async {
            TokioTimer.sleep_until(half).await;
            TokioTimer.sleep_until(MonoNs(0)).await;
        })
        .await
        .expect("past deadlines complete at once");
    });
}
