// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `ManualClock` time movement and `ManualTimer` firing, polled by hand (`LocalExec`
//! arrives in 3a). Each test names the mutation of `clock.rs` it catches.

use std::future::Future;
use std::pin::pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};

use metrale_abstract_io::{Clock, DurNs, MonoNs, Timer, UnixMs};

use super::ManualClock;

const MS: u64 = 1_000_000;

struct Recorder {
    tag: u64,
    log: Arc<Mutex<Vec<u64>>>,
}

impl Wake for Recorder {
    fn wake(self: Arc<Self>) {
        self.log.lock().expect("log poisoned").push(self.tag);
    }
}

fn recorder(tag: u64, log: &Arc<Mutex<Vec<u64>>>) -> Waker {
    Waker::from(Arc::new(Recorder {
        tag,
        log: Arc::clone(log),
    }))
}

fn woken(log: &Arc<Mutex<Vec<u64>>>) -> Vec<u64> {
    log.lock().expect("log poisoned").clone()
}

/// Catches: wall time not moved, or moved by nanoseconds instead of crossed milliseconds.
#[test]
fn advance_ns_moves_monotonic_and_wall_by_crossed_milliseconds() {
    let clock = ManualClock::new(1_000);
    assert_eq!(clock.now_unix_ms(), UnixMs(1_000));
    assert_eq!(clock.monotonic_ns(), MonoNs(0));
    clock.advance_ns(DurNs(600_000));
    assert_eq!(clock.monotonic_ns(), MonoNs(600_000));
    assert_eq!(
        clock.now_unix_ms(),
        UnixMs(1_000),
        "no millisecond crossed yet"
    );
    clock.advance_ns(DurNs(600_000));
    assert_eq!(
        clock.now_unix_ms(),
        UnixMs(1_001),
        "1.2 ms crosses one millisecond"
    );
    clock.handle().advance_ns(DurNs::from_ms(250));
    assert_eq!(
        clock.monotonic_ns(),
        MonoNs(251_200_000),
        "handles share one clock"
    );
    assert_eq!(clock.now_unix_ms(), UnixMs(1_251));
}

/// Catches: a timer woken in registration order instead of deadline order, or a not-yet-due
/// timer woken early.
#[test]
fn timers_fire_in_deadline_order() {
    let clock = ManualClock::new(0);
    let timer = clock.timer();
    let log = Arc::new(Mutex::new(Vec::new()));
    let wakers = [30, 10, 20, 40].map(|tag| recorder(tag, &log));
    let mut s30 = pin!(timer.sleep_until(MonoNs(30 * MS)));
    let mut s10 = pin!(timer.sleep_until(MonoNs(10 * MS)));
    let mut s20 = pin!(timer.sleep_until(MonoNs(20 * MS)));
    let mut s40 = pin!(timer.sleep_until(MonoNs(40 * MS)));
    let cx = |i: usize| Context::from_waker(&wakers[i]);
    assert!(s30.as_mut().poll(&mut cx(0)).is_pending());
    assert!(s10.as_mut().poll(&mut cx(1)).is_pending());
    assert!(s20.as_mut().poll(&mut cx(2)).is_pending());
    assert!(s40.as_mut().poll(&mut cx(3)).is_pending());
    assert_eq!(clock.pending_timers(), 4);
    clock.advance_ns(DurNs::from_ms(30));
    assert_eq!(woken(&log), [10, 20, 30]);
    assert_eq!(clock.pending_timers(), 1, "the 40 ms sleep is not due");
    assert!(s10.as_mut().poll(&mut cx(1)).is_ready());
    assert!(s20.as_mut().poll(&mut cx(2)).is_ready());
    assert!(s30.as_mut().poll(&mut cx(0)).is_ready());
    assert!(s40.as_mut().poll(&mut cx(3)).is_pending());
}

/// Catches: a sleep completing early (after the first 10 ms), or a timer that stays registered
/// and fires again on the second advance.
#[test]
fn a_15_ms_sleeper_across_two_10_ms_advances_wakes_once() {
    let clock = ManualClock::new(0);
    let timer = clock.timer();
    let log = Arc::new(Mutex::new(Vec::new()));
    let waker = recorder(15, &log);
    let mut cx = Context::from_waker(&waker);
    let mut sleep = pin!(timer.sleep_until(MonoNs(15 * MS)));
    assert_eq!(sleep.as_mut().poll(&mut cx), Poll::Pending);
    clock.advance_ns(DurNs::from_ms(10));
    assert!(woken(&log).is_empty(), "not due at 10 ms");
    assert_eq!(sleep.as_mut().poll(&mut cx), Poll::Pending);
    clock.advance_ns(DurNs::from_ms(10));
    assert_eq!(woken(&log), [15]);
    assert_eq!(clock.pending_timers(), 0);
    clock.advance_ns(DurNs::from_ms(10));
    assert_eq!(woken(&log), [15], "a fired timer never fires again");
    assert_eq!(sleep.as_mut().poll(&mut cx), Poll::Ready(()));
}

/// Catches: `set_now` moving monotonic time or waking timers.
#[test]
fn set_now_moves_wall_only_and_wakes_nothing() {
    let clock = ManualClock::new(0);
    let timer = clock.timer();
    let log = Arc::new(Mutex::new(Vec::new()));
    let waker = recorder(1, &log);
    let mut cx = Context::from_waker(&waker);
    let mut sleep = pin!(timer.sleep_until(MonoNs(5)));
    assert_eq!(sleep.as_mut().poll(&mut cx), Poll::Pending);
    clock.set_now(UnixMs(i64::MAX));
    assert_eq!(clock.now_unix_ms(), UnixMs(i64::MAX));
    assert_eq!(clock.monotonic_ns(), MonoNs(0));
    assert!(woken(&log).is_empty());
    assert_eq!(sleep.as_mut().poll(&mut cx), Poll::Pending);
    assert_eq!(clock.pending_timers(), 1);
}

/// Catches: a reached deadline that registers a timer and returns `Pending` (it would never
/// wake: nothing moves the clock). `Waker::noop()` makes a missed wake invisible, so only the
/// first-poll result can pass.
#[test]
fn a_past_deadline_is_ready_on_first_poll() {
    let clock = ManualClock::new(0);
    clock.advance_ns(DurNs(100));
    let timer = clock.timer();
    let mut cx = Context::from_waker(Waker::noop());
    assert_eq!(
        pin!(timer.sleep_until(MonoNs(40))).as_mut().poll(&mut cx),
        Poll::Ready(())
    );
    assert_eq!(
        pin!(timer.sleep_until(MonoNs(100))).as_mut().poll(&mut cx),
        Poll::Ready(()),
        "a deadline equal to now is reached"
    );
    assert_eq!(clock.pending_timers(), 0);
}

/// Catches: `set_monotonic` accepting a regression (replay must never move time back).
#[test]
#[should_panic(expected = "is before the current")]
fn backwards_set_monotonic_panics() {
    let clock = ManualClock::new(0);
    clock.set_monotonic(MonoNs(1_000));
    clock.set_monotonic(MonoNs(999));
}

/// Catches: `set_monotonic` that moves time without waking due timers, or moves the wall clock.
#[test]
fn set_monotonic_wakes_due_timers_and_leaves_wall_time() {
    let clock = ManualClock::new(7);
    let timer = clock.timer();
    let log = Arc::new(Mutex::new(Vec::new()));
    let (w1, w2) = (recorder(1, &log), recorder(2, &log));
    let mut s1 = pin!(timer.sleep_until(MonoNs(500)));
    let mut s2 = pin!(timer.sleep_until(MonoNs(501)));
    assert!(s1.as_mut().poll(&mut Context::from_waker(&w1)).is_pending());
    assert!(s2.as_mut().poll(&mut Context::from_waker(&w2)).is_pending());
    clock.set_monotonic(MonoNs(500));
    assert_eq!(woken(&log), [1]);
    assert_eq!(clock.now_unix_ms(), UnixMs(7));
    clock.set_monotonic(MonoNs(500));
    assert_eq!(
        woken(&log),
        [1],
        "setting the same time again wakes nothing"
    );
}

/// Catches: a dropped sleep that leaves its timer registered.
#[test]
fn dropping_a_pending_sleep_removes_its_timer() {
    let clock = ManualClock::new(0);
    let timer = clock.timer();
    {
        let mut cx = Context::from_waker(Waker::noop());
        let mut sleep = Box::pin(timer.sleep_until(MonoNs(5)));
        assert!(sleep.as_mut().poll(&mut cx).is_pending());
        assert_eq!(clock.pending_timers(), 1);
    }
    assert_eq!(clock.pending_timers(), 0);
}
