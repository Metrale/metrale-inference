// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `ManualClock` time movement and timer firing, polled by hand.

use std::future::Future;
use std::pin::pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};

use metrale_abstract_io::Clock;

use super::ManualClock;

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

#[test]
fn advance_moves_wall_and_monotonic_time() {
    let clock = ManualClock::new(1_000);
    assert_eq!(clock.now_unix_ms(), 1_000);
    assert_eq!(clock.monotonic_ns(), 0);
    clock.advance(250);
    assert_eq!(clock.now_unix_ms(), 1_250);
    assert_eq!(clock.monotonic_ns(), 250_000_000);
    let other = clock.handle();
    other.advance(1);
    assert_eq!(clock.monotonic_ns(), 251_000_000, "handles share one clock");
}

#[test]
fn set_now_moves_wall_only_and_fires_nothing() {
    let clock = ManualClock::new(0);
    let log = Arc::new(Mutex::new(Vec::new()));
    let waker = recorder(1, &log);
    let mut cx = Context::from_waker(&waker);
    let mut sleep = pin!(clock.sleep_ms(5));
    assert_eq!(sleep.as_mut().poll(&mut cx), Poll::Pending);
    clock.set_now(i64::MAX);
    assert_eq!(clock.now_unix_ms(), i64::MAX);
    assert_eq!(clock.monotonic_ns(), 0);
    assert!(log.lock().expect("log poisoned").is_empty());
    assert_eq!(sleep.as_mut().poll(&mut cx), Poll::Pending);
    assert_eq!(clock.pending_timers(), 1);
}

#[test]
fn sleep_zero_is_ready_on_first_poll() {
    let clock = ManualClock::new(0);
    let mut cx = Context::from_waker(Waker::noop());
    let mut sleep = pin!(clock.sleep_ms(0));
    assert_eq!(sleep.as_mut().poll(&mut cx), Poll::Ready(()));
    assert_eq!(clock.pending_timers(), 0);
}

#[test]
fn sleep_does_not_complete_early() {
    let clock = ManualClock::new(0);
    let log = Arc::new(Mutex::new(Vec::new()));
    let waker = recorder(10, &log);
    let mut cx = Context::from_waker(&waker);
    let mut sleep = pin!(clock.sleep_ms(10));
    assert_eq!(sleep.as_mut().poll(&mut cx), Poll::Pending);
    clock.advance(9);
    assert!(log.lock().expect("log poisoned").is_empty());
    assert_eq!(sleep.as_mut().poll(&mut cx), Poll::Pending);
    clock.advance(1);
    assert_eq!(*log.lock().expect("log poisoned"), [10]);
    assert_eq!(sleep.as_mut().poll(&mut cx), Poll::Ready(()));
    assert_eq!(clock.pending_timers(), 0);
}

#[test]
fn advance_wakes_due_timers_in_deadline_order() {
    let clock = ManualClock::new(0);
    let log = Arc::new(Mutex::new(Vec::new()));
    let (w30, w10, w20, w40) = (
        recorder(30, &log),
        recorder(10, &log),
        recorder(20, &log),
        recorder(40, &log),
    );
    let mut s30 = pin!(clock.sleep_ms(30));
    let mut s10 = pin!(clock.sleep_ms(10));
    let mut s20 = pin!(clock.sleep_ms(20));
    let mut s40 = pin!(clock.sleep_ms(40));
    assert!(
        s30.as_mut()
            .poll(&mut Context::from_waker(&w30))
            .is_pending()
    );
    assert!(
        s10.as_mut()
            .poll(&mut Context::from_waker(&w10))
            .is_pending()
    );
    assert!(
        s20.as_mut()
            .poll(&mut Context::from_waker(&w20))
            .is_pending()
    );
    assert!(
        s40.as_mut()
            .poll(&mut Context::from_waker(&w40))
            .is_pending()
    );
    assert_eq!(clock.pending_timers(), 4);
    clock.advance(30);
    assert_eq!(*log.lock().expect("log poisoned"), [10, 20, 30]);
    assert_eq!(clock.pending_timers(), 1, "the 40 ms sleep is not due");
    assert!(s10.as_mut().poll(&mut Context::from_waker(&w10)).is_ready());
    assert!(s20.as_mut().poll(&mut Context::from_waker(&w20)).is_ready());
    assert!(s30.as_mut().poll(&mut Context::from_waker(&w30)).is_ready());
    assert!(
        s40.as_mut()
            .poll(&mut Context::from_waker(&w40))
            .is_pending()
    );
}

#[test]
fn dropping_a_pending_sleep_removes_its_timer() {
    let clock = ManualClock::new(0);
    {
        let mut cx = Context::from_waker(Waker::noop());
        let mut sleep = Box::pin(clock.sleep_ms(5));
        assert!(sleep.as_mut().poll(&mut cx).is_pending());
        assert_eq!(clock.pending_timers(), 1);
    }
    assert_eq!(clock.pending_timers(), 0);
}
