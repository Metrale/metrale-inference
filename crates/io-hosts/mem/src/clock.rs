// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: A [`Clock`] that moves only when a test calls `advance` or
//! `set_now`, so time-dependent code runs the same way on every run.
//!
//! Owner: metrale-io-host-mem.
//! Invariants:
//! - Monotonic time starts at 0 ns and moves only through `advance`.
//! - A sleep completes only once monotonic time reaches its deadline, never early.
//! - `advance` wakes due timers in (deadline, registration) order.
//! - `set_now` moves the wall clock only and never fires a timer.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};

use metrale_abstract_io::Clock;

const NS_PER_MS: u64 = 1_000_000;

struct Timer {
    id: u64,
    deadline_ns: u64,
    waker: Waker,
}

struct State {
    now_unix_ms: i64,
    now_ns: u64,
    next_id: u64,
    timers: Vec<Timer>,
}

/// 2026-10-17: A shared manual clock; clones and handles see the same time.
#[derive(Clone)]
pub struct ManualClock {
    state: Arc<Mutex<State>>,
}

impl ManualClock {
    pub fn new(start_unix_ms: i64) -> Self {
        Self {
            state: Arc::new(Mutex::new(State {
                now_unix_ms: start_unix_ms,
                now_ns: 0,
                next_id: 0,
                timers: Vec::new(),
            })),
        }
    }

    /// Another handle on the same clock (same as `clone`).
    pub fn handle(&self) -> Self {
        self.clone()
    }

    /// Sets the wall clock only; monotonic time and timers are untouched.
    pub fn set_now(&self, unix_ms: i64) {
        self.lock().now_unix_ms = unix_ms;
    }

    /// Moves wall and monotonic time forward by `ms`, then wakes every timer
    /// whose deadline has been reached, earliest deadline first.
    pub fn advance(&self, ms: u64) {
        let due = {
            let mut state = self.lock();
            let step_ns = ms
                .checked_mul(NS_PER_MS)
                .expect("ManualClock::advance: step overflows u64 nanoseconds");
            let step_ms =
                i64::try_from(ms).expect("ManualClock::advance: step overflows i64 milliseconds");
            state.now_ns = state
                .now_ns
                .checked_add(step_ns)
                .expect("ManualClock::advance: monotonic time overflows u64");
            state.now_unix_ms = state
                .now_unix_ms
                .checked_add(step_ms)
                .expect("ManualClock::advance: wall time overflows i64");
            let now_ns = state.now_ns;
            let (mut due, pending): (Vec<Timer>, Vec<Timer>) = state
                .timers
                .drain(..)
                .partition(|timer| timer.deadline_ns <= now_ns);
            state.timers = pending;
            due.sort_by_key(|timer| (timer.deadline_ns, timer.id));
            due
        };
        // 2026-10-17: Wake outside the lock; a waker may poll the sleep inline.
        for timer in due {
            timer.waker.wake();
        }
    }

    /// Sleeps that have been polled, are not yet due, and are still alive.
    pub fn pending_timers(&self) -> usize {
        self.lock().timers.len()
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().expect("ManualClock lock poisoned")
    }
}

impl Clock for ManualClock {
    fn now_unix_ms(&self) -> i64 {
        self.lock().now_unix_ms
    }

    fn monotonic_ns(&self) -> u64 {
        self.lock().now_ns
    }

    async fn sleep_ms(&self, ms: u64) {
        let delay_ns = ms
            .checked_mul(NS_PER_MS)
            .expect("ManualClock::sleep_ms: delay overflows u64 nanoseconds");
        let deadline_ns = self
            .lock()
            .now_ns
            .checked_add(delay_ns)
            .expect("ManualClock::sleep_ms: deadline overflows u64 nanoseconds");
        Sleep {
            clock: self,
            deadline_ns,
            id: None,
        }
        .await;
    }
}

struct Sleep<'a> {
    clock: &'a ManualClock,
    deadline_ns: u64,
    id: Option<u64>,
}

impl Future for Sleep<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        let mut state = this.clock.lock();
        if state.now_ns >= this.deadline_ns {
            if let Some(id) = this.id.take() {
                state.timers.retain(|timer| timer.id != id);
            }
            return Poll::Ready(());
        }
        match this.id {
            Some(id) => {
                let timer = state
                    .timers
                    .iter_mut()
                    .find(|timer| timer.id == id)
                    .expect("ManualClock: a pending sleep lost its timer");
                if !timer.waker.will_wake(cx.waker()) {
                    timer.waker = cx.waker().clone();
                }
            }
            None => {
                let id = state.next_id;
                state.next_id = id.checked_add(1).expect("ManualClock: timer ids exhausted");
                state.timers.push(Timer {
                    id,
                    deadline_ns: this.deadline_ns,
                    waker: cx.waker().clone(),
                });
                this.id = Some(id);
            }
        }
        Poll::Pending
    }
}

impl Drop for Sleep<'_> {
    fn drop(&mut self) {
        let Some(id) = self.id else { return };
        // 2026-10-17: While unwinding the lock may already be poisoned, and a second
        // panic would abort and hide the first; the timer list dies with the test.
        if std::thread::panicking() {
            return;
        }
        self.clock.lock().timers.retain(|timer| timer.id != id);
    }
}

#[cfg(test)]
#[path = "clock_tests.rs"]
mod tests;
