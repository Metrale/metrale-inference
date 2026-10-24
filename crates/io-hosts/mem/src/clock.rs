// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: A [`Clock`] and a [`Timer`] that move only when a test or a replay says so, so
//! time-dependent code runs the same way on every run.
//!
//! Owner: metrale-io-host-mem.
//! Invariants:
//! - Monotonic time starts at 0 ns and moves only through `advance_ns` and `set_monotonic`;
//!   it never goes back (`set_monotonic` panics on a regression).
//! - A sleep completes only once monotonic time reaches its deadline, never early; a deadline
//!   already reached is ready on the first poll and registers nothing.
//! - Moving monotonic time wakes due timers in (deadline, registration) order, outside the lock.
//! - `set_now` moves the wall clock only and never wakes a timer.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};

use metrale_abstract_io::{Clock, DurNs, MonoNs, Timer, UnixMs};

/// 2026-10-17: A registered sleep: its deadline, then its registration number.
type TimerKey = (MonoNs, u64);

struct State {
    wall: UnixMs,
    mono: MonoNs,
    next_id: u64,
    timers: BTreeMap<TimerKey, Waker>,
}

type Shared = Arc<Mutex<State>>;

fn lock(state: &Shared) -> MutexGuard<'_, State> {
    state.lock().expect("ManualClock lock poisoned")
}

/// 2026-10-17: A shared manual clock; clones, handles and its timers see the same time.
#[derive(Clone)]
pub struct ManualClock {
    state: Shared,
}

/// 2026-10-17: The [`Timer`] of one [`ManualClock`]; it fires only when that clock moves.
#[derive(Clone)]
pub struct ManualTimer {
    state: Shared,
}

impl ManualClock {
    /// Wall time starts at `start_unix_ms`, monotonic time at 0 ns, with no timers.
    #[must_use]
    pub fn new(start_unix_ms: i64) -> Self {
        Self {
            state: Arc::new(Mutex::new(State {
                wall: UnixMs(start_unix_ms),
                mono: MonoNs(0),
                next_id: 0,
                timers: BTreeMap::new(),
            })),
        }
    }

    /// Another handle on the same clock (same as `clone`).
    #[must_use]
    pub fn handle(&self) -> Self {
        self.clone()
    }

    /// The timer that fires as this clock moves.
    #[must_use]
    pub fn timer(&self) -> ManualTimer {
        ManualTimer {
            state: Arc::clone(&self.state),
        }
    }

    /// Sets the wall clock only; monotonic time and timers are untouched.
    pub fn set_now(&self, at: UnixMs) {
        lock(&self.state).wall = at;
    }

    /// Moves monotonic time forward by `step`, and wall time by the whole milliseconds that
    /// monotonic time crosses, then wakes every due timer, earliest deadline first.
    pub fn advance_ns(&self, step: DurNs) {
        let due = {
            let mut state = lock(&self.state);
            let to = MonoNs(
                state
                    .mono
                    .0
                    .checked_add(step.0)
                    .expect("ManualClock::advance_ns: monotonic time overflows u64"),
            );
            let crossed_ms = to.0 / NS_PER_MS - state.mono.0 / NS_PER_MS;
            let crossed_ms = i64::try_from(crossed_ms)
                .expect("ManualClock::advance_ns: step overflows i64 milliseconds");
            state.wall = UnixMs(
                state
                    .wall
                    .0
                    .checked_add(crossed_ms)
                    .expect("ManualClock::advance_ns: wall time overflows i64"),
            );
            move_to(&mut state, to)
        };
        wake_all(due);
    }

    /// Sets monotonic time to `at` (trace replay), waking due timers; wall time is untouched.
    /// Panics if `at` is earlier than the current monotonic time.
    pub fn set_monotonic(&self, at: MonoNs) {
        let due = {
            let mut state = lock(&self.state);
            assert!(
                at >= state.mono,
                "ManualClock::set_monotonic: {at:?} is before the current {:?}",
                state.mono
            );
            move_to(&mut state, at)
        };
        wake_all(due);
    }

    /// Sleeps that have been polled, are not yet due, and are still alive.
    #[must_use]
    pub fn pending_timers(&self) -> usize {
        lock(&self.state).timers.len()
    }
}

const NS_PER_MS: u64 = 1_000_000;

/// 2026-10-17: Sets monotonic time and removes the timers it makes due, in key order.
fn move_to(state: &mut State, to: MonoNs) -> Vec<Waker> {
    state.mono = to;
    let pending = match to.0.checked_add(1) {
        Some(next) => state.timers.split_off(&(MonoNs(next), 0)),
        None => BTreeMap::new(),
    };
    std::mem::replace(&mut state.timers, pending)
        .into_values()
        .collect()
}

/// 2026-10-17: Wakes outside the lock, because a waker may poll the sleep inline.
fn wake_all(due: Vec<Waker>) {
    for waker in due {
        waker.wake();
    }
}

impl Clock for ManualClock {
    fn now_unix_ms(&self) -> UnixMs {
        lock(&self.state).wall
    }

    fn monotonic_ns(&self) -> MonoNs {
        lock(&self.state).mono
    }
}

impl Timer for ManualTimer {
    async fn sleep_until(&self, at: MonoNs) {
        Sleep {
            state: &self.state,
            at,
            id: None,
        }
        .await;
    }
}

struct Sleep<'a> {
    state: &'a Shared,
    at: MonoNs,
    id: Option<u64>,
}

impl Future for Sleep<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        let mut state = lock(this.state);
        if state.mono >= this.at {
            if let Some(id) = this.id.take() {
                state.timers.remove(&(this.at, id));
            }
            return Poll::Ready(());
        }
        match this.id {
            Some(id) => {
                let waker = state
                    .timers
                    .get_mut(&(this.at, id))
                    .expect("ManualClock: a pending sleep lost its timer");
                if !waker.will_wake(cx.waker()) {
                    waker.clone_from(cx.waker());
                }
            }
            None => {
                let id = state.next_id;
                state.next_id = id.checked_add(1).expect("ManualClock: timer ids exhausted");
                state.timers.insert((this.at, id), cx.waker().clone());
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
        // panic would abort and hide the first; the timer map dies with the test.
        if std::thread::panicking() {
            return;
        }
        lock(self.state).timers.remove(&(self.at, id));
    }
}

#[cfg(test)]
#[path = "clock_tests.rs"]
mod tests;
