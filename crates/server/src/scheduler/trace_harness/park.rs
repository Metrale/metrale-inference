// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Parks the scheduler thread when it reaches a chosen model call (before the call's
//! trace line is recorded), so a test can act at that exact point of the loop (close the inbox,
//! say) and then let it go on.
//!
//! Owner: scheduler.
//! Invariants: an armed park fires once, on the first recorded line that starts with its
//! prefix; an unarmed one never blocks.

use std::sync::{Condvar, Mutex};

#[derive(Default)]
pub(super) struct LinePark {
    state: Mutex<ParkState>,
    cv: Condvar,
}

#[derive(Default)]
struct ParkState {
    prefix: Option<String>,
    parked: bool,
    released: bool,
}

impl LinePark {
    /// 2026-10-03: Park on the first recorded line starting with `prefix`.
    pub fn arm(&self, prefix: &str) {
        self.state.lock().unwrap().prefix = Some(prefix.to_string());
    }

    /// 2026-10-03: Called on the scheduler thread with each line about to be recorded.
    pub fn check(&self, line: &str) {
        let mut g = self.state.lock().unwrap();
        if !g.prefix.as_deref().is_some_and(|p| line.starts_with(p)) {
            return;
        }
        g.prefix = None;
        g.parked = true;
        self.cv.notify_all();
        while !g.released {
            g = self.cv.wait(g).unwrap();
        }
    }

    /// 2026-10-03: Block until the loop is parked.
    pub fn wait_parked(&self) {
        let mut g = self.state.lock().unwrap();
        while !g.parked {
            g = self.cv.wait(g).unwrap();
        }
    }

    pub fn release(&self) {
        self.state.lock().unwrap().released = true;
        self.cv.notify_all();
    }
}
