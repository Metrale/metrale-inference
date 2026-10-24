// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The harness gate, moved out of `model.rs` unchanged to keep that file under the
//! size cap.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::sync::{Condvar, Mutex};

/// 2026-09-25: A barrier the harness uses to order request arrival against the loop:
/// the scheduler thread parks in a model call until `release`, and the
/// harness can wait for the loop to reach a tick.
#[derive(Default)]
pub(super) struct Gate {
    state: Mutex<GateState>,
    cv: Condvar,
}

#[derive(Default)]
struct GateState {
    started: bool,
    ticks: usize,
    block_at_tick: Option<usize>,
    released: bool,
}

impl Gate {
    /// 2026-09-25: Called by the model on the scheduler thread (from `create_stream`)
    /// before its first tick.
    pub fn wait_start(&self) {
        let mut g = self.state.lock().unwrap();
        while !g.started {
            g = self.cv.wait(g).unwrap();
        }
    }
    pub fn start(&self) {
        self.state.lock().unwrap().started = true;
        self.cv.notify_all();
    }
    /// 2026-09-25: Called by the model once per tick. Parks when the harness asked to
    /// block at this tick, until [`Gate::release`].
    pub fn tick(&self) -> usize {
        let mut g = self.state.lock().unwrap();
        g.ticks += 1;
        let t = g.ticks;
        self.cv.notify_all();
        if g.block_at_tick == Some(t) {
            g.released = false;
            while !g.released {
                g = self.cv.wait(g).unwrap();
            }
        }
        t
    }
    pub fn block_at_tick(&self, t: usize) {
        self.state.lock().unwrap().block_at_tick = Some(t);
    }
    pub fn wait_tick(&self, t: usize) {
        let mut g = self.state.lock().unwrap();
        while g.ticks < t {
            g = self.cv.wait(g).unwrap();
        }
    }
    pub fn release(&self) {
        let mut g = self.state.lock().unwrap();
        g.released = true;
        g.block_at_tick = None;
        self.cv.notify_all();
    }
}
