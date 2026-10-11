// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The co-dispatch window of `drain_pending_requests`: a lone request waits only the
//! probe for a second one; a second arrival inside the probe opens the settle slices, so a burst
//! is still admitted in one tick. A scripted inbox records each wait the window asks for.
//!
//! Owner: server scheduler.
//! Invariants: none beyond the types.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use metrale_scheduler::WaitPolicy;
use parking_lot::Mutex;

use super::drain_pending_requests;
use crate::api::{InferenceRequest, StreamEvent};
use crate::scheduler::LoraAck;
use crate::scheduler::io::request::LoraRotationAck;
use crate::scheduler::io::{Arrivals, FinishFrame, RequestIo, SchedIo};
use crate::scheduler::levers::SchedLevers;
use crate::scheduler::types::{PendingQueue, ResponseSink};
use crate::scheduling_policy::FifoPolicy;

/// 2026-10-05: An inbox that answers each `recv` with the next scripted batch (then nothing) and
/// records the wait policy of every call.
struct Scripted {
    batches: Mutex<std::collections::VecDeque<Vec<InferenceRequest>>>,
    waits: Mutex<Vec<WaitPolicy>>,
}

impl RequestIo for Scripted {
    fn recv(&self, policy: WaitPolicy) -> Arrivals {
        self.waits.lock().push(policy);
        Arrivals {
            requests: self.batches.lock().pop_front().unwrap_or_default(),
            rotations: Vec::new(),
            closed: false,
        }
    }
    fn lora_ack(&self, _ack: LoraRotationAck, _res: Result<LoraAck, String>) {}
    fn emit(&self, _sink: &ResponseSink, _event: StreamEvent, _what: &str) -> bool {
        true
    }
    fn finish(&self, _sink: &mut ResponseSink, _frame: FinishFrame<'_>) {}
    fn error(&self, _sink: &mut ResponseSink, _msg: &str, _what: &'static str) {}
    fn is_cancelled(&self, _flag: Option<&Arc<AtomicBool>>) -> bool {
        false
    }
}

fn req() -> InferenceRequest {
    crate::scheduler::test_support::blocking_request(None)
}

/// 2026-10-05: Drain with co-dispatch on (window 100 ms, settle 10 ms, probe 1 ms) over an inbox
/// that delivers `batches` in order; returns the admitted count and the waits asked for.
fn drain(batches: Vec<Vec<InferenceRequest>>) -> (usize, Vec<WaitPolicy>) {
    let inbox = Arc::new(Scripted {
        batches: Mutex::new(batches.into()),
        waits: Mutex::new(Vec::new()),
    });
    let mut io = SchedIo::for_test();
    io.req = inbox.clone();
    let mut levers = SchedLevers::default();
    levers.prefill_codispatch = true;
    levers.codispatch_window_ms = 100;
    levers.codispatch_settle_ms = 10;
    levers.codispatch_probe_ms = 1;
    let mut pending = PendingQueue::new();
    let got = drain_pending_requests(
        &io,
        &mut pending,
        &[],
        &[],
        &FifoPolicy,
        &levers,
        128,
        false,
    );
    let waits = inbox.waits.lock().clone();
    (got.len(), waits)
}

#[test]
fn a_lone_request_waits_only_the_probe() {
    let (n, waits) = drain(vec![vec![req()]]);
    assert_eq!(n, 1);
    assert_eq!(
        waits,
        vec![
            WaitPolicy::NoWait,
            WaitPolicy::Bounded(Duration::from_millis(1))
        ],
        "a lone request must not wait a settle slice"
    );
}

#[test]
fn a_second_arrival_in_the_probe_keeps_the_window_open() {
    let (n, waits) = drain(vec![vec![req()], vec![req()], vec![req()]]);
    assert_eq!(n, 3, "the burst is admitted in one tick");
    assert_eq!(
        waits,
        vec![
            WaitPolicy::NoWait,
            WaitPolicy::Bounded(Duration::from_millis(1)),
            WaitPolicy::Bounded(Duration::from_millis(10)),
            WaitPolicy::Bounded(Duration::from_millis(10)),
        ]
    );
}

#[test]
fn a_burst_already_queued_starts_with_settle_slices() {
    let (n, waits) = drain(vec![vec![req(), req()]]);
    assert_eq!(n, 2);
    assert_eq!(
        waits,
        vec![
            WaitPolicy::NoWait,
            WaitPolicy::Bounded(Duration::from_millis(10))
        ]
    );
}
