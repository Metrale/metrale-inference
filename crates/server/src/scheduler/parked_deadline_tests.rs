// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-20: `enforce_parked_deadlines` — a request parked outside `active`
//! (swapped or preempted) with an expired deadline is errored and dropped.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::io::{FileSpill, SchedIo};
use super::lifecycle::swap_out_sequence;
use super::mod_helpers::enforce_parked_deadlines;
use super::test_support::{PreemptStubModel, active_seq};
use super::types::PreemptedSeq;
use metrale_cache::kv_spill::KvSpillManager;
use std::sync::Arc;
use std::time::{Duration, Instant};


fn io() -> SchedIo {
    SchedIo::for_test_with(Arc::new(PreemptStubModel::default()))
}

fn past() -> Option<Instant> {
    Some(Instant::now() - Duration::from_secs(1))
}

/// 2026-10-20: An expired preempted request gets its error frame and is
/// dropped; an unexpired one is kept.
#[test]
fn expired_preempted_request_is_errored_and_dropped() {
    let io = io();
    let (mut a, mut rx) = active_seq(1, 3);
    a.timeout_at = past();
    let mut preempted = vec![PreemptedSeq { a, tokens: vec![] }];
    enforce_parked_deadlines(
        io.dev.model(),
        &io,
        None,
        &mut Vec::new(),
        &mut Vec::new(),
        &mut preempted,
    );
    assert!(preempted.is_empty(), "the expired entry must be dropped");
    assert!(rx.try_recv().expect("error frame").is_err());

    // Negative control: an unexpired request is kept and untouched.
    let (mut a, mut rx) = active_seq(1, 3);
    a.timeout_at = Some(Instant::now() + Duration::from_secs(60));
    let mut preempted = vec![PreemptedSeq { a, tokens: vec![] }];
    enforce_parked_deadlines(
        io.dev.model(),
        &io,
        None,
        &mut Vec::new(),
        &mut Vec::new(),
        &mut preempted,
    );
    assert_eq!(preempted.len(), 1);
    assert!(rx.try_recv().is_err(), "no frame for an unexpired request");
}

/// 2026-10-20: An expired swapped request is errored, dropped, and its spill
/// entry does not block the pool forever. This is the hang in the finding: a
/// swapped request whose blocks never fit retried every tick with nothing
/// checking its deadline.
#[test]
fn expired_swapped_request_is_errored_and_dropped() {
    let dir = std::env::temp_dir().join(format!("parked-deadline-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let spill = FileSpill::new(KvSpillManager::new(dir, 1 << 20).expect("spill dir"));
    let io = io();
    let (mut a, mut rx) = active_seq(1, 3);
    a.timeout_at = Some(Instant::now() - Duration::from_secs(1));
    let mut active = vec![a];
    let swapped = swap_out_sequence(&io, &mut active, 0, &spill).expect("swap-out");
    let mut swapped = vec![swapped];
    enforce_parked_deadlines(
        io.dev.model(),
        &io,
        Some(&spill),
        &mut Vec::new(),
        &mut swapped,
        &mut Vec::new(),
    );
    assert!(swapped.is_empty(), "the expired entry must be dropped");
    assert!(rx.try_recv().expect("error frame").is_err());
}
