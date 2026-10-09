// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests of the startup rank-agreement check. 2026-10-10: The comparison on
//! `mismatches` itself (no longer a copy of it), and the whole handshake on two threads over the
//! shared two-rank test communicator: both ranks pass together or fail together, and the first
//! collective is the rendezvous broadcast.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use super::*;
use crate::test_pair_comm::run_pair;

#[test]
fn agreement_is_silent() {
    let items = [("METRALE_GLM_PREFILL_ROWS", 8u64), ("ep_protocol_v2", 0)];
    assert!(mismatches(&items, &[8, 0], 1).is_empty());
}

#[test]
fn a_single_disagreement_names_the_lever_and_both_values() {
    let items = [("METRALE_GLM_PREFILL_ROWS", 4u64), ("ep_protocol_v2", 0)];
    let bad = mismatches(&items, &[8, 0], 1);
    assert_eq!(
        bad,
        vec!["METRALE_GLM_PREFILL_ROWS: rank 1 has 4, rank 0 has 8"]
    );
}

#[test]
fn every_disagreement_is_reported_not_just_the_first() {
    let items = [("METRALE_GLM_PREFILL_ROWS", 4u64), ("ep_protocol_v2", 1)];
    assert_eq!(mismatches(&items, &[8, 0], 1).len(), 2);
}

/// 2026-10-10: Rank `r` runs the check with `vals[r]` as the value of one lever; returns each
/// rank's error text (`None` for `Ok`) and its broadcasts.
fn handshake(vals: [u64; 2]) -> [(Option<String>, Vec<(bool, usize)>); 2] {
    run_pair(move |rank, gpu, comm| {
        assert_ranks_agree(gpu, comm, &[("METRALE_GLM_PREFILL_ROWS", vals[rank])])
            .err()
            .map(|e| e.to_string())
    })
}

/// 2026-10-10: Agreeing ranks both pass, after the rendezvous broadcast of rank 0's values and
/// one verdict broadcast rooted at each rank.
#[test]
fn agreeing_ranks_pass_after_a_rendezvous_and_a_verdict_from_each_rank() {
    for (err, ops) in handshake([512, 512]) {
        assert_eq!(err, None);
        assert_eq!(ops, vec![(true, 0), (false, 0), (false, 1)]);
    }
}

/// 2026-10-10: A worker that disagrees fails its own start naming the lever, AND rank 0's,
/// naming the worker. Negative control: the pre-handshake check (rank 0's broadcast only)
/// returned `Ok` on rank 0 here, which then served against a worker that had stopped.
#[test]
fn a_disagreeing_worker_fails_every_rank() {
    let [(r0, _), (r1, _)] = handshake([512, 16]);
    let r1 = r1.expect("the disagreeing worker fails");
    assert!(
        r1.contains("METRALE_GLM_PREFILL_ROWS: rank 1 has 16, rank 0 has 512"),
        "{r1}"
    );
    let r0 = r0.expect("rank 0 fails too");
    assert!(r0.contains("rank(s) [1] disagree"), "{r0}");
}

#[test]
fn disagreeing_ranks_are_the_zero_verdicts() {
    assert_eq!(disagreeing_ranks(&[1, 1, 1]), Vec::<usize>::new());
    assert_eq!(disagreeing_ranks(&[1, 0, 1, 0]), vec![1, 3]);
}
