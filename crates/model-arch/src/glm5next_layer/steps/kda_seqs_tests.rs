// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Tests of the multi-sequence KDA verify plan (`kda_seqs.rs`): which snapshot
//! slots each run keeps, which groups walk and why, and which plans are refused.
//! 2026-10-16: Moved out of `kda_seqs.rs` with the multi-snapshot plan.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_layer::group_spans;

/// 2026-10-14: `kda_seqs_runs` over `keep` as `forward_spans` passes it.
fn plan(
    spans: &[GroupSpan],
    ks: &[usize],
    keep: Option<&[KdaRowKeep]>,
    kda: &[KdaSeqState],
) -> Result<SeqsPlan> {
    kda_seqs_runs(spans, ks, &snapshot_lists(keep), kda, &[])
}

/// 2026-10-16: The runs of a plan that must take the seqs kernels.
fn runs(p: Result<SeqsPlan>) -> Vec<KdaSeqRun> {
    match p.expect("plan refused") {
        SeqsPlan::Runs(r) => r,
        SeqsPlan::Walk(w) => panic!("walked: {w:?}"),
    }
}

fn walk(p: Result<SeqsPlan>) -> SeqsWalk {
    match p.expect("plan refused") {
        SeqsPlan::Walk(w) => w,
        SeqsPlan::Runs(r) => panic!("took {} runs", r.len()),
    }
}

fn st(i: u64) -> KdaSeqState {
    KdaSeqState {
        conv: DevicePtr(0x1000 + i),
        recurrent: DevicePtr(0x2000 + i),
    }
}

fn snap(i: u64) -> (DevicePtr, DevicePtr) {
    (DevicePtr(0x3000 + i), DevicePtr(0x4000 + i))
}

/// 2026-10-16: Sequence `s`'s `k - 1` snapshot slots, `snap(16 * s + t)` for row `t`.
fn snaps_of(s: u64, k: usize) -> KdaRowKeep {
    KdaRowKeep::Snapshots((0..k as u64 - 1).map(|t| snap(16 * s + t)).collect())
}

fn span(seq: usize, row0: usize, t0: usize, rows: usize) -> GroupSpan {
    GroupSpan {
        seq,
        row0,
        t0,
        rows,
    }
}

/// 2026-10-16: Each run's recurrent state, rows and snapshot recurrent slots.
fn shape(runs: &[KdaSeqRun]) -> Vec<(u64, usize, Vec<u64>)> {
    runs.iter()
        .map(|r| {
            let s = r.snapshots.as_slice().iter().map(|p| p.0.0).collect();
            (r.state.recurrent.0, r.rows, s)
        })
        .collect()
}

/// 2026-10-14: k = 2 verifies under snapshot rollback: each run keeps its one intermediate (the
/// state after row 0), both halves, on its own state, in span order.
#[test]
fn k2_snapshot_verify_keeps_each_runs_first_row() {
    let kda: Vec<KdaSeqState> = (0..3).map(st).collect();
    let keep: Vec<KdaRowKeep> = (0..3).map(|i| snaps_of(i, 2)).collect();
    let spans = [span(0, 0, 0, 2), span(1, 2, 0, 2), span(2, 4, 0, 2)];
    let r = runs(plan(&spans, &[2, 2, 2], Some(&keep), &kda));
    assert_eq!(
        shape(&r),
        vec![
            (0x2000, 2, vec![0x3000]),
            (0x2001, 2, vec![0x3010]),
            (0x2002, 2, vec![0x3020])
        ]
    );
    assert_eq!(r[1].snapshots.as_slice(), &[snap(0x10)]);
}

/// 2026-10-16: Spans of 3, 4 and 8 rows (`--num-drafts` 2, 3, 7) keep all `k - 1` slots in row
/// order, both halves; 8 rows is the `KDA_SEQS_SNAP_MAX` edge.
#[test]
fn spans_of_three_four_and_eight_rows_keep_every_snapshot_in_row_order() {
    for k in [3usize, 4, 8] {
        let kda = [st(0), st(1)];
        let keep = [snaps_of(0, k), snaps_of(1, k)];
        let spans = group_spans(&[k, k], 0, 2 * k);
        let r = runs(plan(&spans, &[k, k], Some(&keep), &kda));
        let want = |s: u64| {
            (0..k as u64 - 1)
                .map(|t| snap(16 * s + t))
                .collect::<Vec<_>>()
        };
        assert_eq!(r.len(), 2, "k = {k}");
        for (s, run) in r.iter().enumerate() {
            assert_eq!(run.rows, k, "k = {k}");
            assert_eq!(
                run.snapshots.as_slice(),
                want(s as u64).as_slice(),
                "k = {k}"
            );
        }
    }
}

/// 2026-10-16: A mixed batch of one-row (`KdaRowKeep::Nothing`, as `verify_multi` keeps for
/// k = 1) and three-row sequences: the one-row runs keep nothing, the three-row runs both slots.
#[test]
fn a_mixed_batch_of_one_and_three_row_sequences() {
    let ks = [1usize, 3, 1, 3];
    let kda: Vec<KdaSeqState> = (0..4).map(st).collect();
    let keep = [
        KdaRowKeep::Nothing,
        snaps_of(1, 3),
        KdaRowKeep::Nothing,
        snaps_of(3, 3),
    ];
    let r = runs(plan(&group_spans(&ks, 0, 8), &ks, Some(&keep), &kda));
    assert_eq!(
        shape(&r),
        vec![
            (0x2000, 1, vec![]),
            (0x2001, 3, vec![0x3010, 0x3011]),
            (0x2002, 1, vec![]),
            (0x2003, 3, vec![0x3030, 0x3031])
        ]
    );
}

/// 2026-10-16: A sequence across two groups keeps each slot exactly once, in the group holding
/// its row: `ks = [3, 8]` in groups of 6 and 5 rows puts sequence 1's rows 0..3 in the first
/// (slots 0..3) and rows 3..8 in the second (slots 3..7; row 7, the last, keeps none).
#[test]
fn a_straddling_span_keeps_each_slot_once_where_its_row_lands() {
    let ks = [3usize, 8];
    let kda = [st(0), st(1)];
    let keep = [snaps_of(0, 3), snaps_of(1, 8)];
    let first = runs(plan(&group_spans(&ks, 0, 6), &ks, Some(&keep), &kda));
    assert_eq!(
        shape(&first),
        vec![
            (0x2000, 3, vec![0x3000, 0x3001]),
            (0x2001, 3, vec![0x3010, 0x3011, 0x3012])
        ]
    );
    let second = runs(plan(&group_spans(&ks, 6, 5), &ks, Some(&keep), &kda));
    assert_eq!(
        shape(&second),
        vec![(0x2001, 5, vec![0x3013, 0x3014, 0x3015, 0x3016])]
    );
    let k2 = [snaps_of(0, 2), snaps_of(1, 2)];
    let tail = [span(1, 0, 1, 1), span(0, 1, 0, 2)];
    let r = runs(plan(&tail, &[2, 2], Some(&k2), &kda));
    assert_eq!(
        shape(&r),
        vec![(0x2001, 1, vec![]), (0x2000, 2, vec![0x3000])]
    );
}

/// 2026-10-16: A snapshot list that is not `k - 1` long is refused, never walked: one short,
/// one long, and on a later span of a group that would otherwise walk for a shared state. The
/// control: the same group with `k - 1` slots takes the seqs kernels.
#[test]
fn a_snapshot_count_mismatch_is_refused_not_walked() {
    let ks = [3usize, 3];
    let kda = [st(0), st(1)];
    let spans = group_spans(&ks, 0, 6);
    let bad = |n: u64| {
        [
            KdaRowKeep::Snapshots((0..n).map(snap).collect()),
            snaps_of(1, 3),
        ]
    };
    for n in [1u64, 3] {
        let e = plan(&spans, &ks, Some(&bad(n)), &kda).expect_err("mismatch accepted");
        assert!(format!("{e}").contains("refused"), "{e}");
    }
    let shared = [st(0), st(0)];
    let keep = [snaps_of(0, 3), KdaRowKeep::Snapshots(vec![snap(1)])];
    assert!(plan(&spans, &ks, Some(&keep), &shared).is_err());
    let good = [snaps_of(0, 3), snaps_of(1, 3)];
    assert_eq!(runs(plan(&spans, &ks, Some(&good), &kda)).len(), 2);
}

/// 2026-10-16: A plan that does not add up is refused: a sequence without a state or row count,
/// a span past its sequence's rows, an empty span.
#[test]
fn spans_outside_their_sequence_are_refused() {
    let kda = [st(0), st(1)];
    assert!(plan(&[span(5, 0, 0, 2)], &[2, 2, 2, 2, 2, 2], None, &kda).is_err());
    assert!(plan(&[span(1, 0, 0, 2)], &[2], None, &kda).is_err());
    assert!(plan(&[span(0, 0, 1, 2)], &[2, 2], None, &kda).is_err());
    assert!(plan(&[span(0, 0, 0, 0), span(1, 0, 0, 2)], &[2, 2], None, &kda).is_err());
    assert_eq!(
        runs(plan(&[span(0, 0, 0, 2)], &[2, 2], None, &kda)).len(),
        1
    );
}

/// 2026-10-16: The walks and their reasons: a decode (no span of two rows), one recurrent or one
/// conv state under two spans (padding), and a span owing more than `KDA_SEQS_SNAP_MAX` slots.
/// The control for the last: the same nine-row sequence split 7 + 2 takes the seqs kernels in
/// both groups.
#[test]
fn walks_name_their_reason() {
    let kda: Vec<KdaSeqState> = (0..3).map(st).collect();
    let ones = [span(0, 0, 0, 1), span(1, 1, 0, 1)];
    assert_eq!(walk(plan(&ones, &[1, 1], None, &kda)), SeqsWalk::Decode);
    let k2 = [span(0, 0, 0, 2), span(1, 2, 0, 2)];
    let shared = [st(0), st(0)];
    assert_eq!(
        walk(plan(&k2, &[2, 2], None, &shared)),
        SeqsWalk::SharedState
    );
    let conv_shared = [
        st(0),
        KdaSeqState {
            conv: st(0).conv,
            recurrent: st(1).recurrent,
        },
    ];
    assert_eq!(
        walk(plan(&k2, &[2, 2], None, &conv_shared)),
        SeqsWalk::SharedState
    );
    let keep = [snaps_of(0, 9)];
    let whole = group_spans(&[9], 0, 9);
    assert_eq!(
        walk(plan(&whole, &[9], Some(&keep), &kda)),
        SeqsWalk::SnapshotsOverCap
    );
    let a = runs(plan(&group_spans(&[9], 0, 7), &[9], Some(&keep), &kda));
    let b = runs(plan(&group_spans(&[9], 7, 2), &[9], Some(&keep), &kda));
    assert_eq!(
        (a[0].snapshots.as_slice().len(), b[0].snapshots.as_slice()),
        (KDA_SEQS_SNAP_MAX, &[snap(7)][..])
    );
}

/// 2026-10-14: The last row of a k-row span is never snapshotted (k - 1 snapshots), so a span
/// that starts on it takes none, and a plan without a keep takes none at all.
#[test]
fn rows_past_the_snapshot_count_and_no_keep_take_no_snapshot() {
    let kda: Vec<KdaSeqState> = (0..2).map(st).collect();
    let keep = [snaps_of(0, 2), KdaRowKeep::Nothing];
    let spans = [span(0, 0, 1, 1), span(1, 1, 0, 2)];
    let r = runs(plan(&spans, &[2, 2], Some(&keep), &kda));
    assert_eq!(shape(&r), vec![(0x2000, 1, vec![]), (0x2001, 2, vec![])]);
    let r = runs(plan(&[span(0, 0, 0, 2)], &[2, 2], None, &kda));
    assert_eq!(shape(&r), vec![(0x2000, 2, vec![])]);
}

/// 2026-10-16: SNAPZ: a span starting at its sequence's row 0 takes that sequence's defer
/// (indexed by `span.seq`, not by span order); the second group of a straddling span
/// (`t0 > 0`) and a sequence past the defer list take none. The control: without a defer list
/// no run defers.
#[test]
fn only_a_spans_row_zero_takes_its_sequences_defer() {
    let kda: Vec<KdaSeqState> = (0..3).map(st).collect();
    let d = |i: u64| KdaDefer {
        flag: DevicePtr(0x9000 + 4 * i),
        h: DevicePtr(0x3000 + i),
        conv: DevicePtr(0x4000 + i),
    };
    let defer = [Some(d(0)), Some(d(1))];
    let ks = [2usize, 2, 2];
    let first = [span(1, 0, 0, 2), span(0, 2, 0, 1), span(2, 3, 0, 1)];
    let r = runs(kda_seqs_runs(&first, &ks, &[], &kda, &defer));
    let got: Vec<_> = r.iter().map(|r| r.defer).collect();
    assert_eq!(got, vec![Some(d(1)), Some(d(0)), None]);
    let second = [span(0, 0, 1, 1), span(2, 1, 0, 2)];
    let r = runs(kda_seqs_runs(&second, &ks, &[], &kda, &defer));
    assert!(r.iter().all(|r| r.defer.is_none()));
    let r = runs(kda_seqs_runs(&first, &ks, &[], &kda, &[]));
    assert!(r.iter().all(|r| r.defer.is_none()));
}

/// 2026-10-16: The walk log fires once per reason: the first walk of each non-decode reason is
/// new, a repeat is not, a decode never is.
#[test]
fn each_walk_reason_is_logged_once_and_a_decode_never() {
    let seen = AtomicU8::new(0);
    assert!(!first_walk(&seen, SeqsWalk::Decode));
    assert!(first_walk(&seen, SeqsWalk::SharedState));
    assert!(!first_walk(&seen, SeqsWalk::SharedState));
    assert!(first_walk(&seen, SeqsWalk::NotReady));
    assert!(first_walk(&seen, SeqsWalk::SnapshotsOverCap));
    assert!(!first_walk(&seen, SeqsWalk::NotReady));
    assert!(!first_walk(&seen, SeqsWalk::Decode));
}
