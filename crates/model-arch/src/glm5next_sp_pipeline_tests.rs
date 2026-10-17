// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: SP pipeline: the sub-chunk split, the order, the deferred
//! section form and the workspace routing. The program's happens-before model is
//! `glm5next_sp_pipeline_hb_tests.rs`.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_sp_overlap::{self as ov, Ev, Op};

/// 2026-10-14: The split is `prefill_segments`' (`rows.min(end - t)` until the end), and only
/// a leading run of two or more SP sub-chunks pipelines; a narrow tail stays serial.
#[test]
fn sub_chunks_and_the_pipelined_run() {
    assert_eq!(sub_chunks(32832, 8208), vec![8208; 4]);
    assert_eq!(sub_chunks(20000, 8208), vec![8208, 8208, 3584]);
    assert_eq!(sub_chunks(16500, 8208), vec![8208, 8208, 84]);
    assert_eq!(sub_chunks(100, 8208), vec![100]);
    assert!(sub_chunks(0, 8208).is_empty());
    let takes =
        |len: usize| -> Vec<bool> { sub_chunks(len, 8208).iter().map(|&k| k >= 192).collect() };
    assert_eq!(pipelined(&takes(32832)), 4);
    assert_eq!(pipelined(&takes(16500)), 2);
    assert_eq!(pipelined(&takes(8300)), 0);
    assert_eq!(pipelined(&takes(8208)), 0);
    assert_eq!(pipelined(&[false, true, true]), 0);
    assert_eq!(pipelined(&[true, true, false, true]), 2);
}

/// 2026-10-14: Every stage of every sub-chunk once; within a sub-chunk the serial body's stage
/// order; and the documented interleave at four sub-chunks.
#[test]
fn order_runs_each_stage_once_in_body_order() {
    use Stage::*;
    for n in 1..=6 {
        let o = order(n);
        assert_eq!(o.len(), 5 * n);
        for s in 0..n {
            let at = |st| o.iter().position(|&x| x == (st, s)).unwrap();
            assert!(at(Head) < at(Mixer) && at(Mixer) < at(Mid));
            assert!(at(Mid) < at(Mlp) && at(Mlp) < at(Tail));
        }
        let mixers: Vec<usize> = o.iter().filter(|x| x.0 == Mixer).map(|x| x.1).collect();
        assert_eq!(
            mixers,
            (0..n).collect::<Vec<_>>(),
            "the mixers run in row order"
        );
    }
    assert_eq!(
        order(4),
        vec![
            (Head, 0),
            (Mixer, 0),
            (Head, 1),
            (Mid, 0),
            (Mixer, 1),
            (Head, 2),
            (Mlp, 0),
            (Mid, 1),
            (Mixer, 2),
            (Tail, 0),
            (Head, 3),
            (Mlp, 1),
            (Mid, 2),
            (Mixer, 3),
            (Tail, 1),
            (Mlp, 2),
            (Mid, 3),
            (Tail, 2),
            (Mlp, 3),
            (Tail, 3),
        ]
    );
}

/// 2026-10-14: Event indices are distinct and below `EVENTS`.
#[test]
fn event_indices_are_distinct() {
    use Event::*;
    let mut seen = std::collections::HashSet::new();
    for ev in [Head, MixerRs, Mid, MlpRs, Tail, RsIn, Join] {
        assert!(event_index(ev) < EVENTS && seen.insert(event_index(ev)));
    }
    assert_eq!(seen.len(), EVENTS);
}

/// 2026-10-14: A deferred section: the comm form at one piece too, its end recorded and never
/// waited; no all-gather, the computes alone.
#[test]
fn a_deferred_section_leaves_its_end_open() {
    use ov::On::{Comm, Main};
    assert_eq!(
        ov::schedule_deferred(1, true),
        vec![
            Op::Record(Ev::In, Main),
            Op::Wait(Comm, Ev::In),
            Op::Compute(0),
            Op::Record(Ev::Computed(0), Main),
            Op::Wait(Comm, Ev::Computed(0)),
            Op::AllGather(0, Comm),
            Op::Record(Ev::Out, Comm),
        ]
    );
    assert_eq!(
        ov::schedule_deferred(2, false),
        vec![Op::Compute(0), Op::Compute(1)]
    );
    for m in 2..=ov::MAX_PIECES {
        let mut joined = ov::schedule(m, false, true);
        assert_eq!(joined.pop(), Some(Op::Wait(Main, Ev::Out)));
        assert_eq!(
            ov::schedule_deferred(m, true),
            joined,
            "the deferred form is the joined one minus its wait"
        );
    }
}

/// 2026-10-14: The receive workspace for the MLP's reduce-scatter is the pipeline's, at the
/// SP workspace's slot size; the mixer's is SP-mHC's own; a mismatched world is refused. The
/// normed slots rotate over three and refuse rows past the reserved size.
#[test]
fn workspace_routes_slots() {
    let pw = PipelineWorkspace {
        normed: [DevicePtr(0x1000), DevicePtr(0x2000)],
        normed_bytes: 4096,
        recv: DevicePtr(0x3000),
        slot_bytes: 512,
        world: 3,
        events: [0; EVENTS],
    };
    let own = sp::SpWorkspace {
        recv: DevicePtr(0x9000),
        slot_bytes: 512,
        world: 3,
    };
    let mid = pw.recv_for(&own, RsKind::Mixer).unwrap();
    assert_eq!((mid.recv, mid.slot_bytes), (own.recv, 512));
    let tail = pw.recv_for(&own, RsKind::Mlp).unwrap();
    assert_eq!((tail.recv, tail.slot_bytes, tail.world), (pw.recv, 512, 3));
    let other = sp::SpWorkspace { world: 2, ..own };
    assert!(pw.recv_for(&other, RsKind::Mlp).is_err());
    let base = DevicePtr(0x8000);
    let slots: Vec<DevicePtr> = (0..4)
        .map(|s| pw.normed_at(base, s, 4096).unwrap())
        .collect();
    assert_eq!(slots, vec![base, pw.normed[0], pw.normed[1], base]);
    assert!(pw.normed_at(base, 1, 4097).is_err());
    assert_eq!(
        extra_bytes(8208 * 4096 * 2, 2736 * 4096 * 2, 3),
        179_306_496
    );
}
