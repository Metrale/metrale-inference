// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: A program with a side-stream group, on the mock backend: each launch is issued on
//! its group's stream, and the fork and join events record and wait in plan order; a segment run
//! still ends joined; a side launch without a side stream is refused.
//!
//! Owner: model-layers circuit executor tests.
//! Invariants: none beyond the types.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use metrale_circuit::{EventKind, Stream, StreamEvent};
use metrale_gpu_runtime::gpu::mock::{MockGpuBackend, StreamOp};

use super::{Launch, LaunchKind, Program, Segment, SegmentOf, SideLane, StepEnv};

type Log = Arc<Mutex<Vec<(&'static str, u64)>>>;

fn kernel(log: &Log, name: &'static str, group: usize, lane: Stream) -> Launch {
    let log = log.clone();
    Launch {
        group,
        kernel: name.to_string(),
        kind: LaunchKind::Kernel,
        covers: 1,
        lane,
        run: Box::new(move |e| -> Result<()> {
            log.lock().unwrap().push((name, e.stream));
            Ok(())
        }),
    }
}

fn event(kind: EventKind, after: Option<usize>, before: Option<usize>) -> StreamEvent {
    StreamEvent {
        kind,
        after,
        before,
    }
}

/// 2026-10-04: quant (main), shared expert (side), router (main), blend (main): fork before the
/// shared expert, join before the blend, as `streams::derive_events` derives for that plan.
fn program(log: &Log, lane: Option<SideLane>) -> Program {
    let fork = event(EventKind::Fork, Some(0), Some(1));
    let join = event(EventKind::Join, Some(1), Some(3));
    let launches = vec![
        kernel(log, "quant", 0, Stream::Main),
        SideLane::launch(lane, &fork, 1).unwrap(),
        kernel(log, "shared", 1, Stream::Side),
        kernel(log, "router", 2, Stream::Main),
        SideLane::launch(lane, &join, 3).unwrap(),
        kernel(log, "blend", 3, Stream::Main),
    ];
    Program {
        mode: metrale_circuit::Mode::Prefill,
        rows: 4,
        plan_digest: String::new(),
        segments: vec![
            Segment {
                of: SegmentOf::Layer(0),
                launches: 0..3,
            },
            Segment {
                of: SegmentOf::Layer(1),
                launches: 3..6,
            },
        ],
        launches,
        lane,
    }
}

fn env(gpu: &MockGpuBackend, stream: u64) -> StepEnv<'_> {
    StepEnv {
        gpu,
        stream,
        gdn: &[],
        max_blocks_per_seq: 0,
        prefill: None,
    }
}

// 2026-10-04: Mutation: issuing a side launch on the step's stream, or a record/wait on the
// wrong stream or in the other order, changes the logs.
#[test]
fn side_launches_run_on_the_side_stream_between_their_fork_and_join() {
    let gpu = MockGpuBackend::new();
    gpu.set_distinct_streams();
    let lane = SideLane::create(&gpu).unwrap();
    assert!(lane.stream != 0 && lane.fork != lane.join);
    let log: Log = Arc::default();
    let p = program(&log, Some(lane));
    let main = 7;
    p.run(&env(&gpu, main)).unwrap();
    assert_eq!(
        *log.lock().unwrap(),
        vec![
            ("quant", main),
            ("shared", lane.stream),
            ("router", main),
            ("blend", main)
        ]
    );
    assert_eq!(
        gpu.stream_ops(),
        vec![
            StreamOp::Record {
                event: lane.fork,
                stream: main
            },
            StreamOp::Wait {
                stream: lane.stream,
                event: lane.fork
            },
            StreamOp::Record {
                event: lane.join,
                stream: lane.stream
            },
            StreamOp::Wait {
                stream: main,
                event: lane.join
            },
        ]
    );
    lane.free(&gpu).unwrap();
    assert_eq!(
        gpu.stream_ops()[4..],
        [StreamOp::Destroy(lane.fork), StreamOp::Destroy(lane.join)]
    );
}

// 2026-10-04: The first segment holds the fork and the side launch but not the join. Mutation:
// a segment run that does not end with a join leaves the side stream unjoined.
#[test]
fn a_segment_run_ends_with_the_side_stream_joined() {
    let gpu = MockGpuBackend::new();
    gpu.set_distinct_streams();
    let lane = SideLane::create(&gpu).unwrap();
    let log: Log = Arc::default();
    let p = program(&log, Some(lane));
    p.run_segments(|s| s == SegmentOf::Layer(0), &env(&gpu, 7))
        .unwrap();
    assert_eq!(
        gpu.stream_ops().last(),
        Some(&StreamOp::Wait {
            stream: 7,
            event: lane.join
        })
    );
}

// 2026-10-04: Mutation: running a side launch on the step's stream when the program has no side
// stream would hide a missing lane.
#[test]
fn a_side_launch_without_a_side_stream_is_refused() {
    let gpu = MockGpuBackend::new();
    let log: Log = Arc::default();
    let p = Program {
        lane: None,
        launches: vec![kernel(&log, "shared", 0, Stream::Side)],
        ..program(&log, Some(SideLane::create(&gpu).unwrap()))
    };
    let e = p.run(&env(&gpu, 7)).unwrap_err();
    assert!(format!("{e:#}").contains("side stream"), "{e:#}");
    assert!(log.lock().unwrap().is_empty());
    assert!(SideLane::launch(None, &event(EventKind::Join, Some(0), None), 0).is_err());
}
