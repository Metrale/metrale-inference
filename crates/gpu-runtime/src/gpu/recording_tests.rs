// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `RecordingGpu`: distinct handles per kernel, every stream operation in one log,
//! denied lookups.
//!
//! Owner: gpu-runtime.
//! Invariants: none beyond the types.

use super::*;
use crate::kernel_args::KernelLaunch;

/// 2026-10-23: Two kernels get two handles and the log names each launch by its kernel, with
/// its geometry and arguments; a second lookup of one kernel returns its first handle. Without
/// distinct handles a gate comparing two paths could not see a swapped kernel.
#[test]
fn launches_are_recorded_by_the_kernel_they_name() {
    let gpu = RecordingGpu::new(48);
    let a = gpu.kernel("m", "a").unwrap();
    let b = gpu.kernel("m", "b").unwrap();
    assert_ne!(a.0, b.0);
    assert_eq!(gpu.kernel("m", "a").unwrap().0, a.0);
    KernelLaunch::new(&gpu, b)
        .grid([3, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(DevicePtr(0x10))
        .arg_u32(7)
        .launch(5)
        .unwrap();
    gpu.memset_async(DevicePtr(0x20), 0, 64, 5).unwrap();
    KernelLaunch::new(&gpu, KernelHandle(0x77))
        .launch(5)
        .unwrap();
    let ops = gpu.take();
    assert_eq!(ops.len(), 3);
    assert_eq!(
        ops[0],
        Op::Launch {
            kernel: "m::b".into(),
            grid: [3, 1, 1],
            block: [256, 1, 1],
            shared_mem: 0,
            stream: 5,
            args: vec![
                MockArg::Buffer(DevicePtr(0x10)),
                MockArg::Bytes(7u32.to_le_bytes().to_vec())
            ],
        }
    );
    assert_eq!(
        ops[1],
        Op::Memset {
            ptr: DevicePtr(0x20),
            value: 0,
            bytes: 64,
            stream: Some(5)
        }
    );
    assert_eq!(
        ops[2].kernel(),
        Some("?0x77"),
        "an unknown handle is marked"
    );
    assert!(gpu.take().is_empty(), "take empties the log");
}

/// 2026-10-23: A denied pair fails its lookup and every other pair still resolves.
#[test]
fn a_denied_kernel_fails_its_lookup_only() {
    let gpu = RecordingGpu::new(48);
    gpu.deny("m", "a");
    assert!(gpu.kernel("m", "a").is_err());
    assert!(gpu.kernel("m", "b").is_ok());
    assert_eq!(gpu.sm_count().unwrap(), 48);
}

/// 2026-10-23: Side streams are distinct and none is the default stream, so a fork's launches
/// are told apart from the main stream's in a log.
#[test]
fn side_streams_are_distinct_and_not_the_default() {
    let gpu = RecordingGpu::new(1);
    let (a, b) = (gpu.create_stream().unwrap(), gpu.create_stream().unwrap());
    assert!(a != 0 && b != 0 && a != b, "{a} {b}");
}

/// 2026-10-23: The stream graph keeps streams, events and collectives and drops kernels: two
/// logs that launch different kernels on the same streams have equal graphs, a moved launch or
/// a missing wait does not.
#[test]
fn the_stream_graph_sees_streams_events_and_collectives_only() {
    let gpu = RecordingGpu::new(1);
    let (a, b) = (gpu.kernel("m", "a").unwrap(), gpu.kernel("m", "b").unwrap());
    let run = |k, side: u64, wait: bool| {
        gpu.record_event(9, 7).unwrap();
        if wait {
            gpu.stream_wait_event(side, 9).unwrap();
        }
        KernelLaunch::new(&gpu, k)
            .grid([1, 1, 1])
            .block([1, 1, 1])
            .launch(side)
            .unwrap();
        gpu.record_comm("exchange", Some(7));
        stream_graph(&gpu.take())
    };
    let base = run(a, 5, true);
    assert_eq!(
        base,
        ["record e9@7", "wait 5<-e9", "launch@5", "comm@Some(7)"]
    );
    assert_eq!(run(b, 5, true), base);
    assert_ne!(run(a, 7, true), base);
    assert_ne!(run(a, 5, false), base);
}
