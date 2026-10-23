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
