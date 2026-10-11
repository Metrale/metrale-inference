// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Launcher ABI and refusal controls; numeric execution is a CUDA test.
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};
use metrale_model_layers::layers::ops::projection_bias_bf16;

fn run(
    gpu: &MockGpuBackend,
    kernel: u64,
    ptrs: [u64; 3],
    rows: u32,
    cols: u32,
) -> anyhow::Result<()> {
    projection_bias_bf16(
        gpu,
        KernelHandle(kernel),
        DevicePtr(ptrs[0]),
        DevicePtr(ptrs[1]),
        DevicePtr(ptrs[2]),
        rows,
        cols,
        19,
    )
}

#[test]
fn tail_and_row_broadcast_launch_abi_is_explicit() {
    let gpu = MockGpuBackend::new();
    run(&gpu, 7, [0x1000, 0x8000, 0x10000], 2, 259).unwrap();
    let launches = gpu.launches_snapshot();
    assert_eq!(launches.len(), 1);
    let launch = &launches[0];
    assert_eq!(launch.func, 7);
    assert_eq!(launch.grid, [3, 1, 1]);
    assert_eq!(launch.block, [256, 1, 1]);
    assert_eq!(launch.stream, 19);
    assert_eq!(
        launch.args,
        vec![
            MockArg::Buffer(DevicePtr(0x1000)),
            MockArg::Buffer(DevicePtr(0x8000)),
            MockArg::Buffer(DevicePtr(0x10000)),
            MockArg::Bytes(518u32.to_le_bytes().to_vec()),
            MockArg::Bytes(259u32.to_le_bytes().to_vec())
        ]
    );
}

#[test]
fn invalid_geometry_addresses_and_aliases_never_launch() {
    let gpu = MockGpuBackend::new();
    let valid = [0x1000, 0x8000, 0x10000];
    for (rows, cols) in [(0, 1), (1, 0), (u32::MAX, 2)] {
        assert!(run(&gpu, 7, valid, rows, cols).is_err());
    }
    assert!(run(&gpu, 0, valid, 1, 32).is_err());
    for index in 0..3 {
        for invalid in [0, valid[index] + 1, u64::MAX - 3] {
            let mut ptrs = valid;
            ptrs[index] = invalid;
            assert!(run(&gpu, 7, ptrs, 2, 32).is_err());
        }
    }
    for output in [0x1000, 0x1004, 0x8000, 0x8002] {
        assert!(run(&gpu, 7, [valid[0], valid[1], output], 2, 32).is_err());
    }
    assert!(gpu.launches_snapshot().is_empty());
    run(&gpu, 7, valid, 2, 32).unwrap();
    assert_eq!(gpu.launches_snapshot().len(), 1);
}
