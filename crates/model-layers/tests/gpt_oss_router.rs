// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Router launch admission and dense-score ABI controls.
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};
use metrale_model_layers::layers::ops::gpt_oss_router_bf16;

fn run(gpu: &MockGpuBackend, k: u64, ptrs: [u64; 3], rows: u32) -> anyhow::Result<()> {
    gpt_oss_router_bf16(
        gpu,
        KernelHandle(k),
        DevicePtr(ptrs[0]),
        DevicePtr(ptrs[1]),
        DevicePtr(ptrs[2]),
        rows,
        13,
    )
}

#[test]
fn dense_scores_and_four_ids_are_bound_to_rows() {
    let gpu = MockGpuBackend::new();
    run(&gpu, 7, [0x1000, 0x8000, 0x10000], 3).unwrap();
    let logs = gpu.launches_snapshot();
    assert_eq!(logs.len(), 1);
    assert_eq!(logs[0].grid, [3, 1, 1]);
    assert_eq!(logs[0].block, [256, 1, 1]);
    assert_eq!(logs[0].stream, 13);
    assert_eq!(
        logs[0].args,
        vec![
            MockArg::Buffer(DevicePtr(0x1000)),
            MockArg::Buffer(DevicePtr(0x8000)),
            MockArg::Buffer(DevicePtr(0x10000)),
            MockArg::Bytes(32u32.to_le_bytes().to_vec()),
            MockArg::Bytes(4u32.to_le_bytes().to_vec())
        ]
    );
}

#[test]
fn refused_ranges_never_launch_but_valid_control_does() {
    let gpu = MockGpuBackend::new();
    let valid = [0x1000, 0x8000, 0x10000];
    for rows in [0, u32::MAX] {
        assert!(run(&gpu, 7, valid, rows).is_err());
    }
    assert!(run(&gpu, 0, valid, 2).is_err());
    for index in 0..3 {
        for value in [0, valid[index] + 1, u64::MAX - 3] {
            let mut ptrs = valid;
            ptrs[index] = value;
            assert!(run(&gpu, 7, ptrs, 2).is_err());
        }
    }
    for ptrs in [
        [0x1000, 0x1040, 0x10000],
        [0x1000, 0x8000, 0x107e],
        [0x1000, 0x8000, 0x801e],
    ] {
        assert!(run(&gpu, 7, ptrs, 2).is_err());
    }
    assert!(gpu.launches_snapshot().is_empty());
    run(&gpu, 7, valid, 2).unwrap();
    assert_eq!(gpu.launches_snapshot().len(), 1);
}
