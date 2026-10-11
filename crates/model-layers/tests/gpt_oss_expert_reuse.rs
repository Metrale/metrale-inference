// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Reuse-plan pointer admission and exact launch geometry.
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};
use metrale_model_layers::layers::ops::{GptOssTokenExperts, gpt_oss_mxfp4_reuse_experts};
#[test]
fn bounded_plan_grid_and_slot_stride_are_explicit() {
    let gpu = MockGpuBackend::new();
    gpt_oss_mxfp4_reuse_experts(
        &gpu,
        KernelHandle(7),
        DevicePtr(0x1000),
        DevicePtr(0x20000),
        DevicePtr(0x30000),
        DevicePtr(0x40000),
        DevicePtr(0x50000),
        DevicePtr(0x60000),
        &GptOssTokenExperts {
            tokens: 16,
            rows: 35,
            cols: 96,
            per_slot_input: true,
        },
        19,
    )
    .unwrap();
    let calls = gpu.launches_snapshot();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].grid, [9, 32, 4]);
    assert_eq!(calls[0].block, [128, 1, 1]);
    assert_eq!(calls[0].stream, 19);
    assert_eq!(calls[0].args[4], MockArg::Buffer(DevicePtr(0x50000)));
    assert_eq!(
        calls[0].args[9],
        MockArg::Bytes(1536u32.to_le_bytes().to_vec())
    );
}
#[test]
fn invalid_plan_pointer_or_geometry_never_launches() {
    let gpu = MockGpuBackend::new();
    for (kernel, plan, tokens) in [
        (0, 0x50000, 16),
        (7, 0, 16),
        (7, 0x50001, 16),
        (7, u64::MAX - 3, 16),
        (7, 0x60000, 16),
        (7, 0x50000, 0),
        (7, 0x50000, 17),
    ] {
        assert!(
            gpt_oss_mxfp4_reuse_experts(
                &gpu,
                KernelHandle(kernel),
                DevicePtr(0x1000),
                DevicePtr(0x20000),
                DevicePtr(0x30000),
                DevicePtr(0x40000),
                DevicePtr(plan),
                DevicePtr(0x60000),
                &GptOssTokenExperts {
                    tokens,
                    rows: 35,
                    cols: 96,
                    per_slot_input: true
                },
                19,
            )
            .is_err()
        );
    }
    assert!(gpu.launches_snapshot().is_empty());
}
