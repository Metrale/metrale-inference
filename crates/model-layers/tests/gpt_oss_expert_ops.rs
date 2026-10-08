// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Production launcher ABI/refusal tests. CUDA numeric parity is separate.
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};
#[path = "../src/layers/ops/gpt_oss_expert_ops.rs"]
mod ops;

#[test]
fn launches_keep_bf16_boundaries_and_explicit_shapes() {
    let gpu = MockGpuBackend::new();
    let k = KernelHandle(7);
    ops::gpt_oss_expert_bias_bf16(&gpu, k, DevicePtr(0x1000), DevicePtr(0x8000), 3, 259, 19)
        .unwrap();
    ops::gpt_oss_swiglu_bf16(&gpu, k, DevicePtr(0x10000), DevicePtr(0x20000), 259, 19).unwrap();
    ops::gpt_oss_expert_reduce_bf16(
        &gpu,
        k,
        DevicePtr(0x100000),
        DevicePtr(0x200000),
        DevicePtr(0x300000),
        DevicePtr(0x400000),
        3,
        259,
        19,
    )
    .unwrap();
    let calls = gpu.launches_snapshot();
    assert_eq!(calls.len(), 3);
    for (call, grid) in calls.iter().zip([4, 2, 4]) {
        assert_eq!(call.grid, [grid, 1, 1]);
        assert_eq!(call.block, [256, 1, 1]);
        assert_eq!(call.stream, 19);
    }
    assert_eq!(
        calls[0].args,
        vec![
            MockArg::Buffer(DevicePtr(0x1000)),
            MockArg::Buffer(DevicePtr(0x8000)),
            MockArg::Bytes(3u32.to_le_bytes().to_vec()),
            MockArg::Bytes(259u32.to_le_bytes().to_vec())
        ]
    );
    assert_eq!(calls[1].args.len(), 3);
    assert_eq!(calls[2].args.len(), 6);
    assert_eq!(
        calls[2].args[4],
        MockArg::Bytes(3u32.to_le_bytes().to_vec())
    );
    assert_eq!(
        calls[2].args[5],
        MockArg::Bytes(259u32.to_le_bytes().to_vec())
    );
}

#[test]
fn rejects_aliases_missing_kernel_overflow_and_invalid_alignment() {
    let gpu = MockGpuBackend::new();
    let p = DevicePtr;
    let k = KernelHandle(7);
    for bad in [0, 1, u64::MAX - 1] {
        assert!(ops::gpt_oss_expert_bias_bf16(&gpu, k, p(bad), p(0x8000), 3, 259, 0).is_err());
        assert!(ops::gpt_oss_swiglu_bf16(&gpu, k, p(0x1000), p(bad), 259, 0).is_err());
        assert!(
            ops::gpt_oss_expert_reduce_bf16(
                &gpu,
                k,
                p(0x100000),
                p(0x200000),
                p(bad),
                p(0x400000),
                3,
                259,
                0
            )
            .is_err()
        );
    }
    assert!(ops::gpt_oss_expert_bias_bf16(&gpu, k, p(0x1000), p(0x1002), 3, 259, 0).is_err());
    assert!(ops::gpt_oss_swiglu_bf16(&gpu, k, p(0x1000), p(0x1002), 259, 0).is_err());
    assert!(
        ops::gpt_oss_expert_reduce_bf16(
            &gpu,
            k,
            p(0x100000),
            p(0x200000),
            p(0x300000),
            p(0x100002),
            3,
            259,
            0
        )
        .is_err()
    );
    for (rows, cols) in [(0, 259), (3, 0), (u32::MAX, 2)] {
        assert!(
            ops::gpt_oss_expert_bias_bf16(&gpu, k, p(0x1000), p(0x8000), rows, cols, 0).is_err()
        );
        assert!(
            ops::gpt_oss_expert_reduce_bf16(
                &gpu,
                k,
                p(0x100000),
                p(0x200000),
                p(0x300000),
                p(0x400000),
                rows,
                cols,
                0
            )
            .is_err()
        );
    }
    assert!(ops::gpt_oss_swiglu_bf16(&gpu, KernelHandle(0), p(0x1000), p(0x8000), 259, 0).is_err());
    assert!(gpu.launches_snapshot().is_empty());
}
