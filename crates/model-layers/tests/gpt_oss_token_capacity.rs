// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Explicit larger token-grid launch bounds; expert reuse remains at 16.
use metrale_gpu_runtime::gpu::{
    DevicePtr, KernelHandle,
    mock::{MockArg, MockGpuBackend},
};
use metrale_model_layers::layers::ops::{
    GptOssTokenExperts, gpt_oss_mxfp4_reuse_experts, gpt_oss_mxfp4_token_experts,
    gpt_oss_token_expert_bias,
};

#[test]
fn larger_token_grid_preserves_slot_major_stride_and_bias_extent() {
    for tokens in [17, 31, 32, 63, 64, 65, 127, 128] {
        let gpu = MockGpuBackend::new();
        let g = GptOssTokenExperts {
            tokens,
            rows: 35,
            cols: 96,
            per_slot_input: true,
        };
        gpt_oss_mxfp4_token_experts(
            &gpu,
            KernelHandle(1),
            DevicePtr(0x1000000),
            DevicePtr(0x2000000),
            DevicePtr(0x3000000),
            DevicePtr(0x4000000),
            DevicePtr(0x5000000),
            &g,
            7,
        )
        .unwrap();
        gpt_oss_token_expert_bias(
            &gpu,
            KernelHandle(2),
            DevicePtr(0x5000000),
            DevicePtr(0x6000000),
            DevicePtr(0x4000000),
            tokens,
            35,
            7,
        )
        .unwrap();
        let calls = gpu.launches_snapshot();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].grid, [9, 4, tokens]);
        assert_eq!(
            calls[0].args.last().unwrap(),
            &MockArg::Bytes((tokens * 96).to_le_bytes().to_vec())
        );
        assert_eq!(calls[1].grid, [(4 * tokens * 35).div_ceil(256), 1, 1]);
    }
}

#[test]
fn reuse_does_not_inherit_larger_token_grid_admission() {
    let gpu = MockGpuBackend::new();
    for tokens in [17, 64, 128, 129] {
        let g = GptOssTokenExperts {
            tokens,
            rows: 35,
            cols: 96,
            per_slot_input: true,
        };
        assert!(
            gpt_oss_mxfp4_reuse_experts(
                &gpu,
                KernelHandle(1),
                DevicePtr(0x1000000),
                DevicePtr(0x2000000),
                DevicePtr(0x3000000),
                DevicePtr(0x4000000),
                DevicePtr(0x6000000),
                DevicePtr(0x5000000),
                &g,
                7
            )
            .is_err()
        );
    }
    assert!(gpu.launches_snapshot().is_empty());
}

#[test]
fn explicit_wide_reuse_preserves_group_stride_and_refuses_overflow() {
    use metrale_model_layers::layers::ops::gpt_oss_mxfp4_reuse_wide_experts;
    for tokens in [17, 31, 63, 64, 65, 127, 128, 129] {
        let gpu = MockGpuBackend::new();
        let g = GptOssTokenExperts {
            tokens,
            rows: 35,
            cols: 96,
            per_slot_input: true,
        };
        let result = gpt_oss_mxfp4_reuse_wide_experts(
            &gpu,
            KernelHandle(1),
            DevicePtr(0x1000000),
            DevicePtr(0x2000000),
            DevicePtr(0x3000000),
            DevicePtr(0x4000000),
            DevicePtr(0x6000000),
            DevicePtr(0x5000000),
            &g,
            7,
        );
        if tokens > 128 {
            assert!(result.is_err());
            assert!(gpu.launches_snapshot().is_empty());
        } else {
            result.unwrap();
            let calls = gpu.launches_snapshot();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].grid, [9, 32, tokens.div_ceil(4)]);
            assert_eq!(
                calls[0].args.last().unwrap(),
                &MockArg::Bytes((tokens * 96).to_le_bytes().to_vec())
            );
        }
    }
}
