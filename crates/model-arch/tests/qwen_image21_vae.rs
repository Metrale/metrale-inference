// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Original-FP32 single-frame decoder boundary and shape refusals.
use metrale_gpu_runtime::gpu::{DevicePtr, mock::MockGpuBackend};
use metrale_model_arch::qwen_image21::vae::{
    DiagnosticVaeConv, ImageShape, channel_norm_f32, silu_f32,
};
use metrale_model_weights::weights::{WeightDtype, WeightTensor};
#[test]
fn vae_retains_original_fp32_and_spatial_shape() {
    let gpu = MockGpuBackend::new();
    let shape = ImageShape::new(64, 5, 7).unwrap();
    let weight = WeightTensor {
        ptr: DevicePtr(1 << 30),
        shape: vec![1152, 64, 3, 3],
        dtype: WeightDtype::FP32,
    };
    let bias = WeightTensor {
        ptr: DevicePtr(2 << 30),
        shape: vec![1152],
        dtype: WeightDtype::FP32,
    };
    {
        let mut conv = DiagnosticVaeConv::new(&gpu, shape, &weight, &bias).unwrap();
        assert_eq!(conv.output_shape().dimensions(), [1152, 5, 7]);
        assert!(conv.forward(DevicePtr(0), 9).is_err());
        assert!(gpu.launches_snapshot().is_empty());
        let out = conv.forward(DevicePtr(3 << 30), 9).unwrap();
        assert!(conv.forward(out, 9).is_err());
        assert!(conv.forward(DevicePtr(out.0 + 4), 9).is_err());
        assert!(conv.forward(DevicePtr(u64::MAX - 3), 9).is_err());
        assert_eq!(gpu.launches_snapshot().len(), 1);
        let gamma = WeightTensor {
            ptr: DevicePtr(4 << 30),
            shape: vec![1152, 1, 1, 1],
            dtype: WeightDtype::FP32,
        };
        channel_norm_f32(
            &gpu,
            conv.output_shape(),
            out,
            &gamma,
            DevicePtr(5 << 30),
            9,
        )
        .unwrap();
        silu_f32(
            &gpu,
            conv.output_shape(),
            DevicePtr(5 << 30),
            DevicePtr(6 << 30),
            9,
        )
        .unwrap();
        assert_eq!(gpu.launches_snapshot().len(), 3);
        assert!(gpu.launches_snapshot().iter().all(|l| l.stream == 9));
    }
    assert_eq!(gpu.alloc_count(), 0);
}
#[test]
fn vae_refuses_precision_substitution_and_invalid_geometry() {
    let gpu = MockGpuBackend::new();
    assert!(ImageShape::new(0, 1, 1).is_err());
    assert!(ImageShape::new(64, u32::MAX, 2).is_err());
    let shape = ImageShape::new(64, 5, 7).unwrap();
    let bias = WeightTensor {
        ptr: DevicePtr(8192),
        shape: vec![64],
        dtype: WeightDtype::FP32,
    };
    for (s, dtype) in [
        (vec![64, 64, 1, 1], WeightDtype::BF16),
        (vec![64, 64, 2, 2], WeightDtype::FP32),
        (vec![64, 32, 3, 3], WeightDtype::FP32),
    ] {
        let w = WeightTensor {
            ptr: DevicePtr(4096),
            shape: s,
            dtype,
        };
        assert!(DiagnosticVaeConv::new(&gpu, shape, &w, &bias).is_err());
    }
    assert_eq!(gpu.alloc_count(), 0);
}
