// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: VAE original-FP32 residual/upsampling composition and refusals.
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, mock::MockGpuBackend};
use metrale_model_arch::qwen_image21::{
    vae::ImageShape,
    vae_layout::{upsample_f32, upsample_shape},
    vae_residual::{DiagnosticVaeResidual, ResidualWeights},
};
use metrale_model_weights::weights::{WeightDtype, WeightTensor};
fn w(i: u64, shape: &[usize]) -> WeightTensor {
    WeightTensor {
        ptr: DevicePtr((i + 1) << 30),
        shape: shape.to_vec(),
        dtype: WeightDtype::FP32,
    }
}
#[test]
fn residual_composes_identity_and_projection_without_weight_conversion() {
    for (ic, oc) in [(1152, 1152), (1152, 576)] {
        let gpu = MockGpuBackend::new();
        let weights = [
            w(0, &[ic, 1, 1, 1]),
            w(1, &[oc, ic, 3, 3]),
            w(2, &[oc]),
            w(3, &[oc, 1, 1, 1]),
            w(4, &[oc, oc, 3, 3]),
            w(5, &[oc]),
            w(6, &[oc, ic, 1, 1]),
            w(7, &[oc]),
        ];
        {
            let mut block = DiagnosticVaeResidual::new(
                &gpu,
                ImageShape::new(ic as u32, 3, 5).unwrap(),
                ResidualWeights {
                    norm1: &weights[0],
                    conv1: &weights[1],
                    bias1: &weights[2],
                    norm2: &weights[3],
                    conv2: &weights[4],
                    bias2: &weights[5],
                    shortcut: if ic == oc {
                        None
                    } else {
                        Some((&weights[6], &weights[7]))
                    },
                },
            )
            .unwrap();
            let result = block.forward(DevicePtr(16 << 30), 17).unwrap();
            let launches = gpu.launches_snapshot();
            assert_eq!(launches.len(), if ic == oc { 7 } else { 8 });
            assert_eq!(
                launches.last().unwrap().func,
                gpu.kernel("nllb_encoder", "nllb_add_inplace").unwrap().0
            );
            assert!(launches.iter().all(|l| l.stream == 17));
            assert!(block.forward(result, 17).is_err());
            assert_eq!(gpu.launches_snapshot().len(), launches.len());
        }
        assert_eq!(gpu.alloc_count(), 0);
    }
}
#[test]
fn upsampling_refuses_fractional_repeat_and_aliases_before_launch() {
    let gpu = MockGpuBackend::new();
    let shape = ImageShape::new(1152, 3, 5).unwrap();
    assert_eq!(
        upsample_shape(shape, 576, 2).unwrap().dimensions(),
        [576, 6, 10]
    );
    assert!(upsample_shape(shape, 100, 2).is_err());
    assert!(upsample_shape(shape, 576, 3).is_err());
    assert!(upsample_f32(&gpu, shape, 576, 2, DevicePtr(4096), DevicePtr(4100), 0).is_err());
    assert!(gpu.launches_snapshot().is_empty());
    upsample_f32(
        &gpu,
        shape,
        576,
        2,
        DevicePtr(1 << 30),
        DevicePtr(2 << 30),
        9,
    )
    .unwrap();
    assert_eq!(
        gpu.launches_snapshot()[0].func,
        gpu.kernel("image_vae", "image_vae_upsample_f32").unwrap().0
    );
}
