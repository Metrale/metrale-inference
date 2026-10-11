// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Dense text-only encoder composition, ownership and refusal controls.
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, mock::MockGpuBackend};
use metrale_model_arch::qwen_image21::encoder::{DiagnosticTextBlock, TextBlockWeights};
use metrale_model_weights::weights::{WeightDtype, WeightTensor};
fn tensors() -> Vec<WeightTensor> {
    [
        vec![4096],
        vec![4096, 4096],
        vec![1024, 4096],
        vec![1024, 4096],
        vec![4096, 4096],
        vec![128],
        vec![128],
        vec![4096],
        vec![12288, 4096],
        vec![12288, 4096],
        vec![4096, 12288],
    ]
    .into_iter()
    .enumerate()
    .map(|(i, shape)| WeightTensor {
        ptr: DevicePtr((i as u64 + 1) << 30),
        shape,
        dtype: WeightDtype::BF16,
    })
    .collect()
}
#[test]
fn dense_encoder_executes_full_block_without_final_norm_or_head() {
    let gpu = MockGpuBackend::new();
    let weights = tensors();
    let borrowed = || TextBlockWeights(std::array::from_fn(|i| &weights[i]));
    assert!(DiagnosticTextBlock::new(&gpu, 0, borrowed()).is_err());
    assert!(DiagnosticTextBlock::new(&gpu, 4097, borrowed()).is_err());
    let input = gpu.alloc(33 * 8192).unwrap();
    let before = gpu.alloc_count();
    {
        let mut block = DiagnosticTextBlock::new(&gpu, 33, borrowed()).unwrap();
        assert_eq!(gpu.alloc_count(), before + 1);
        assert!(
            block
                .forward(DevicePtr(0), DevicePtr(8192), DevicePtr(16384), 7)
                .is_err()
        );
        assert!(gpu.launches_snapshot().is_empty());
        let output = block
            .forward(input, DevicePtr(2 << 40), DevicePtr(3 << 40), 7)
            .unwrap();
        assert!(!output.is_null());
        let launches = gpu.launches_snapshot();
        assert_eq!(launches.len(), 21);
        assert!(launches.iter().all(|l| l.stream == 7));
        assert!(
            launches
                .iter()
                .any(|l| l.grid == [32, 2, 1] && l.block == [128, 1, 1])
        );
    }
    assert_eq!(gpu.alloc_count(), before);
}
#[test]
fn dense_encoder_rejects_moe_and_wrong_kv_geometry() {
    for (index, shape, dtype) in [
        (2, vec![4096, 4096], WeightDtype::BF16),
        (8, vec![32, 12288, 4096], WeightDtype::BF16),
        (0, vec![4096], WeightDtype::FP32),
    ] {
        let gpu = MockGpuBackend::new();
        let mut weights = tensors();
        weights[index].shape = shape;
        weights[index].dtype = dtype;
        assert!(
            DiagnosticTextBlock::new(
                &gpu,
                3,
                TextBlockWeights(std::array::from_fn(|i| &weights[i]))
            )
            .is_err()
        );
        assert_eq!(gpu.alloc_count(), 0);
    }
}
