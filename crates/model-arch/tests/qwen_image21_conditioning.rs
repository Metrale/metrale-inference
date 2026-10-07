// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Time-conditioning geometry, storage and scheduler-input refusal.
use metrale_gpu_runtime::gpu::{DevicePtr, mock::MockGpuBackend};
use metrale_model_arch::qwen_image21::conditioning::{
    ConditioningWeights, DiagnosticConditioning, timestep_frequencies,
};
use metrale_model_weights::weights::{WeightDtype, WeightTensor};
#[test]
fn conditioning_owns_time_zero_row_and_seven_native_stages() {
    let gpu = MockGpuBackend::new();
    let before = gpu.alloc_count();
    let weights: Vec<_> = [[4096, 256], [4096, 4096], [16384, 4096], [4096, 4096]]
        .into_iter()
        .enumerate()
        .map(|(i, s)| WeightTensor {
            ptr: DevicePtr((i as u64 + 1) << 30),
            shape: s.to_vec(),
            dtype: WeightDtype::BF16,
        })
        .collect();
    {
        let mut c = DiagnosticConditioning::new(
            &gpu,
            2,
            ConditioningWeights {
                time_in: &weights[0],
                time_out: &weights[1],
                modulation: &weights[2],
                final_scale: &weights[3],
            },
        )
        .unwrap();
        assert_eq!(gpu.alloc_count(), before + 1);
        for bad in [
            vec![0.5],
            vec![f32::NAN, 0.5],
            vec![-0.1, 0.5],
            vec![0.5, 1.1],
        ] {
            assert!(c.forward(&bad, 9).is_err());
        }
        assert!(gpu.launches_snapshot().is_empty());
        let outputs = c.forward(&[0.0, 1.0], 9).unwrap();
        let launches = gpu.launches_snapshot();
        assert_eq!(launches.len(), 7);
        assert_eq!(launches[0].grid, [3, 1, 1]);
        assert_eq!(launches[0].block, [256, 1, 1]);
        assert_eq!(launches[5].grid, [128, 1, 1]);
        assert!(launches.iter().all(|l| l.stream == 9));
        assert_ne!(outputs.modulation, outputs.final_scale);
        assert_ne!(outputs.time_embedding, outputs.final_scale);
    }
    assert_eq!(gpu.alloc_count(), before);
}
#[test]
fn time_frequency_layout_is_positive_decreasing_and_starts_at_one() {
    let frequencies = timestep_frequencies();
    assert_eq!(frequencies.len(), 128);
    assert_eq!(frequencies[0], 1.0);
    assert!(frequencies.iter().all(|f| f.is_finite() && *f > 0.0));
    assert!(frequencies.windows(2).all(|p| p[0] > p[1]));
}
