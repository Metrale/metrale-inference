// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Owned scratch, borrowed weights and executable IO stage composition.
use metrale_gpu_runtime::gpu::{DevicePtr, mock::MockGpuBackend};
use metrale_model_arch::qwen_image21::{
    io::{DiagnosticVisualIo, ProjectionWeights},
    layout::JointLayout,
};
use metrale_model_weights::weights::{WeightDtype, WeightTensor};
#[test]
fn visual_io_composes_native_projections_and_target_gather() {
    let gpu = MockGpuBackend::new();
    let layout = JointLayout::new(2, 1, &[false, true], &[[1, 2, 2]], &[true, false]).unwrap();
    let weights: Vec<_> = [
        vec![4096, 64],
        vec![4096],
        vec![4096, 4096],
        vec![4096, 4096],
        vec![64, 4096],
    ]
    .into_iter()
    .enumerate()
    .map(|(i, shape)| WeightTensor {
        ptr: DevicePtr((i as u64 + 1) << 30),
        shape,
        dtype: WeightDtype::BF16,
    })
    .collect();
    let before = gpu.alloc_count();
    {
        let mut io = DiagnosticVisualIo::new(
            &gpu,
            &layout,
            ProjectionWeights {
                image_in: &weights[0],
                text_norm: &weights[1],
                text_in: &weights[2],
                text_out: &weights[3],
                image_out: &weights[4],
            },
        )
        .unwrap();
        assert_eq!(gpu.alloc_count(), before + 1);
        assert!(io.input_project(DevicePtr(0), DevicePtr(8192), 7).is_err());
        assert!(io.output_project(DevicePtr(0), DevicePtr(8192), 7).is_err());
        assert!(gpu.launches_snapshot().is_empty());
        let joint = io
            .input_project(DevicePtr(1 << 40), DevicePtr(2 << 40), 7)
            .unwrap();
        assert_eq!(gpu.launches_snapshot().len(), 6);
        let target = io.output_project(joint, DevicePtr(3 << 40), 7).unwrap();
        assert_ne!(joint, target);
        assert_eq!(gpu.launches_snapshot().len(), 10);
        assert!(gpu.launches_snapshot().iter().all(|l| l.stream == 7));
    }
    assert_eq!(gpu.alloc_count(), before);
}
