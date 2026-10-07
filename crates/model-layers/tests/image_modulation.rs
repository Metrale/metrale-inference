// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Selection/ABI controls; GPU arithmetic needs the separate parity run.
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops::{ImageModulationLayout, image_modulation_scale_bf16};
#[test]
fn timestep_selection_preserves_sample_and_prefix_rows() {
    let layout =
        ImageModulationLayout::new(2, 4, 4096, 4, 2, Some(&[false, true, false, true])).unwrap();
    assert_eq!(layout.selected_rows(), &[2, 0, 2, 0, 2, 1, 2, 1]);
    let unsplit = ImageModulationLayout::new(2, 4, 4096, 1, 0, None).unwrap();
    assert_eq!(unsplit.selected_rows(), &[0, 0, 0, 0, 1, 1, 1, 1]);
    // Known-bad all-target selection cannot match the causal-prefix contract.
    assert_ne!(layout.selected_rows(), unsplit.selected_rows());
}
#[test]
fn invalid_layouts_and_null_buffers_refuse_before_launch() {
    assert!(ImageModulationLayout::new(0, 1, 4096, 4, 0, None).is_err());
    assert!(ImageModulationLayout::new(1, 2, 4096, 4, 0, Some(&[true])).is_err());
    assert!(ImageModulationLayout::new(1, 2, 4096, 4, 4, None).is_err());
    assert!(ImageModulationLayout::new(u32::MAX, 2, 4096, 4, 0, None).is_err());
    assert!(ImageModulationLayout::new(1, 1, u32::MAX, 4, 0, None).is_err());
    let gpu = MockGpuBackend::new();
    let layout = ImageModulationLayout::new(1, 1, 4096, 4, 0, None).unwrap();
    let kernel = gpu
        .kernel("image_modulation", "image_modulation_scale_bf16")
        .unwrap();
    assert!(image_modulation_scale_bf16(&gpu, kernel, &layout, [DevicePtr(0); 4], 0).is_err());
}

#[test]
fn selected_component_and_abi_are_explicit() {
    use metrale_gpu_runtime::gpu::mock::MockArg;
    let gpu = MockGpuBackend::new();
    let layout = ImageModulationLayout::new(2, 3, 4096, 4, 2, None).unwrap();
    let kernel = gpu
        .kernel("image_modulation", "image_modulation_scale_bf16")
        .unwrap();
    let buffers = [
        DevicePtr(4096),
        DevicePtr(8192),
        DevicePtr(12288),
        DevicePtr(16384),
    ];
    image_modulation_scale_bf16(&gpu, kernel, &layout, buffers, 7).unwrap();
    let launches = gpu.launches_snapshot();
    assert_eq!(launches.len(), 1);
    let launch = &launches[0];
    assert_eq!(launch.grid, [96, 1, 1]);
    assert_eq!(launch.stream, 7);
    let mut expected: Vec<_> = buffers.into_iter().map(MockArg::Buffer).collect();
    expected.extend([6u32, 4096, 16384, 8192].map(|v| MockArg::Bytes(v.to_ne_bytes().to_vec())));
    assert_eq!(launch.args, expected);
}
