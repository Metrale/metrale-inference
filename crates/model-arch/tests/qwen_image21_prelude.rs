// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Native composition/ownership checks; mock does no arithmetic.
use metrale_gpu_runtime::gpu::{DevicePtr, mock::MockGpuBackend};
use metrale_model_arch::qwen_image21::DiagnosticImagePrelude;
use metrale_model_weights::{
    qwen_image21::{Block, Config},
    weights::{WeightDtype, WeightTensor},
};
fn fixture() -> (Config, WeightTensor) {
    let c = Config::parse(include_str!(
        "../../model-weights/tests/fixtures/qwen-image21/config.json"
    ))
    .unwrap();
    (
        c,
        WeightTensor {
            ptr: DevicePtr(1 << 32),
            shape: vec![4096, 4096],
            dtype: WeightDtype::BF16,
        },
    )
}
fn block(w: &WeightTensor) -> Block<'_, WeightTensor> {
    Block {
        q: w,
        k: w,
        v: w,
        o: w,
        q_norm: w,
        k_norm: w,
        gate: w,
        up: w,
        down: w,
    }
}
#[test]
fn prelude_composes_native_norm_modulation_and_three_projections() {
    let (config, w) = fixture();
    let b = block(&w);
    let gpu = MockGpuBackend::new();
    let before = gpu.alloc_count();
    {
        let mut p =
            DiagnosticImagePrelude::new(&config, &b, &gpu, 2, 3, Some(&[false, true, true]))
                .unwrap();
        assert_eq!(gpu.alloc_count(), before + 1);
        let qkv = p
            .project(DevicePtr(1 << 24), DevicePtr(1 << 25), 9)
            .unwrap();
        assert_ne!(qkv.q, qkv.k);
        assert_ne!(qkv.k, qkv.v);
        let launches = gpu.launches_snapshot();
        assert_eq!(launches.len(), 5);
        assert!(launches.iter().all(|l| l.stream == 9));
        assert_eq!(launches[0].grid, [6, 1, 1]);
        assert_eq!(launches[0].shared_mem, 1024);
        assert_eq!(launches[1].grid, [96, 1, 1]);
        assert!(
            launches[2..]
                .iter()
                .all(|l| l.grid == [32, 1, 1] && l.block == [256, 1, 1])
        );
        assert!(p.project(DevicePtr(0), DevicePtr(1 << 25), 9).is_err());
        assert_eq!(gpu.launches_snapshot().len(), 5);
    }
    assert_eq!(gpu.alloc_count(), before);
}
#[test]
fn missing_native_kernel_and_wrong_projection_dtype_refuse() {
    let (config, mut w) = fixture();
    let gpu = MockGpuBackend::new();
    gpu.deny_kernel("dense_gemm_bf16", "dense_gemm_bf16_pipelined");
    assert!(DiagnosticImagePrelude::new(&config, &block(&w), &gpu, 1, 1, None).is_err());
    assert_eq!(gpu.alloc_count(), 0);
    let gpu = MockGpuBackend::new();
    w.dtype = WeightDtype::UInt8;
    assert!(DiagnosticImagePrelude::new(&config, &block(&w), &gpu, 1, 1, None).is_err());
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn head_norm_stages_bf16_and_refuses_double_normalization() {
    let (config, w) = fixture();
    let b = block(&w);
    let gpu = MockGpuBackend::new();
    let mut p = DiagnosticImagePrelude::new(&config, &b, &gpu, 2, 3, None).unwrap();
    let norm = WeightTensor {
        ptr: DevicePtr(2 << 32),
        shape: vec![128],
        dtype: WeightDtype::BF16,
    };
    assert!(p.normalize_qk(&norm, &norm, 0).is_err());
    p.project(DevicePtr(1 << 24), DevicePtr(1 << 25), 3)
        .unwrap();
    p.normalize_qk(&norm, &norm, 3).unwrap();
    let launches = gpu.launches_snapshot();
    assert_eq!(launches.len(), 9);
    assert_eq!(launches[5].grid, [192, 1, 1]);
    assert_eq!(launches[5].block, [128, 1, 1]);
    assert_eq!(launches[6].grid, [96, 1, 1]);
    assert!(p.normalize_qk(&norm, &norm, 0).is_err());
    assert_eq!(gpu.launches_snapshot().len(), 9);
    use metrale_model_arch::qwen_image21::rope::ImageRopeLayout;
    let wrong = ImageRopeLayout::new(&[false], &[]).unwrap();
    assert!(p.rotate_qk(&wrong, 3).is_err());
    let layout = ImageRopeLayout::new(&[false, true, false], &[[1, 1, 1]]).unwrap();
    p.rotate_qk(&layout, 3).unwrap();
    assert_eq!(gpu.launches_snapshot().len(), 11);
    assert_eq!(gpu.launches_snapshot()[9].grid, [48, 1, 1]);
    assert!(p.rotate_qk(&layout, 3).is_err());
}

#[test]
fn rotary_geometry_centers_odd_grids_and_advances_text_correctly() {
    use metrale_model_arch::qwen_image21::rope::ImageRopeLayout;
    let mask = [
        false, false, true, true, true, true, true, true, false, true, true, false,
    ];
    let layout = ImageRopeLayout::new(&mask, &[[1, 3, 2], [1, 2, 1]]).unwrap();
    assert_eq!(
        layout.positions(),
        &[
            [0, 0, 0],
            [1, 1, 1],
            [2, -2, -1],
            [2, -2, 0],
            [2, -1, -1],
            [2, -1, 0],
            [2, 0, -1],
            [2, 0, 0],
            [5, 5, 5],
            [6, -1, -1],
            [6, 0, -1],
            [8, 8, 8]
        ]
    );
    assert_eq!(layout.frequencies().len(), mask.len() * 128);
    assert_eq!(&layout.frequencies()[..4], &[1.0, 0.0, 1.0, 0.0]);
    assert!(ImageRopeLayout::new(&[true, false], &[[1, 1, 2]]).is_err());
    assert!(ImageRopeLayout::new(&[true], &[]).is_err());
    assert!(ImageRopeLayout::new(&[true], &[[2, 1, 1]]).is_err());
    assert!(ImageRopeLayout::new(&vec![false; 8193], &[]).is_err());
}
