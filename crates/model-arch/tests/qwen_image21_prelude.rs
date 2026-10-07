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

#[test]
fn attention_reuses_gather_prefixes_and_zeroes_fully_masked_queries() {
    use metrale_model_arch::qwen_image21::{ImageQkvBuffers, attention::DiagnosticBlockAttention};
    let gpu = MockGpuBackend::new();
    let before = gpu.alloc_count();
    {
        let mut attention = DiagnosticBlockAttention::new(
            &gpu,
            2,
            &[-1, 0, 0],
            &[false, true, true, true, false, true],
        )
        .unwrap();
        let qkv = ImageQkvBuffers {
            q: DevicePtr(1 << 20),
            k: DevicePtr(1 << 21),
            v: DevicePtr(1 << 22),
        };
        attention.forward(qkv, 7).unwrap();
        let launches = gpu.launches_snapshot();
        // Two shared gathers, then five nonempty queries; sample0 query0 is zero.
        assert_eq!(launches.len(), 7);
        assert_eq!(launches[0].grid, [4, 1, 1]);
        assert_eq!(launches[1].grid, [4, 1, 1]);
        assert!(
            launches[2..]
                .iter()
                .all(|l| l.grid == [32, 1, 1] && l.block == [128, 1, 1])
        );
        assert!(launches.iter().all(|l| l.stream == 7));
        assert_eq!(gpu.alloc_count(), before + 1);
    }
    assert_eq!(gpu.alloc_count(), before);
    for (ids, valid) in [
        (vec![-2], vec![true]),
        (vec![0, -1, 0], vec![true; 3]),
        (vec![-1], vec![]),
    ] {
        assert!(DiagnosticBlockAttention::new(&gpu, 1, &ids, &valid).is_err());
    }
    let mut empty = DiagnosticBlockAttention::new(&gpu, 1, &[-1], &[false]).unwrap();
    let old = gpu.launches_snapshot().len();
    empty
        .forward(
            ImageQkvBuffers {
                q: DevicePtr(2),
                k: DevicePtr(4),
                v: DevicePtr(6),
            },
            0,
        )
        .unwrap();
    assert_eq!(gpu.launches_snapshot().len(), old);
}

#[test]
fn attention_projection_requires_rotated_inputs_and_matching_geometry() {
    use metrale_model_arch::qwen_image21::{
        attention::DiagnosticBlockAttention, rope::ImageRopeLayout,
    };
    let (config, weight) = fixture();
    let b = block(&weight);
    let gpu = MockGpuBackend::new();
    let mut p = DiagnosticImagePrelude::new(&config, &b, &gpu, 1, 3, None).unwrap();
    let mut attention = DiagnosticBlockAttention::new(&gpu, 1, &[-1, 0, 0], &[true; 3]).unwrap();
    assert!(p.attend_project(&mut attention, &weight, 0).is_err());
    p.project(DevicePtr(1 << 20), DevicePtr(1 << 21), 0)
        .unwrap();
    let norm = WeightTensor {
        ptr: DevicePtr(1 << 30),
        shape: vec![128],
        dtype: WeightDtype::BF16,
    };
    p.normalize_qk(&norm, &norm, 0).unwrap();
    p.rotate_qk(
        &ImageRopeLayout::new(&[false, true, true], &[[1, 1, 2]]).unwrap(),
        0,
    )
    .unwrap();
    let mut wrong = DiagnosticBlockAttention::new(&gpu, 1, &[-1], &[true]).unwrap();
    assert!(p.attend_project(&mut wrong, &weight, 0).is_err());
    let other_gpu = MockGpuBackend::new();
    let mut other = DiagnosticBlockAttention::new(&other_gpu, 1, &[-1, 0, 0], &[true; 3]).unwrap();
    assert!(p.attend_project(&mut other, &weight, 0).is_err());
    let before = gpu.launches_snapshot().len();
    p.attend_project(&mut attention, &weight, 0).unwrap();
    assert_eq!(gpu.launches_snapshot().len() - before, 6); // gathers + 3 queries + projection
    assert!(p.attend_project(&mut attention, &weight, 0).is_err());
}
