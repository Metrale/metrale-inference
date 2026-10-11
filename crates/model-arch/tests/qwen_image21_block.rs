// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Complete diagnostic block composition and ownership, no GPU math.
use metrale_gpu_runtime::gpu::{DevicePtr, mock::MockGpuBackend};
use metrale_model_arch::qwen_image21::{block::DiagnosticImageBlock, rope::ImageRopeLayout};
use metrale_model_weights::{
    qwen_image21::{Block, Config},
    weights::{WeightDtype, WeightTensor},
};
fn fixture() -> (Config, Vec<WeightTensor>) {
    let config = Config::parse(include_str!(
        "../../model-weights/tests/fixtures/qwen-image21/config.json"
    ))
    .unwrap();
    let shapes = vec![
        vec![4096, 4096],
        vec![4096, 4096],
        vec![4096, 4096],
        vec![4096, 4096],
        vec![128],
        vec![128],
        vec![12288, 4096],
        vec![12288, 4096],
        vec![4096, 12288],
    ];
    (
        config,
        shapes
            .into_iter()
            .enumerate()
            .map(|(i, shape)| WeightTensor {
                ptr: DevicePtr((i as u64 + 1) << 30),
                shape,
                dtype: WeightDtype::BF16,
            })
            .collect(),
    )
}
fn bind(w: &[WeightTensor]) -> Block<'_, WeightTensor> {
    Block {
        q: &w[0],
        k: &w[1],
        v: &w[2],
        o: &w[3],
        q_norm: &w[4],
        k_norm: &w[5],
        gate: &w[6],
        up: &w[7],
        down: &w[8],
    }
}
#[test]
fn full_block_composes_all_stages_and_releases_owned_scratch() {
    let (config, w) = fixture();
    let gpu = MockGpuBackend::new();
    let before = gpu.alloc_count();
    {
        let mut block = DiagnosticImageBlock::new(
            &config,
            &bind(&w),
            &gpu,
            2,
            &[-1, 0, 0],
            &[true; 6],
            Some(&[false, true, true]),
        )
        .unwrap();
        assert_eq!(gpu.alloc_count(), before + 3);
        let rope = ImageRopeLayout::new(&[false, true, true], &[[1, 1, 2]]).unwrap();
        let output = block
            .forward(DevicePtr(1 << 24), DevicePtr(1 << 25), &rope, 7)
            .unwrap();
        let launches = gpu.launches_snapshot();
        assert_eq!(launches.len(), 28);
        assert!(launches.iter().all(|l| l.stream == 7));
        assert_eq!(launches[24].grid, [96, 1, 1]); // second FFN projection, width12288
        assert_eq!(launches[25].grid, [288, 1, 1]); // staged activation:6*12288 elements
        assert_eq!(launches[26].grid, [32, 1, 1]); // down projection to4096
        assert_eq!(
            block.forward(output, DevicePtr(1 << 25), &rope, 7).unwrap(),
            output
        );
        assert_eq!(gpu.launches_snapshot().len(), 56);
    }
    assert_eq!(gpu.alloc_count(), before);
}
#[test]
fn full_block_refuses_wrong_weights_and_rope_visibility_before_launch() {
    let (config, mut w) = fixture();
    let gpu = MockGpuBackend::new();
    w[8].shape = vec![12288, 4096];
    assert!(
        DiagnosticImageBlock::new(&config, &bind(&w), &gpu, 1, &[-1, 0, 0], &[true; 3], None)
            .is_err()
    );
    assert_eq!(gpu.alloc_count(), 0);
    w[8].shape = vec![4096, 12288];
    let mut block =
        DiagnosticImageBlock::new(&config, &bind(&w), &gpu, 1, &[-1, 0, 0], &[true; 3], None)
            .unwrap();
    let wrong = ImageRopeLayout::new(&[false, false, false], &[]).unwrap();
    assert!(
        block
            .forward(DevicePtr(1 << 24), DevicePtr(1 << 25), &wrong, 0)
            .is_err()
    );
    let split = ImageRopeLayout::new(&[false, true, true], &[[1, 1, 1], [1, 1, 1]]).unwrap();
    assert!(
        block
            .forward(DevicePtr(1 << 24), DevicePtr(1 << 25), &split, 0)
            .is_err()
    );
    assert!(gpu.launches_snapshot().is_empty());
}
