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
