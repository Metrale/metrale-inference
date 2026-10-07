// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-06: Refuse invalid geometry/aliasing before dispatch, preserve stream.
#[path = "../src/qwen_image21/vae_attention.rs"]
pub mod attention;
use metrale_gpu_runtime::gpu::{DevicePtr, mock::MockGpuBackend};
#[test]
fn refusals_precede_valid_wide_head_dispatch() {
    let gpu = MockGpuBackend::new();
    for (q, o, c, n) in [
        (0, 1 << 30, 1152, 3),
        (4097, 1 << 30, 1152, 3),
        (4096, 1 << 30, 1024, 3),
        (4096, 1 << 30, 1152, 0),
        (4096, 1 << 30, 1152, 16385),
        (4096, 4100, 1152, 3),
        (u64::MAX - 3, 4096, 1152, 3),
    ] {
        assert!(attention::attention_f32(&gpu, DevicePtr(q), DevicePtr(o), c, n, 7).is_err());
    }
    assert!(gpu.launches_snapshot().is_empty());
    attention::attention_f32(&gpu, DevicePtr(4096), DevicePtr(1 << 30), 1152, 3, 7).unwrap();
    let launches = gpu.launches_snapshot();
    assert_eq!(launches.len(), 1);
    assert_eq!(launches[0].stream, 7);
}
