// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Isolate launch admission tests from unrelated CUDA-only unit tests.
use metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M;
#[allow(dead_code)]
#[path = "../src/layers/ops/dense_batchm_fp32.rs"]
mod implementation;

// 2026-10-07: Inspect actual production launches, including a partial final row tile.
#[test]
fn every_row_is_covered_without_exceeding_the_shared_cta_capacity() {
    use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle, mock::MockGpuBackend};
    for m in [1u32, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128] {
        let gpu = MockGpuBackend::new();
        implementation::dense_gemv_batchm_fp32(
            &gpu,
            KernelHandle(1),
            DevicePtr(0x10000),
            DevicePtr(0x20000),
            DevicePtr(0x30000),
            m,
            32,
            64,
            36,
            7,
        )
        .unwrap();
        let calls = gpu.launches_snapshot();
        assert_eq!(calls.len(), 1);
        let launch = &calls[0];
        assert_eq!(launch.grid, [8, m.div_ceil(16), 1]);
        assert_eq!(launch.block, [256, 1, 1]);
        assert_eq!(launch.stream, 7);
        let rows_per_y = m.div_ceil(launch.grid[1]);
        assert!(rows_per_y <= 16);
        let covered: Vec<_> = (0..launch.grid[1])
            .flat_map(|y| y * rows_per_y..((y + 1) * rows_per_y).min(m))
            .collect();
        assert_eq!(covered, (0..m).collect::<Vec<_>>());
    }
}
