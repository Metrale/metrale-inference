// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Launch/selection controls in separate processes for cached lever state.
//! Actual CUDA arithmetic and measured serving evidence are separate gates.

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};
use metrale_model_layers::layers::ops::{dense_gemv_batchm, dense_gemv_batchm_split};
use metrale_model_layers::weight_map::DenseWeight;

#[test]
fn dense_row_selection_preserves_defaults_and_refuses_missing_opt_in() {
    let Ok(mode) = std::env::var("DENSE_ROW_TEST_CHILD") else {
        for enabled in ["0", "1"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "dense_row_selection_preserves_defaults_and_refuses_missing_opt_in",
                    "--nocapture",
                ])
                .env("DENSE_ROW_TEST_CHILD", enabled)
                .env("METRALE_DENSE_GEMV_SMALL_ROWS", enabled)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("row-controls-complete"));
        }
        return;
    };
    assert!(mode == "0" || mode == "1");
    let enabled = mode == "1";
    let p = DevicePtr(0x1000);
    let w = DenseWeight { weight: p };
    for _ in 0..2 {
        let gpu = MockGpuBackend::new();
        for m in 1..=5 {
            dense_gemv_batchm(&gpu, KernelHandle(123), p, &w, p, m, 8, 64, 12, 71).unwrap();
        }
        dense_gemv_batchm_split(&gpu, KernelHandle(123), p, &w, p, 11, 3, 8, 64, 12, 71).unwrap();
        dense_gemv_batchm_split(&gpu, KernelHandle(123), p, &w, p, 47, 3, 8, 64, 12, 71).unwrap();
        let launches = gpu.launches_snapshot();
        assert_eq!(launches.len(), 7);
        for (i, launch) in launches.iter().enumerate() {
            let selected = enabled && (i < 4 || i == 5);
            assert_eq!(launch.func, if selected { 0xDEAD } else { 123 });
            assert_eq!(launch.grid, [2, if i >= 5 { 3 } else { 1 }, 1]);
            assert_eq!(launch.stream, 71);
        }
        assert_eq!(gpu.kernel_lookups_snapshot().len(), usize::from(enabled));
    }
    // Only selection is tested here: these unmeasured geometries keep the existing
    // launch contract and do not qualify its numerical behavior.
    for (n, k) in [(7, 64), (8, 63), (0, 64), (8, 0)] {
        let gpu = MockGpuBackend::new();
        dense_gemv_batchm(&gpu, KernelHandle(123), p, &w, p, 4, n, k, 12, 0).unwrap();
        assert_eq!(gpu.launches_snapshot()[0].func, 123);
        assert!(gpu.kernel_lookups_snapshot().is_empty());
    }
    for (m, y) in [(0, 1), (1, 0), (17, 1)] {
        let gpu = MockGpuBackend::new();
        assert!(
            dense_gemv_batchm_split(&gpu, KernelHandle(123), p, &w, p, m, y, 8, 64, 12, 0).is_err()
        );
        assert!(gpu.launches_snapshot().is_empty());
    }
    let gpu = MockGpuBackend::new();
    gpu.deny_kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm4");
    let result = dense_gemv_batchm(&gpu, KernelHandle(123), p, &w, p, 4, 8, 64, 12, 0);
    assert_eq!(result.is_err(), enabled);
    assert_eq!(gpu.launches_snapshot().len(), usize::from(!enabled));
    println!("row-controls-complete");
}
