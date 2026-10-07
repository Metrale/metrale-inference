// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Explicit diagnostic capacity accounting; serving admission stays unchanged.
use metrale_gpu_runtime::gpu::{GpuBackend, mock::MockGpuBackend};
use metrale_model_arch::weight_loader::gpt_oss::runtime::PrefillScratch;

#[test]
fn explicit_larger_scratch_matches_budget_and_releases_every_byte() {
    for (rows, expected) in [(16, 2_837_120), (64, 11_348_096), (128, 22_696_064)] {
        let gpu = MockGpuBackend::new();
        let before = gpu.live_bytes().unwrap();
        assert_eq!(
            PrefillScratch::required_bytes(rows, 8192).unwrap(),
            expected
        );
        let mut scratch = PrefillScratch::new(&gpu, rows, 8192).unwrap();
        assert_eq!(gpu.live_bytes().unwrap() - before, expected);
        scratch.release(&gpu, 0).unwrap();
        assert_eq!(gpu.live_bytes().unwrap(), before);
    }
}

#[test]
fn unsupported_capacity_refuses_before_allocating() {
    let gpu = MockGpuBackend::new();
    for (rows, blocks) in [(0, 8), (129, 8), (usize::MAX, 8), (128, 0), (128, 131073)] {
        assert!(PrefillScratch::required_bytes(rows, blocks).is_err());
        assert!(PrefillScratch::new(&gpu, rows, blocks).is_err());
    }
    assert_eq!(gpu.live_bytes().unwrap(), 0);
}
