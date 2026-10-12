// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Diagnostic constructor accounting and rollback; default scratch stays unchanged.
use metrale_gpu_runtime::gpu::{GpuBackend, mock::MockGpuBackend};
use metrale_model_arch::weight_loader::gpt_oss::runtime::PrefillScratch;
#[test]
fn missing_tc_modules_release_the_already_allocated_base() {
    for (module, kernel) in [
        (
            "gpt_oss_mxfp4_mma",
            "moe_w4a16_grouped_gemm_ptrtable_e8m0_gpt",
        ),
        ("moe_v41", "moe_v41_gather_rows"),
    ] {
        let gpu = MockGpuBackend::new();
        gpu.deny_kernel(module, kernel);
        assert!(PrefillScratch::new_packed_tc_diagnostic(&gpu, 17).is_err());
        assert_eq!(gpu.live_bytes(), Some(0));
        assert_eq!(gpu.live_alloc_count(), 0);
        let mut default = PrefillScratch::new(&gpu, 128, 17).unwrap();
        default.release(&gpu, 0).unwrap();
        assert_eq!(gpu.live_bytes(), Some(0));
    }
}
#[test]
fn explicit_tc_accounts_every_byte_and_releases_idempotently() {
    let gpu = MockGpuBackend::new();
    let mut scratch = PrefillScratch::new_packed_tc_diagnostic(&gpu, 17).unwrap();
    let extra = 4 * 128 * 5760 * 2 + 256 + 256 + 128 + 144 + 3 * 4 * 128 * 4;
    assert_eq!(PrefillScratch::packed_tc_extra_bytes(), extra);
    assert_eq!(
        gpu.live_bytes(),
        Some(PrefillScratch::required_bytes(128, 17).unwrap() + extra)
    );
    assert_eq!(gpu.live_alloc_count(), 2);
    scratch.release(&gpu, 0).unwrap();
    scratch.release(&gpu, 0).unwrap();
    assert_eq!(gpu.live_bytes(), Some(0));
}
