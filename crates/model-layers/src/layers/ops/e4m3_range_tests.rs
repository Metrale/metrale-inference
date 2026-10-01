// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Debug builds refuse an activation an unscaled E4M3 cast would saturate, at
//! the launchers, and only for kernels that declare the cast.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::super::{
    E4m3Saturation, MOE_ROUTED_DOWN_KNOWN, NEMOTRON_SHARED_EXPERT_E4M3, allow_e4m3_saturation,
    e4m3_saturations, moe_w4a16_grouped_gemm_ptrtable_n128, w4a16_gemm, w4a16_gemm_n128,
};
use crate::weight_map::QuantizedWeight;

const E4M3_A: KernelHandle = KernelHandle(0xE4);
const BF16_A: KernelHandle = KernelHandle(0xBF);
const M: u32 = 4;
const K: u32 = 64;

fn gpu() -> MockGpuBackend {
    let gpu = MockGpuBackend::new();
    for k in [E4M3_A, BF16_A] {
        gpu.set_kernel_n_tile(k, 128);
    }
    gpu.set_kernel_a_e4m3(E4M3_A);
    gpu
}

/// 2026-09-29: An `[M, K]` BF16 activation of `0.5` everywhere but one `peak`.
fn activation(gpu: &MockGpuBackend, peak: f32) -> DevicePtr {
    let mut v = vec![0.5f32; (M * K) as usize];
    v[(M * K / 2 + 3) as usize] = peak;
    let bytes: Vec<u8> = v
        .iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let p = gpu.alloc(bytes.len()).unwrap();
    gpu.copy_h2d(&bytes, p).unwrap();
    p
}

fn launch(gpu: &MockGpuBackend, kernel: KernelHandle, input: DevicePtr) {
    let w = QuantizedWeight::null();
    w4a16_gemm_n128(gpu, kernel, input, &w, DevicePtr::NULL, M, 256, K, 0).unwrap();
}

/// 2026-09-29: 448 itself is representable; a kernel that keeps A in BF16 takes any value;
/// both launch.
#[test]
fn in_range_or_bf16_activations_launch() {
    let gpu = gpu();
    let before = gpu.launch_count();
    launch(&gpu, E4M3_A, activation(&gpu, 448.0));
    launch(&gpu, BF16_A, activation(&gpu, 5000.0));
    let w = QuantizedWeight::null();
    w4a16_gemm(
        &gpu,
        BF16_A,
        activation(&gpu, 5000.0),
        &w,
        DevicePtr::NULL,
        M,
        256,
        K,
        0,
    )
    .unwrap();
    assert_eq!(gpu.launch_count(), before + 3);
}

/// 2026-09-29: The refusals exist only in debug builds.
#[cfg(debug_assertions)]
mod debug {
    use super::*;
    use crate::layers::ops::{fp8_gemm_n128, w4a16_gemm_n128_m128};

    #[test]
    #[should_panic(expected = "unscaled E4M3 cast would saturate")]
    fn an_e4m3_kernel_fed_past_448_panics_in_debug_builds() {
        let gpu = gpu();
        // 2026-09-29: 450 is the next BF16 value above 448.
        let a = activation(&gpu, 450.0);
        launch(&gpu, E4M3_A, a);
    }

    #[test]
    #[should_panic(expected = "unscaled E4M3 cast would saturate")]
    fn a_negative_peak_counts_by_magnitude() {
        let gpu = gpu();
        let a = activation(&gpu, -512.0);
        let w = QuantizedWeight::null();
        w4a16_gemm_n128_m128(&gpu, E4M3_A, a, &w, DevicePtr::NULL, M, 256, K, 0).unwrap();
    }

    /// 2026-09-29: The FP8 activation path casts A with `bf16_to_fp8`, which checks its source
    /// whatever the GEMM kernel declares.
    #[test]
    #[should_panic(expected = "unscaled E4M3 cast would saturate")]
    fn the_fp8_activation_path_is_checked_without_a_declaration() {
        let gpu = gpu();
        let a = activation(&gpu, 600.0);
        fp8_gemm_n128(
            &gpu,
            BF16_A,
            a,
            DevicePtr::NULL,
            DevicePtr::NULL,
            M,
            256,
            K,
            0,
        )
        .unwrap();
    }
}

/// 2026-09-30: `[rows, K]` BF16 rows of `0.5`, with `peak` at row `peak_row`.
fn rows(gpu: &MockGpuBackend, n: usize, peak_row: usize, peak: f32) -> DevicePtr {
    let k = K as usize;
    let mut v = vec![0.5f32; n * k];
    v[peak_row * k + 7] = peak;
    let bytes: Vec<u8> = v
        .iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let p = gpu.alloc(bytes.len()).unwrap();
    gpu.copy_h2d(&bytes, p).unwrap();
    p
}

fn i32s(gpu: &MockGpuBackend, v: &[i32]) -> DevicePtr {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = gpu.alloc(bytes.len()).unwrap();
    gpu.copy_h2d(&bytes, p).unwrap();
    p
}

/// 2026-09-30: Two experts over 3 sorted positions (`0..2`, `2..3`) gathering token rows
/// `[4, 1, 2]` out of 6; rows 0, 3 and 5 are never read.
fn grouped(gpu: &MockGpuBackend, kernel: KernelHandle, a: DevicePtr, ids: bool) {
    let offsets = i32s(gpu, &[0, 2, 3]);
    let sorted = if ids {
        i32s(gpu, &[4, 1, 2])
    } else {
        DevicePtr::NULL
    };
    moe_w4a16_grouped_gemm_ptrtable_n128(
        gpu,
        kernel,
        a,
        DevicePtr::NULL,
        DevicePtr::NULL,
        DevicePtr::NULL,
        DevicePtr::NULL,
        offsets,
        sorted,
        2,
        256,
        K,
        1,
        0,
    )
    .unwrap();
}

/// 2026-09-30: The grouped check reads only the rows the kernel gathers, and only for a
/// kernel that declares the cast. Mutation: checking rows `0..=max` instead of the gathered
/// set panics on the unread row 5 below.
#[test]
fn a_grouped_launch_checks_only_the_rows_it_reads() {
    let gpu = gpu();
    let before = gpu.launch_count();
    grouped(&gpu, E4M3_A, rows(&gpu, 6, 3, 5000.0), true);
    grouped(&gpu, E4M3_A, rows(&gpu, 6, 5, 5000.0), false);
    grouped(&gpu, BF16_A, rows(&gpu, 6, 4, 5000.0), true);
    assert_eq!(gpu.launch_count(), before + 3);
}

/// 2026-09-30: A test-only disclosure, so the count is not shared with another test.
const TEST_SITE: E4m3Saturation = E4m3Saturation {
    site: "e4m3_range_tests",
    why: "test",
    bound: 600.0,
};

#[cfg(debug_assertions)]
mod disclosed {
    use super::*;

    /// 2026-09-30: Inside a scope, a saturation up to the bound launches and is counted once
    /// per launch; the opt-in flag's entry covers any magnitude.
    #[test]
    fn a_disclosed_saturation_is_counted_not_fatal() {
        let gpu = gpu();
        let before = gpu.launch_count();
        {
            let _s = allow_e4m3_saturation(TEST_SITE);
            launch(&gpu, E4M3_A, activation(&gpu, 510.0));
            grouped(&gpu, E4M3_A, rows(&gpu, 6, 4, 590.0), true);
        }
        assert_eq!(e4m3_saturations(TEST_SITE.site), 2);
        {
            let _s = allow_e4m3_saturation(NEMOTRON_SHARED_EXPERT_E4M3);
            launch(&gpu, E4M3_A, activation(&gpu, 60000.0));
        }
        assert!(e4m3_saturations(NEMOTRON_SHARED_EXPERT_E4M3.site) >= 1);
        assert_eq!(gpu.launch_count(), before + 3);
    }

    /// 2026-09-30: The known 35B value (510) is covered; nothing above the entry's bound is.
    #[test]
    fn the_known_routed_down_value_is_covered() {
        let gpu = gpu();
        let _s = allow_e4m3_saturation(MOE_ROUTED_DOWN_KNOWN);
        launch(&gpu, E4M3_A, activation(&gpu, 510.0));
    }

    #[test]
    #[should_panic(expected = "past the 512 that `MoE routed-down input` discloses")]
    fn past_a_disclosed_bound_still_panics() {
        let gpu = gpu();
        let _s = allow_e4m3_saturation(MOE_ROUTED_DOWN_KNOWN);
        launch(&gpu, E4M3_A, activation(&gpu, 520.0));
    }

    /// 2026-09-30: The allowance ends with its scope.
    #[test]
    #[should_panic(expected = "unscaled E4M3 cast would saturate")]
    fn a_dropped_scope_no_longer_covers() {
        let gpu = gpu();
        drop(allow_e4m3_saturation(TEST_SITE));
        launch(&gpu, E4M3_A, activation(&gpu, 510.0));
    }

    /// 2026-09-30: The grouped check panics on an undeclared saturation in a row it reads.
    #[test]
    #[should_panic(expected = "unscaled E4M3 cast would saturate")]
    fn a_grouped_launch_panics_on_a_gathered_row() {
        let gpu = gpu();
        grouped(&gpu, E4M3_A, rows(&gpu, 6, 4, 5000.0), true);
    }

    /// 2026-09-30: Without `sorted_token_ids` the rows are the positions themselves.
    #[test]
    #[should_panic(expected = "unscaled E4M3 cast would saturate")]
    fn a_grouped_launch_without_ids_checks_the_positions() {
        let gpu = gpu();
        grouped(&gpu, E4M3_A, rows(&gpu, 6, 2, 5000.0), false);
    }
}
