// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Debug builds refuse an activation an unscaled E4M3 cast would saturate, at
//! the launchers, and only for kernels that declare the cast.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::super::{w4a16_gemm, w4a16_gemm_n128};
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
