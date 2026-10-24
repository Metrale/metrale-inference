// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Correctness and speed of the int8 prefill GEMM in the `w4a16` module
//! (`int8_gemm_faith2` and the `requant_*` kernels it runs after).
//!
//! Owner: model-arch examples (int8 GEMM kernels).
//! Invariants: none beyond the types. Each correctness arm prints a cosine against a host
//! reference and PASS above 0.999; a FAIL is printed, not fatal.
//!
//! The kernels are only in `kernels/gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu`, so the build's first
//! target, which `ptx_modules()` loads, must be the qwen3.6-27b one. The speed section times two
//! prefill shapes and prints TFLOP/s.

mod checks_faith;
mod requant_e2e;
mod speed;

use anyhow::Result;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use checks_faith::check_faith2;
use requant_e2e::requant_e2e;
use speed::speed_shape;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn i8(&mut self) -> i8 {
        (self.next_u64() % 255) as i8 - 127
    }
    fn pos_scale(&mut self) -> f32 {
        0.001 + (self.next_u64() % 1000) as f32 * 0.0005
    }
}

fn up(gpu: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(b, p)?;
    Ok(p)
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let hfaith2 = gpu.kernel("w4a16", "int8_gemm_faith2")?;
    let hreqw = gpu.kernel("w4a16", "requant_w_nvfp4_int8")?;
    let hreqa = gpu.kernel("w4a16", "requant_a_bf16_int8")?;

    let (m, n, k) = (128usize, 256usize, 512usize);
    let nb = k / 32;
    let mut rng = Rng(0xC0FFEE);
    let a_i8: Vec<i8> = (0..m * k).map(|_| rng.i8()).collect();
    let b_i8: Vec<i8> = (0..n * k).map(|_| rng.i8()).collect();
    let a_sc: Vec<f32> = (0..m * nb).map(|_| rng.pos_scale()).collect();
    let b_sc: Vec<f32> = (0..n * nb).map(|_| rng.pos_scale()).collect();

    // 2026-09-25: Host reference: C[m,n] = sum over 32-wide blocks of
    // (integer sum of A[m,k] * B[n,k]) * As[m,blk] * Bs[n,blk].
    let mut c_ref = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f32;
            for blk in 0..nb {
                let mut s = 0i32;
                for kk in 0..32 {
                    let ki = blk * 32 + kk;
                    s += a_i8[mi * k + ki] as i32 * b_i8[ni * k + ki] as i32;
                }
                acc += s as f32 * a_sc[mi * nb + blk] * b_sc[ni * nb + blk];
            }
            c_ref[mi * n + ni] = acc;
        }
    }

    let a_p = up(gpu, bytemuck_i8(&a_i8))?;
    let b_p = up(gpu, bytemuck_i8(&b_i8))?;
    let as_p = up(gpu, bytemuck_f32(&a_sc))?;
    let bs_p = up(gpu, bytemuck_f32(&b_sc))?;
    check_faith2(gpu, stream, hfaith2, a_p, b_p, as_p, bs_p, &c_ref, m, n, k)?;

    // 2026-09-25: End to end: NVFP4 weights through `requant_w_nvfp4_int8` (per-16 E4M3 scales
    // re-blocked to per-32 int8 scales) and BF16 activations through `requant_a_bf16_int8`,
    // then int8_gemm_faith2, against a host GEMM on the dequantised weights.
    requant_e2e(gpu, stream, hreqw, hreqa, hfaith2)?;

    println!("\n=== int8 speed (TFLOP/s) ===");
    for &(label, m, n, k) in &[
        ("gate/up M=4096", 4096usize, 17408usize, 5120usize),
        ("down    M=4096", 4096, 5120, 17408),
    ] {
        speed_shape(gpu, stream, hfaith2, label, m, n, k)?;
    }
    Ok(())
}

fn bytemuck_i8(v: &[i8]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) }
}
fn bytemuck_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
