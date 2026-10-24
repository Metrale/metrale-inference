// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The end-to-end requant arm: NVFP4 weights and BF16 activations through the
//! `requant_*` kernels, then `int8_gemm_faith2`, against a host GEMM on the dequantised
//! weights.
//!
//! Owner: model-arch examples (int8 GEMM kernels).
//! Invariants: every buffer the arm allocates is freed before it returns `Ok`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::{Rng, bf16_bits_to_f32, up};

pub(super) fn requant_e2e(
    gpu: &dyn GpuBackend,
    stream: u64,
    hreqw: KernelHandle,
    hreqa: KernelHandle,
    hfaith2: KernelHandle,
) -> Result<()> {
    let (m2, n2, k2) = (256usize, 256usize, 512usize);
    let nb2 = k2 / 32;
    let mut rng = Rng(0xBEEF);
    // 2026-09-25: NVFP4 weights: packed E2M1 nibbles [n2, k2/2], per-16 E4M3 scales
    // [n2, k2/16], and one scale2.
    let e2m1_lut = [
        0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let e4m3_dec = |b: u8| -> f32 {
        let s = (b >> 7) & 1;
        let e = (b >> 3) & 0xF;
        let mm = b & 0x7;
        let v = if e == 0 {
            (mm as f32) * 0.001953125
        } else {
            (1.0 + (mm as f32) / 8.0) * 2f32.powi(e as i32 - 7)
        };
        if s == 1 { -v } else { v }
    };
    let scale2 = 0.05f32;
    let nibbles: Vec<u8> = (0..n2 * k2).map(|_| (rng.next_u64() % 16) as u8).collect();
    let mut packed = vec![0u8; n2 * k2 / 2];
    for i in 0..n2 * k2 / 2 {
        packed[i] = nibbles[2 * i] | (nibbles[2 * i + 1] << 4);
    }
    // 2026-09-25: Scale bytes with exponent 1..=14, any mantissa and sign 0: positive,
    // normal and finite.
    let e4m3_bytes: Vec<u8> = (0..n2 * (k2 / 16))
        .map(|_| {
            let e = 1 + (rng.next_u64() % 14) as u8;
            let mm = (rng.next_u64() % 8) as u8;
            (e << 3) | mm
        })
        .collect();
    let mut w_real = vec![0f32; n2 * k2];
    for ni in 0..n2 {
        for ki in 0..k2 {
            let nib = nibbles[ni * k2 + ki] as usize;
            let s16 = e4m3_dec(e4m3_bytes[ni * (k2 / 16) + ki / 16]) * scale2;
            w_real[ni * k2 + ki] = e2m1_lut[nib] * s16;
        }
    }
    // 2026-09-25: BF16 activations truncated from f32; the reference uses the truncated
    // values.
    let a_f: Vec<f32> = (0..m2 * k2).map(|_| (rng.i8() as f32) * 0.02).collect();
    let a_bf16_bits: Vec<u16> = a_f.iter().map(|&x| (x.to_bits() >> 16) as u16).collect();
    let a_bf16_round: Vec<f32> = a_bf16_bits.iter().map(|&b| bf16_bits_to_f32(b)).collect();
    let mut cref = vec![0f32; m2 * n2];
    for mi in 0..m2 {
        for ni in 0..n2 {
            let mut acc = 0f32;
            for ki in 0..k2 {
                acc += a_bf16_round[mi * k2 + ki] * w_real[ni * k2 + ki];
            }
            cref[mi * n2 + ni] = acc;
        }
    }
    let packed_p = up(gpu, &packed)?;
    let e4m3_p = up(gpu, &e4m3_bytes)?;
    let a_bf16_u8: Vec<u8> = a_bf16_bits.iter().flat_map(|&b| b.to_le_bytes()).collect();
    let abf_p = up(gpu, &a_bf16_u8)?;
    let wi8_p = gpu.alloc(n2 * k2)?;
    let wsc_p = gpu.alloc(n2 * nb2 * 4)?;
    let ai8_p = gpu.alloc(m2 * k2)?;
    let asc_p = gpu.alloc(m2 * nb2 * 4)?;
    let wblocks = (n2 * nb2) as u32;
    KernelLaunch::new(gpu, hreqw)
        .grid([wblocks.div_ceil(128), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(packed_p)
        .arg_ptr(e4m3_p)
        .arg_f32(scale2)
        .arg_ptr(wi8_p)
        .arg_ptr(wsc_p)
        .arg_u32(n2 as u32)
        .arg_u32(k2 as u32)
        .launch(stream)?;
    let ablocks = (m2 * nb2) as u32;
    KernelLaunch::new(gpu, hreqa)
        .grid([ablocks.div_ceil(128), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(abf_p)
        .arg_ptr(ai8_p)
        .arg_ptr(asc_p)
        .arg_u32(m2 as u32)
        .arg_u32(k2 as u32)
        .launch(stream)?;
    let ce = gpu.alloc(m2 * n2 * 2)?;
    KernelLaunch::new(gpu, hfaith2)
        .grid([(n2 as u32).div_ceil(128), (m2 as u32).div_ceil(128), 1])
        .block([256, 1, 1])
        .arg_ptr(ai8_p)
        .arg_ptr(wi8_p)
        .arg_ptr(asc_p)
        .arg_ptr(wsc_p)
        .arg_ptr(ce)
        .arg_u32(m2 as u32)
        .arg_u32(n2 as u32)
        .arg_u32(k2 as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut re = vec![0u8; m2 * n2 * 2];
    gpu.copy_d2h(ce, &mut re)?;
    let cg: Vec<f32> = re
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
    for i in 0..m2 * n2 {
        let (x, y) = (cref[i] as f64, cg[i] as f64);
        d += x * y;
        nr += x * x;
        ng += y * y;
    }
    let cos = d / (nr.sqrt() * ng.sqrt());
    println!(
        "REQUANT e2e (NVFP4 w->int8 + bf16 a->int8 -> faith2) vs host dequant: cosine={cos:.6}  RESULT: {}",
        if cos > 0.999 { "PASS" } else { "FAIL" }
    );

    for p in [packed_p, e4m3_p, abf_p, wi8_p, wsc_p, ai8_p, asc_p, ce] {
        let _ = gpu.free(p);
    }
    Ok(())
}
