// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `f32` to packed NVFP4 on the host, the inverse of [`super::nvfp4_dequant`].
//!
//! 2026-10-03: Moved here from the GLM-5.3 loader, which re-exports it; `metrale-ml-utils`
//! quantizes synthetic weights with it.
//!
//! Owner: metrale-core.
//!
//! The GLM-5.3 loader's `bind_expert` binds a routed-expert projection stored as packed U8
//! directly; one stored as BF16 is quantised here, once, at load.
//!
//! 2026-10-04: The rounding is the load-time GPU quantizer's, bit for bit
//! (`kernels/gb10/common/quantize_bf16_to_nvfp4.cu`, compiled with `--fmad=false`), so a
//! checkpoint quantized on the host holds the bytes the engine's NVFP4 tier builds at load. The
//! GPU quantizer is the reference because certified serves run it.
//!
//! ```text
//! scale_2     = amax(tensor) / (6 * 448)                 per tensor, f32 (1.0 when amax is 0)
//! scale byte  = e4m3_gpu(amax(block16) * (1 / scale_2) / 6)
//! code        = e2m1_gpu( w * (1 / (e4m3(scale byte) * scale_2)) )
//! ```
//!
//! Invariants:
//! - `e4m3_gpu` is the kernel's `float_to_fp8_e4m3`: saturates at 448, flushes magnitudes below
//!   2^-9 to zero, rounds a subnormal as `floor(v * 512 + 0.5)` capped at code 7, and rounds a
//!   normal mantissa half away from zero.
//! - `e2m1_gpu` is the kernel's `quantize_e2m1`: thresholds 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5,
//!   each inclusive, so every tie goes to the smaller magnitude; the sign bit is set for
//!   `v < 0` only (a negative zero is code 0).
//! - Codes are chosen against the decoded block scale, not the requested one.
//! - Even flat index is the low nibble, as in the dequantiser.
//! - `model-engine/tests/nvfp4_host_gpu_kat.rs` checks these against the GPU on a GB10.

use super::{FP8_E4M3_LUT, NVFP4_E2M1_LUT, NVFP4_GROUP_SIZE};
use anyhow::{Result, bail};

/// 2026-09-25: `E4M3` codes `0x00..=0x7E`: every finite non-negative value,
/// ascending. `0x7F` is NaN and `0x80..` are the negatives, neither of which a
/// block scale may be.
const E4M3_FINITE_CODES: usize = 0x7F;

/// 2026-09-25: The largest `E2M1` magnitude (6.0), the top of the codebook's
/// non-negative half.
fn e2m1_max() -> f32 {
    NVFP4_E2M1_LUT[7]
}

/// 2026-09-25: The largest finite `E4M3` value (448.0).
fn e4m3_max() -> f32 {
    FP8_E4M3_LUT[E4M3_FINITE_CODES - 1]
}

/// 2026-09-25: One quantised projection, in the three pieces `Nvfp4Proj` is built from.
#[derive(Debug)]
pub struct Nvfp4Blob {
    /// 2026-09-25: `[rows, cols / 2]` U8, two `e2m1` codes per byte.
    pub packed: Vec<u8>,
    /// 2026-09-25: `[rows, cols / 16]` `E4M3` block scales, one byte each.
    pub scales: Vec<u8>,
    /// 2026-09-25: The per-tensor global scale.
    pub scale_2: f32,
}

/// 2026-09-25: Quantise a row-major `f32 [rows, cols]` weight to NVFP4.
///
/// Errors on a length that is not `rows * cols`, on `cols` that is zero or not a
/// multiple of 16, and on any non-finite value.
pub fn quantize_to_nvfp4(
    what: &str,
    values: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Nvfp4Blob> {
    if values.len() != rows * cols {
        bail!(
            "{what}: {} elements, expected [{rows}, {cols}] = {}",
            values.len(),
            rows * cols
        );
    }
    // 2026-09-25: `cols == 0` passes `is_multiple_of` and would make the band
    // width zero, which `chunks` panics on.
    if cols == 0 || !cols.is_multiple_of(NVFP4_GROUP_SIZE) {
        bail!(
            "{what}: {cols} columns is not a whole number of {NVFP4_GROUP_SIZE}-element \
             NVFP4 blocks"
        );
    }
    // 2026-09-25: `f32::max` ignores NaN (it returns the other operand), so
    // testing the folded amax would miss every NaN; the scan rejects per element.
    let mut amax = 0.0f32;
    for &v in values {
        if !v.is_finite() {
            bail!("{what}: weight contains a non-finite value; refusing to quantise it");
        }
        amax = amax.max(v.abs());
    }
    // 2026-09-25: An all-zero tensor has no amax to scale by; with `1.0` every
    // block scale and every code is zero.
    let scale_2 = if amax > 0.0 {
        amax / (e2m1_max() * e4m3_max())
    } else {
        1.0
    };

    let groups_per_row = cols / NVFP4_GROUP_SIZE;
    let mut packed = vec![0u8; rows * cols / 2];
    let mut scales = vec![0u8; rows * groups_per_row];

    // 2026-09-25: Rows are independent once `scale_2` is known (a block never
    // reaches past its own 16 columns), so the sweep runs one scoped thread per
    // row band, over disjoint `chunks_mut` slices.
    let bands = std::thread::available_parallelism().map_or(1, |n| n.get());
    let band_rows = rows.div_ceil(bands).max(1);
    std::thread::scope(|s| {
        for ((v, p), sc) in values
            .chunks(band_rows * cols)
            .zip(packed.chunks_mut(band_rows * cols / 2))
            .zip(scales.chunks_mut(band_rows * groups_per_row))
        {
            s.spawn(move || quantize_rows(v, p, sc, cols, scale_2));
        }
    });
    Ok(Nvfp4Blob {
        packed,
        scales,
        scale_2,
    })
}

/// 2026-09-25: One band of whole rows. `values`, `packed` and `scales` are the
/// band's slices of the three buffers, so every index here is band-local.
/// `packed` must be zeroed: codes are OR-ed in.
fn quantize_rows(values: &[f32], packed: &mut [u8], scales: &mut [u8], cols: usize, scale_2: f32) {
    let groups_per_row = cols / NVFP4_GROUP_SIZE;
    let e4m3 = &FP8_E4M3_LUT;
    let inv_scale_2 = if scale_2 > 0.0 { 1.0 / scale_2 } else { 0.0 };
    for (r, row) in values.chunks(cols).enumerate() {
        for g in 0..groups_per_row {
            let base = g * NVFP4_GROUP_SIZE;
            let block = &row[base..base + NVFP4_GROUP_SIZE];
            let bmax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let requested = if bmax > 0.0 {
                bmax * inv_scale_2 / e2m1_max()
            } else {
                0.0
            };
            let sb = e4m3_scale_gpu(requested);
            scales[r * groups_per_row + g] = sb;
            // 2026-09-25: The decoded scale, not the requested one: the codes
            // are chosen against what the stored byte means.
            let eff = e4m3[sb as usize] * scale_2;
            let inv = if eff > 0.0 { 1.0 / eff } else { 0.0 };
            for (i, &v) in block.iter().enumerate() {
                let code = e2m1_gpu(v * inv);
                let flat = r * cols + base + i;
                if flat.is_multiple_of(2) {
                    packed[flat / 2] |= code;
                } else {
                    packed[flat / 2] |= code << 4;
                }
            }
        }
    }
}

/// 2026-10-04: The GPU's `quantize_e2m1`: inclusive thresholds, so a tie takes the smaller
/// magnitude; the sign bit only for `v < 0`.
fn e2m1_gpu(v: f32) -> u8 {
    let a = v.abs();
    let idx = if a <= 0.25 {
        0
    } else if a <= 0.75 {
        1
    } else if a <= 1.25 {
        2
    } else if a <= 1.75 {
        3
    } else if a <= 2.5 {
        4
    } else if a <= 3.5 {
        5
    } else if a <= 5.0 {
        6
    } else {
        7
    };
    if v < 0.0 { 0x8 | idx } else { idx }
}

/// 2026-10-04: The GPU's `float_to_fp8_e4m3` for a block scale (`v >= 0`): saturate at 448,
/// flush below 2^-9, subnormals `floor(v * 512 + 0.5)` capped at 7, normal mantissas rounded half
/// away from zero. Never the NaN code.
pub fn e4m3_scale_gpu(v: f32) -> u8 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 1) as u8;
    if bits & 0x7FFF_FFFF == 0 {
        return sign << 7;
    }
    let a = v.abs().min(448.0);
    let b = a.to_bits();
    let exp = ((b >> 23) & 0xFF) as i32 - 127;
    let man = b & 0x7F_FFFF;
    if exp < -9 {
        return sign << 7;
    }
    if exp < -6 {
        let m = ((a * 512.0 + 0.5) as i32).clamp(0, 7) as u8;
        return (sign << 7) | m;
    }
    let mut e = (exp + 7).max(1);
    let mut m = (man + (1 << 19)) >> 20;
    if e > 15 {
        e = 15;
        m = 6;
    } else if m > 7 {
        m = 0;
        e += 1;
        if e > 15 {
            e = 15;
            m = 6;
        }
    }
    (sign << 7) | ((e as u8) << 3) | m as u8
}

#[cfg(test)]
#[path = "nvfp4_quant_tests.rs"]
mod tests;
