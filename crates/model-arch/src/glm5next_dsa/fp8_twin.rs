// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: Host twin of the FP8 indexer cache's quantization (`dsa_indexer_fp8.cu`): the
//! 128-point rotation, the power-of-two scale, the E4M3 encode and the read-back.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Each function is bit-exact with its device counterpart for the same f32 input: the
//!   kernels build with `--fmad=false` and no fast-math, the rotation's adds and the scale
//!   division are single IEEE operations in the same order, and the E4M3 conversion is a
//!   saturating round-to-nearest-even. `examples/dsa_fp8_indexer_gpu.rs` checks it on a GPU.
//! - [`dequantize_pool`] inverts [`quantize_pool`] up to E4M3 rounding: the rotation is
//!   scaled orthogonal (`H * H = 128 I`), so the key's error has the rotated error's L2 norm
//!   divided by `sqrt(128)`.

use metrale_cache::kv_dequant::e4m3_lut;

/// 2026-10-16: Channels per pool key; the rotation's size.
pub const D: usize = super::fp8_rows::FP8_INDEX_HEAD_DIM;

/// 2026-10-16: The largest finite E4M3 magnitude.
pub const E4M3_MAX: f32 = 448.0;

/// 2026-10-16: `dsa_fwht128`: the unnormalized Walsh-Hadamard transform, in place.
pub fn fwht128(v: &mut [f32; D]) {
    let mut h = 1usize;
    while h < D {
        for i in 0..D / 2 {
            let a = (i / h) * 2 * h + i % h;
            let (x, y) = (v[a], v[a + h]);
            v[a] = x + y;
            v[a + h] = x - y;
        }
        h <<= 1;
    }
}

/// 2026-10-16: `dsa_ue8m0_scale`: `2^k`, `k = ceil(log2(amax / 448))` from the f32 bits of
/// `amax / 448`, clamped to `[-126, 127]`.
pub fn ue8m0_scale(amax: f32) -> f32 {
    let bits = (amax / E4M3_MAX).to_bits();
    let k = ((bits >> 23) & 0xff) as i32 - 127 + i32::from(bits & 0x7f_ffff != 0);
    f32::from_bits(((k.clamp(-126, 127) + 127) as u32) << 23)
}

/// 2026-10-16: `__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3)` for a finite `v`: the
/// nearest finite code (ties to the even code, past 448 saturating to 448), with the sign bit
/// of `v` (so -0.0 is 0x80).
pub fn e4m3_encode(v: f32) -> u8 {
    let lut = e4m3_lut();
    let a = v.abs();
    let mut best = 0u8;
    for c in 1u8..=0x7e {
        let (dc, db) = ((lut[c as usize] - a).abs(), (lut[best as usize] - a).abs());
        if dc < db || (dc == db && c % 2 == 0) {
            best = c;
        }
    }
    if v.is_sign_negative() {
        best | 0x80
    } else {
        best
    }
}

/// 2026-10-16: `dsa_e4m3_to_f32` for a finite code.
pub fn e4m3_decode(b: u8) -> f32 {
    e4m3_lut()[b as usize]
}

/// 2026-10-16: The store's quantization of one pool key: rotate, take the scale from the
/// rotated amax, encode `y / scale`. Returns the entry and its scale.
pub fn quantize_pool(key: &[f32; D]) -> ([u8; D], f32) {
    let mut y = *key;
    fwht128(&mut y);
    let amax = y.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = ue8m0_scale(amax);
    let mut q = [0u8; D];
    for (qi, yi) in q.iter_mut().zip(y) {
        *qi = e4m3_encode(yi / scale);
    }
    (q, scale)
}

/// 2026-10-16: `DsaFp8Pools::key_block`: decode, scale, rotate back, divide by 128.
pub fn dequantize_pool(q: &[u8; D], scale: f32) -> [f32; D] {
    let mut v = [0.0f32; D];
    for (vi, qi) in v.iter_mut().zip(q) {
        *vi = e4m3_decode(*qi) * scale;
    }
    fwht128(&mut v);
    v.map(|x| x * (1.0 / 128.0))
}

#[cfg(test)]
#[path = "fp8_twin_tests.rs"]
mod tests;
