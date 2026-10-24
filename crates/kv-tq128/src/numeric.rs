// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The scalar arithmetic the kernels rely on, in software so the crate stays
//! `no_std`: an IEEE correctly rounded `sqrt` (what `sqrtf` compiles to without fast-math), fp16
//! and bf16 conversions rounding to nearest-even (`__float2half`, `__float2bfloat16_rn`), and
//! the 32-lane xor-butterfly sum the kernels reduce a group with.
//!
//! Owner: kv-tq128.
//! Invariants: every function is exact or correctly rounded; none depends on the host FPU mode.

/// 2026-10-18: The low 16 bits of `v`, without a cast.
fn low_u16(v: u32) -> u16 {
    let [a, b, _, _] = v.to_le_bytes();
    u16::from_le_bytes([a, b])
}

/// 2026-10-18: `sqrt(x)` correctly rounded to nearest-even, as IEEE 754 and `sqrtf` define it.
pub fn sqrt_f32(x: f32) -> f32 {
    if x.is_nan() || x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 || x.is_infinite() {
        return x;
    }
    let bits = x.to_bits();
    let biased = (bits >> 23) & 0xFF;
    let frac = bits & 0x7F_FFFF;
    // x = m * 2^e with m a 24-bit integer (normalized) and e unbiased.
    let (mut m, mut e) = if biased == 0 {
        let shift = frac.leading_zeros() - 8;
        (frac << shift, -149_i32 - i32::try_from(shift).unwrap_or(0))
    } else {
        (frac | 0x80_0000, i32::try_from(biased).unwrap_or(0) - 150)
    };
    // m' in [2^24, 2^26) with e even, so N = m' << 24 has a 25-bit root.
    if e & 1 != 0 {
        m <<= 1;
        e -= 1;
    } else {
        m <<= 2;
        e -= 2;
    }
    let n = u64::from(m) << 24;
    let q = n.isqrt();
    let sticky = n != q * q;
    let mut mant = q >> 1;
    if q & 1 == 1 && (sticky || mant & 1 == 1) {
        mant += 1;
    }
    // value = mant * 2^p, p = (e - 24) / 2 + 1, with mant in [2^23, 2^24].
    let mut p = (e - 24) / 2 + 1;
    if mant == 1 << 24 {
        mant >>= 1;
        p += 1;
    }
    let exp = u32::try_from(p + 23 + 127).unwrap_or(0);
    let mant32 = u32::try_from(mant & 0x7F_FFFF).unwrap_or(0);
    f32::from_bits((exp << 23) | mant32)
}

/// 2026-10-18: f32 to IEEE binary16 bits, round to nearest-even; overflow gives infinity and
/// NaN a quiet NaN (`__float2half` agrees on every finite input).
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = low_u16((b >> 16) & 0x8000);
    let abs = b & 0x7FFF_FFFF;
    if abs > 0x7F80_0000 {
        return sign | 0x7E00;
    }
    // 65520 = the midpoint between 65504 and 2^16; it and everything above round to infinity.
    if abs >= 0x477F_F000 {
        return sign | 0x7C00;
    }
    if abs < 0x3880_0000 {
        // Below 2^-14: an fp16 subnormal or zero, in units of 2^-24.
        if abs <= 0x3300_0000 {
            return sign;
        }
        let m = (abs & 0x7F_FFFF) | 0x80_0000;
        let shift = 126 - (abs >> 23);
        let mut q = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        if rem > half || (rem == half && q & 1 == 1) {
            q += 1;
        }
        return sign | low_u16(q);
    }
    let e = (abs >> 23) - 112;
    let m = abs & 0x7F_FFFF;
    let mut q = (e << 10) | (m >> 13);
    let rem = m & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && q & 1 == 1) {
        q += 1;
    }
    sign | low_u16(q)
}

/// 2026-10-18: IEEE binary16 bits to f32 (exact).
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let h = u32::from(h);
    let sign = (h & 0x8000) << 16;
    let e = (h >> 10) & 0x1F;
    let m = h & 0x3FF;
    let bits = match (e, m) {
        (0, 0) => sign,
        (0, _) => {
            let shift = m.leading_zeros() - 21;
            let mant = (m << shift) & 0x3FF;
            sign | ((113 - shift) << 23) | (mant << 13)
        }
        (31, 0) => sign | 0x7F80_0000,
        (31, _) => sign | 0x7FC0_0000 | (m << 13),
        _ => sign | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// 2026-10-18: f32 to bf16 bits, round to nearest-even (`__float2bfloat16_rn` on finite input);
/// NaN stays a quiet NaN.
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return low_u16((b >> 16) | 0x40);
    }
    let rounded = b + 0x7FFF + ((b >> 16) & 1);
    low_u16(rounded >> 16)
}

/// 2026-10-18: bf16 bits to f32 (exact).
pub fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

/// 2026-10-18: The sum a warp computes from one partial per lane with
/// `v += __shfl_xor_sync(.., v, off)` for `off` = 16, 8, 4, 2, 1: every lane holds the same
/// result, returned here.
pub fn warp_xor_sum(mut lanes: [f32; 32]) -> f32 {
    for off in [16_usize, 8, 4, 2, 1] {
        let prev = lanes;
        for (l, v) in lanes.iter_mut().enumerate() {
            *v = prev[l] + prev[l ^ off];
        }
    }
    lanes[0]
}

/// 2026-10-18: The sum of squares of a 128-group as the kernels form it: lane `l` holds
/// elements `4l..4l+4` and adds their squares in order, then the warp reduces with
/// [`warp_xor_sum`].
pub fn group_sum_squares(x: &[f32; 128]) -> f32 {
    let mut lanes = [0.0_f32; 32];
    for (l, v) in lanes.iter_mut().enumerate() {
        let s = &x[4 * l..4 * l + 4];
        let (a, b, c, d) = (s[0] * s[0], s[1] * s[1], s[2] * s[2], s[3] * s[3]);
        *v = ((a + b) + c) + d;
    }
    warp_xor_sum(lanes)
}
