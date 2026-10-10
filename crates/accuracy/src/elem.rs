// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Element formats as rounding models: precision, exponent range, round-to-nearest-even
//! into the format, and the byte codecs the references read operands through.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - Every rounding is RNE on the exact f64 value; no environment switch changes it (a reference
//!   that obeyed a bisect escape hatch would move with the code under test).
//! - Decoding never masks a NaN: a NaN byte decodes to NaN, so a reference fed a corrupt operand
//!   fails loudly instead of reading zero.
//! - `round(x)` of a finite `x` beyond the largest finite value is `None` for a non-saturating
//!   format; the caller states whether the kernel saturates.

use metrale_circuit::pipeline::Num;

/// 2026-10-09: A binary floating-point element format, described by what rounding needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Elem {
    /// 2026-10-09: Spelling, as the family manifest's pipeline writes it where it has one.
    pub name: &'static str,
    /// 2026-10-09: Significand bits, the implicit bit included (bf16: 8, f32: 24, e2m1: 2).
    pub precision: u32,
    /// 2026-10-09: Exponent of the smallest normal value (subnormals share it).
    pub emin: i32,
    /// 2026-10-09: Largest finite value.
    pub max_finite: f64,
    /// 2026-10-09: The format has a sign bit (UE4M3 and UE8M0 scales do not).
    pub signed: bool,
    /// 2026-10-09: Results below the normal range flush to zero (`.ftz`, fast-math), which the
    /// rounding floor must cover.
    pub ftz: bool,
}

/// 2026-10-09: f64, the reference arithmetic.
pub const F64: Elem = Elem {
    name: "f64",
    precision: 53,
    emin: -1022,
    max_finite: f64::MAX,
    signed: true,
    ftz: false,
};
/// 2026-10-09: IEEE binary32.
pub const F32: Elem = Elem {
    name: "f32",
    precision: 24,
    emin: -126,
    max_finite: f32::MAX as f64,
    signed: true,
    ftz: false,
};
/// 2026-10-09: bfloat16.
pub const BF16: Elem = Elem {
    name: "bf16",
    precision: 8,
    emin: -126,
    max_finite: 3.389_531_389_251_535_5e38,
    signed: true,
    ftz: false,
};
/// 2026-10-09: IEEE binary16.
pub const F16: Elem = Elem {
    name: "f16",
    precision: 11,
    emin: -14,
    max_finite: 65504.0,
    signed: true,
    ftz: false,
};
/// 2026-10-09: OCP FP8 E4M3FN (no infinities, max 448).
pub const E4M3: Elem = Elem {
    name: "e4m3",
    precision: 4,
    emin: -6,
    max_finite: 448.0,
    signed: true,
    ftz: false,
};
/// 2026-10-09: OCP FP4 E2M1 (values 0, 0.5, 1, 1.5, 2, 3, 4, 6).
pub const E2M1: Elem = Elem {
    name: "e2m1",
    precision: 2,
    emin: 0,
    max_finite: 6.0,
    signed: true,
    ftz: false,
};
/// 2026-10-09: Unsigned E4M3, the NVFP4 block scale.
pub const UE4M3: Elem = Elem {
    name: "ue4m3",
    precision: 4,
    emin: -6,
    max_finite: 448.0,
    signed: false,
    ftz: false,
};
/// 2026-10-09: UE8M0, a power-of-two scale (MX formats).
pub const UE8M0: Elem = Elem {
    name: "ue8m0",
    precision: 1,
    emin: -127,
    max_finite: 1.701_411_834_604_692_3e38,
    signed: false,
    ftz: false,
};

/// 2026-10-09: IEEE binary32 with denormal results flushed to zero.
pub const F32_FTZ: Elem = Elem { ftz: true, ..F32 };

/// 2026-10-09: The rounding model of a pipeline precision ([`Num`] is the manifest's spelling).
pub fn of_num(n: Num) -> Elem {
    match n {
        Num::Bf16 => BF16,
        Num::F16 => F16,
        Num::F32 => F32,
        Num::E4m3 => E4M3,
        Num::E2m1 => E2M1,
    }
}

impl Elem {
    /// 2026-10-09: Unit roundoff `u = 2^-precision`: RNE moves a normal value by at most `u|x|`.
    pub fn unit_roundoff(&self) -> f64 {
        pow2(-(self.precision as i32))
    }

    /// 2026-10-09: Spacing of the subnormal range, `2^(emin - precision + 1)`: the smallest
    /// positive value (with `ftz`, the smallest a result keeps is `2^emin`).
    pub fn min_subnormal(&self) -> f64 {
        pow2(self.emin - self.precision as i32 + 1)
    }

    /// 2026-10-09: The absolute floor of a rounding: below the normal range RNE moves a value by
    /// at most half the subnormal spacing (with `ftz`, by up to the smallest normal), which the
    /// relative bound `u|x|` does not cover.
    pub fn underflow_floor(&self) -> f64 {
        if self.ftz {
            pow2(self.emin)
        } else {
            self.min_subnormal() * 0.5
        }
    }

    /// 2026-10-09: RNE of `x` into the format. `None` when the result is beyond the largest
    /// finite value (overflow) or `x` is not finite, or `x < 0` for an unsigned format.
    pub fn round(&self, x: f64) -> Option<f64> {
        if !x.is_finite() || (!self.signed && x < 0.0) {
            return None;
        }
        if x == 0.0 {
            return Some(x);
        }
        let e = exponent(x).max(self.emin);
        let quantum = pow2(e - (self.precision as i32 - 1));
        let r = (x / quantum).round_ties_even() * quantum;
        let r = if self.ftz && r.abs() < pow2(self.emin) {
            0.0f64.copysign(x)
        } else {
            r
        };
        (r.abs() <= self.max_finite).then_some(r)
    }

    /// 2026-10-09: RNE with saturation to the largest finite value, as `cvt.satfinite` does.
    pub fn round_saturating(&self, x: f64) -> Option<f64> {
        if !x.is_finite() || (!self.signed && x < 0.0) {
            return None;
        }
        let clamped = x.clamp(-self.max_finite, self.max_finite);
        self.round(clamped)
    }

    /// 2026-10-09: `x` is a value of the format.
    pub fn holds(&self, x: f64) -> bool {
        self.round(x) == Some(x)
    }

    /// 2026-10-09: The closed interval of reals that round (RNE, either tie direction) to the
    /// format value `y`: `y` minus half the spacing below it to `y` plus half the spacing above.
    /// For zero under `ftz`, everything below the smallest normal.
    pub fn preimage(&self, y: f64) -> (f64, f64) {
        if y == 0.0 {
            let h = if self.ftz {
                pow2(self.emin)
            } else {
                self.min_subnormal() * 0.5
            };
            return (-h, h);
        }
        let e = exponent(y).max(self.emin);
        let above = pow2(e - (self.precision as i32 - 1));
        // 2026-10-09: At a power of two (above the subnormal range) the spacing below halves.
        let is_pow2 = y.abs() == pow2(exponent(y));
        let below = if is_pow2 && exponent(y) > self.emin {
            above * 0.5
        } else {
            above
        };
        let (down, up) = if y > 0.0 {
            (below, above)
        } else {
            (above, below)
        };
        (y - down * 0.5, y + up * 0.5)
    }
}

/// 2026-10-09: `2^e` exactly, for the exponents f64 represents as normals or subnormals.
pub fn pow2(e: i32) -> f64 {
    if e >= -1022 {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (e + 1074))
    }
}

/// 2026-10-09: The binary exponent of a finite non-zero f64 (`floor(log2 |x|)`), exactly.
fn exponent(x: f64) -> i32 {
    let bits = x.abs().to_bits();
    let biased = (bits >> 52) as i32;
    if biased == 0 {
        // 2026-10-09: An f64 subnormal: no operand of a supported format is one, but stay exact.
        -1074 + (63 - bits.leading_zeros() as i32)
    } else {
        biased - 1023
    }
}

/// 2026-10-09: Decode a bf16 bit pattern.
pub fn bf16_to_f64(bits: u16) -> f64 {
    f32::from_bits(u32::from(bits) << 16) as f64
}

/// 2026-10-09: Encode a bf16 value. `None` when `x` is not a bf16 value: the caller rounds
/// first, so an unrounded encode is a reference bug, not a silent truncation.
pub fn f64_to_bf16(x: f64) -> Option<u16> {
    BF16.holds(x).then(|| ((x as f32).to_bits() >> 16) as u16)
}

/// 2026-10-09: Decode an IEEE half bit pattern.
pub fn f16_to_f64(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((bits >> 10) & 0x1f);
    let man = f64::from(bits & 0x3ff);
    match exp {
        0 => sign * man * pow2(-24),
        31 if man == 0.0 => sign * f64::INFINITY,
        31 => f64::NAN,
        e => sign * (1.0 + man / 1024.0) * pow2(e - 15),
    }
}

/// 2026-10-09: Decode an FP8 E4M3FN byte; `0x7f`/`0xff` are NaN.
pub fn e4m3_to_f64(b: u8) -> f64 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((b >> 3) & 0x0f);
    let man = f64::from(b & 0x07);
    match (exp, b & 0x7f) {
        (_, 0x7f) => f64::NAN,
        (0, _) => sign * man * pow2(-9),
        (e, _) => sign * (1.0 + man / 8.0) * pow2(e - 7),
    }
}

/// 2026-10-09: Encode an E4M3 value (the caller rounds first).
pub fn f64_to_e4m3(x: f64) -> Option<u8> {
    if !E4M3.holds(x) {
        return None;
    }
    (0u8..=0xff).find(|&b| {
        b & 0x7f != 0x7f && e4m3_to_f64(b) == x && (b & 0x80 != 0) == x.is_sign_negative()
    })
}

/// 2026-10-09: Decode an unsigned E4M3 scale byte (the sign bit must be clear).
pub fn ue4m3_to_f64(b: u8) -> f64 {
    if b & 0x80 != 0 {
        f64::NAN
    } else {
        e4m3_to_f64(b)
    }
}

/// 2026-10-09: Decode an E2M1 nibble (low four bits).
pub fn e2m1_to_f64(nib: u8) -> f64 {
    const MAG: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let m = MAG[usize::from(nib & 0x7)];
    if nib & 0x8 != 0 { -m } else { m }
}

/// 2026-10-09: Encode an E2M1 value as a nibble (the caller rounds first).
pub fn f64_to_e2m1(x: f64) -> Option<u8> {
    if !E2M1.holds(x) {
        return None;
    }
    (0u8..16).find(|&n| e2m1_to_f64(n) == x && ((n & 0x8 != 0) == x.is_sign_negative()))
}

/// 2026-10-09: Decode a UE8M0 scale byte (`0xff` is NaN).
pub fn ue8m0_to_f64(b: u8) -> f64 {
    if b == 0xff {
        f64::NAN
    } else {
        pow2(i32::from(b) - 127)
    }
}

#[cfg(test)]
#[path = "elem_tests.rs"]
mod tests;
