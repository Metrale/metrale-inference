// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The software numerics against independent references: std's IEEE `sqrt`, an
//! exhaustive fp16 table, and nearest-neighbour checks for bf16.

mod common;

use common::Rng;
use metrale_kv_tq128::numeric::{
    bf16_bits_to_f32, f16_bits_to_f32, f32_to_bf16_bits, f32_to_f16_bits, group_sum_squares,
    sqrt_f32, warp_xor_sum,
};

#[test]
fn sqrt_is_bit_identical_to_ieee_sqrt() {
    let mut rng = Rng::new(1);
    let edges = [
        0.0_f32,
        -0.0,
        f32::MIN_POSITIVE,
        1e-45,
        1.0,
        2.0,
        4.0,
        f32::MAX,
        f32::INFINITY,
    ];
    for x in edges {
        assert_eq!(sqrt_f32(x).to_bits(), x.sqrt().to_bits(), "x = {x:e}");
    }
    for _ in 0..2_000_000 {
        let x = f32::from_bits((rng.next_u64() as u32) & 0x7FFF_FFFF);
        if x.is_finite() {
            assert_eq!(sqrt_f32(x).to_bits(), x.sqrt().to_bits(), "x = {x:e}");
        }
    }
    // Every f32 in [1, 4) whose root is exactly half-way-adjacent: the full mantissa range of
    // two binades, both exponent parities.
    for m in 0..(1u32 << 23) {
        for e in [127_u32, 128] {
            let x = f32::from_bits((e << 23) | m);
            assert_eq!(sqrt_f32(x).to_bits(), x.sqrt().to_bits(), "x = {x:e}");
        }
    }
    assert!(sqrt_f32(-1.0).is_nan());
    assert!(sqrt_f32(f32::NAN).is_nan());
}

fn f16_value(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((h >> 10) & 0x1F);
    let m = f64::from(h & 0x3FF);
    match e {
        0 => sign * m * 2f64.powi(-24),
        31 if m == 0.0 => sign * f64::INFINITY,
        31 => f64::NAN,
        _ => sign * (1.0 + m / 1024.0) * 2f64.powi(e - 15),
    }
}

#[test]
fn f16_decode_matches_the_format_definition_for_all_65536_codes() {
    for h in 0..=u16::MAX {
        let (got, want) = (f64::from(f16_bits_to_f32(h)), f16_value(h));
        if want.is_nan() {
            assert!(got.is_nan(), "{h:#06x}");
        } else {
            assert_eq!(got.to_bits(), want.to_bits(), "{h:#06x}");
        }
    }
}

/// The nearest fp16 to `x` by search over the finite positive codes, ties to the even code.
fn f16_nearest(x: f32) -> u16 {
    let sign: u16 = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = f64::from(x.abs());
    if a >= 65520.0 {
        return sign | 0x7C00;
    }
    let codes: Vec<u16> = (0..0x7C00).collect();
    let i = codes.partition_point(|&c| f16_value(c) <= a);
    let lo = codes[i.saturating_sub(1)];
    let Some(&hi) = codes.get(i) else {
        return sign | lo;
    };
    let (dl, dh) = (a - f16_value(lo), f16_value(hi) - a);
    let pick = if dl < dh || (dl == dh && lo & 1 == 0) {
        lo
    } else {
        hi
    };
    sign | pick
}

#[test]
fn f16_encode_rounds_to_nearest_even() {
    for h in 0..0x7C00_u16 {
        for s in [0, 0x8000] {
            assert_eq!(
                f32_to_f16_bits(f16_bits_to_f32(h | s)),
                h | s,
                "round trip {h:#06x}"
            );
        }
        if h == 0x7BFF {
            continue; // the next code is infinity; the overflow edge is asserted below
        }
        // The exact midpoint to the next code is representable in f32 and must go to even.
        let (a, b) = (f16_value(h), f16_value(h + 1));
        let mid = ((a + b) / 2.0) as f32;
        let even = if h & 1 == 0 { h } else { h + 1 };
        assert_eq!(f32_to_f16_bits(mid), even, "midpoint above {h:#06x}");
        assert_eq!(
            f32_to_f16_bits(mid.next_down()),
            h,
            "below midpoint {h:#06x}"
        );
        assert_eq!(
            f32_to_f16_bits(mid.next_up()),
            h + 1,
            "above midpoint {h:#06x}"
        );
    }
    let mut rng = Rng::new(2);
    for _ in 0..20_000 {
        let x = f32::from_bits((rng.next_u64() as u32 & 0x87FF_FFFF) | 0x3000_0000);
        assert_eq!(f32_to_f16_bits(x), f16_nearest(x), "x = {x:e}");
    }
    assert_eq!(f32_to_f16_bits(65504.0), 0x7BFF);
    assert_eq!(f32_to_f16_bits(65519.996), 0x7BFF);
    assert_eq!(f32_to_f16_bits(65520.0), 0x7C00);
    assert_eq!(f32_to_f16_bits(-1e9), 0xFC00);
    assert_eq!(
        f32_to_f16_bits(2.9802322e-8),
        0x0000,
        "2^-25 ties to even zero"
    );
    assert_eq!(f32_to_f16_bits(2.9802326e-8), 0x0001);
    assert_eq!(f32_to_f16_bits(f32::NAN) & 0x7C00, 0x7C00);
}

#[test]
fn bf16_encode_picks_the_nearest_neighbour_ties_to_even() {
    let mut rng = Rng::new(3);
    for _ in 0..200_000 {
        let x = f32::from_bits(rng.next_u64() as u32);
        if !x.is_finite() || x.abs() > 3.3e38 {
            continue;
        }
        let r = f32_to_bf16_bits(x);
        let lo_bits = x.to_bits() & 0xFFFF_0000;
        let (lo, hi) = (f32::from_bits(lo_bits), f32::from_bits(lo_bits + 0x1_0000));
        let (dl, dh) = (
            (f64::from(x) - f64::from(lo)).abs(),
            (f64::from(hi) - f64::from(x)).abs(),
        );
        let lo16 = (lo_bits >> 16) as u16;
        let want = if dl < dh || (dl == dh && lo16 & 1 == 0) {
            lo16
        } else {
            lo16 + 1
        };
        assert_eq!(r, want, "x = {x:e}");
        assert_eq!(bf16_bits_to_f32(r).to_bits(), u32::from(r) << 16);
    }
    assert!(bf16_bits_to_f32(f32_to_bf16_bits(f32::NAN)).is_nan());
}

#[test]
fn the_warp_sum_is_the_xor_butterfly_tree() {
    // Integers sum exactly in any order.
    let lanes: [f32; 32] = core::array::from_fn(|i| i as f32);
    assert_eq!(warp_xor_sum(lanes), 496.0);
    // 1e8 + 1 - 1e8 is order dependent: the butterfly pairs lane l with l ^ 16 first, so
    // (1e8 + -1e8) cancels before the 1 is added.
    let mut l = [0.0_f32; 32];
    l[0] = 1e8;
    l[16] = -1e8;
    l[1] = 1.0;
    assert_eq!(warp_xor_sum(l), 1.0);
    let mut s = [0.0_f32; 32];
    s[0] = 1e8;
    s[1] = -1e8;
    s[2] = 1.0;
    // Pairs (0,16)=1e8, (1,17)=-1e8, (2,18)=1 -> off 8,4: unchanged -> off 2: lane0 = 1e8 + 1
    // rounds to 1e8 -> off 1: 1e8 + -1e8 = 0.
    assert_eq!(warp_xor_sum(s), 0.0);
    let x: [f32; 128] = core::array::from_fn(|i| (i % 7) as f32);
    let exact: f32 = x.iter().map(|v| v * v).sum();
    assert_eq!(group_sum_squares(&x), exact);
}
