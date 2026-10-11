// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The rounding models against exhaustive decodes: every byte of every 8-bit format
//! and every nibble of E2M1 must be a fixed point of `round`, and rounding must land on the
//! nearest representable value with ties to even.

use super::*;

fn all_e4m3() -> Vec<f64> {
    (0u8..=0xff)
        .map(e4m3_to_f64)
        .filter(|v| v.is_finite())
        .collect()
}

#[test]
fn every_e4m3_byte_is_a_fixed_point_and_round_trips() {
    for b in 0u8..=0xff {
        let v = e4m3_to_f64(b);
        if b & 0x7f == 0x7f {
            assert!(v.is_nan(), "byte {b:#x} must decode to NaN");
            continue;
        }
        assert_eq!(E4M3.round(v), Some(v), "byte {b:#x} = {v}");
        assert_eq!(f64_to_e4m3(v), Some(b), "byte {b:#x} = {v}");
    }
    assert_eq!(e4m3_to_f64(0x7e), 448.0);
    assert_eq!(e4m3_to_f64(0x01), pow2(-9));
}

#[test]
fn e2m1_table_is_the_ocp_one() {
    let mags: Vec<f64> = (0u8..8).map(e2m1_to_f64).collect();
    assert_eq!(mags, vec![0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]);
    for n in 0u8..16 {
        let v = e2m1_to_f64(n);
        assert_eq!(E2M1.round(v), Some(v));
        if v != 0.0 {
            assert_eq!(f64_to_e2m1(v), Some(n));
        }
    }
}

#[test]
fn rounding_is_nearest_with_ties_to_even() {
    let vals = all_e4m3();
    let mut sorted: Vec<f64> = vals.iter().copied().filter(|v| *v >= 0.0).collect();
    sorted.sort_by(f64::total_cmp);
    sorted.dedup();
    for w in sorted.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        let mid = (lo + hi) / 2.0;
        let even = if f64_to_e4m3(lo).unwrap() & 1 == 0 {
            lo
        } else {
            hi
        };
        assert_eq!(E4M3.round(mid), Some(even), "tie between {lo} and {hi}");
        assert_eq!(E4M3.round(lo + (hi - lo) * 0.25), Some(lo));
        assert_eq!(E4M3.round(lo + (hi - lo) * 0.75), Some(hi));
    }
    // 2026-10-09: E2M1 ties: 2.5 lies between 2 (even significand) and 3; 5 between 4 and 6.
    assert_eq!(E2M1.round(2.5), Some(2.0));
    assert_eq!(E2M1.round(5.0), Some(4.0));
    assert_eq!(E2M1.round(0.25), Some(0.0));
    assert_eq!(E2M1.round(0.75), Some(1.0));
}

#[test]
fn overflow_is_refused_and_saturation_is_explicit() {
    assert_eq!(
        E4M3.round(464.0),
        Some(448.0),
        "a tie between 448 and 480 goes to the even 448"
    );
    assert_eq!(
        E4M3.round(480.0),
        None,
        "480 is on the grid but beyond the largest finite 448"
    );
    assert_eq!(E4M3.round_saturating(1e9), Some(448.0));
    assert_eq!(E2M1.round(7.0), None);
    assert_eq!(E2M1.round_saturating(-100.0), Some(-6.0));
    assert_eq!(
        UE4M3.round(-1.0),
        None,
        "an unsigned scale has no negative values"
    );
}

#[test]
fn bf16_matches_the_hardware_conversion_on_f32_inputs() {
    // 2026-10-09: f32 -> bf16 RNE by bit arithmetic (the device's cvt.rn.bf16.f32) on a sweep of
    // patterns, including ties and subnormals.
    let mut s = 0x1234_5678_u32;
    for _ in 0..200_000 {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        let x = f32::from_bits(s);
        if !x.is_finite() {
            continue;
        }
        let bits = x.to_bits();
        let rne = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16;
        let want = bf16_to_f64(rne);
        if want.is_infinite() {
            assert_eq!(BF16.round(f64::from(x)), None);
        } else {
            assert_eq!(
                BF16.round(f64::from(x)),
                Some(want),
                "x = {x:e} ({bits:#x})"
            );
        }
    }
}

#[test]
fn f16_decode_spans_subnormals_and_specials() {
    assert_eq!(f16_to_f64(0x3c00), 1.0);
    assert_eq!(f16_to_f64(0x0001), pow2(-24));
    assert_eq!(f16_to_f64(0x7bff), 65504.0);
    assert!(f16_to_f64(0x7c00).is_infinite());
    assert!(f16_to_f64(0x7e00).is_nan());
    assert_eq!(F16.min_subnormal(), pow2(-24));
}

#[test]
fn unit_roundoff_bounds_every_normal_rounding() {
    for (fmt, x) in [
        (BF16, 1.0 + 1.0 / 512.0),
        (F32, 1.0 + pow2(-25)),
        (E4M3, 1.0625),
    ] {
        let r = fmt.round(x).unwrap();
        assert!(
            (r - x).abs() <= fmt.unit_roundoff() * x.abs(),
            "{}",
            fmt.name
        );
    }
    assert_eq!(E2M1.unit_roundoff(), 0.25);
    assert_eq!(
        UE8M0.round(3.0),
        Some(4.0),
        "UE8M0 holds powers of two only; 3 ties to 4"
    );
    assert!(ue8m0_to_f64(0xff).is_nan());
    assert!(
        ue4m3_to_f64(0x80).is_nan(),
        "a set sign bit is no unsigned scale"
    );
}
