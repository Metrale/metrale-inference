// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The streams are pinned bit for bit (a golden: any change to the generator changes
//! every mock), independent per tensor, and have the moments synthesis assumes.
//!
//! Owner: metrale-ml-utils.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn splitmix_matches_its_published_first_output() {
    // 2026-10-03: SplitMix64 seeded with 0 first returns 0xE220A8397B1DCDAF.
    assert_eq!(Stream { key: 0 }.bits(0), 0xE220_A839_7B1D_CDAF);
}

#[test]
fn a_tensor_stream_is_pinned() {
    let s = Stream::for_tensor(42, "model.layers.0.mlp.down_proj.weight", "BF16", &[8, 16]);
    let got: Vec<u32> = (0..4).map(|i| s.normal4(i).to_bits()).collect();
    let again: Vec<u32> = (0..4).map(|i| s.normal4(i).to_bits()).collect();
    assert_eq!(got, again);
    assert_eq!(
        GOLDEN_NORMAL4,
        got.as_slice(),
        "the generator changed: every mock changes"
    );
}

/// 2026-10-03: Recorded from this implementation on 2026-10-03.
const GOLDEN_NORMAL4: &[u32] = &[3_214_871_821, 1_073_701_373, 3_221_368_641, 1_074_184_624];

#[test]
fn streams_differ_by_seed_name_dtype_and_shape() {
    let b = |s: Stream| s.bits(0);
    let base = Stream::for_tensor(1, "a", "BF16", &[2]);
    assert_ne!(b(base), b(Stream::for_tensor(2, "a", "BF16", &[2])));
    assert_ne!(b(base), b(Stream::for_tensor(1, "b", "BF16", &[2])));
    assert_ne!(b(base), b(Stream::for_tensor(1, "a", "F32", &[2])));
    assert_ne!(b(base), b(Stream::for_tensor(1, "a", "BF16", &[3])));
    assert_ne!(b(base), b(base.derive(1)));
}

#[test]
fn normal_draws_have_unit_variance_and_zero_mean() {
    let s = Stream::for_tensor(9, "x", "BF16", &[1]);
    for (name, f) in [
        (
            "normal4",
            (|s: Stream, i| s.normal4(i)) as fn(Stream, u64) -> f32,
        ),
        ("normal12", |s: Stream, i| s.normal12(i)),
    ] {
        let n = 200_000u64;
        let (mut sum, mut sq) = (0.0f64, 0.0f64);
        for i in 0..n {
            let v = f(s, i) as f64;
            sum += v;
            sq += v * v;
        }
        let mean = sum / n as f64;
        let var = sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.01, "{name} mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "{name} variance {var}");
    }
    assert!((0..10_000).all(|i| s.normal4(i).abs() <= 3.4642));
}

#[test]
fn below_stays_in_range() {
    let s = Stream::for_tensor(3, "ids", "I32", &[1]);
    assert!((0..10_000).all(|i| s.below(i, 7) < 7));
}
