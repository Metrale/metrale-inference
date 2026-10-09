// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The generator stream is pinned (a changed stream silently changes every corpus),
//! keyed streams are independent, and each class produces what its definition says.

use super::*;
use crate::elem::{BF16, E2M1, E4M3};

#[test]
fn splitmix64_stream_is_the_published_one() {
    // 2026-10-09: SplitMix64 from state 0: the reference implementation's first two outputs.
    let mut r = SplitMix64::new(0);
    assert_eq!(r.next_u64(), 0xe220_a839_7b1d_cdaf);
    assert_eq!(r.next_u64(), 0x6e78_9e6a_a1b9_65f4);
}

#[test]
fn keyed_streams_separate_their_parts() {
    let a = SplitMix64::keyed(1, &["ab", "c"]).next_u64();
    let b = SplitMix64::keyed(1, &["a", "bc"]).next_u64();
    let c = SplitMix64::keyed(2, &["ab", "c"]).next_u64();
    let a2 = SplitMix64::keyed(1, &["ab", "c"]).next_u64();
    assert_ne!(a, b);
    assert_ne!(a, c);
    assert_eq!(a, a2);
}

#[test]
fn gaussian_has_unit_moments() {
    let mut r = SplitMix64::new(3);
    let n = 200_000;
    let xs: Vec<f64> = (0..n).map(|_| r.gaussian()).collect();
    let mean = xs.iter().sum::<f64>() / n as f64;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
    assert!(mean.abs() < 0.01, "mean {mean}");
    assert!((var - 1.0).abs() < 0.02, "var {var}");
}

#[test]
fn classes_produce_their_definitions_in_format() {
    let mut r = SplitMix64::new(5);
    let t = tensor(&mut r, InputClass::NearOverflow, 8, 64, 1.0, E4M3);
    assert!(
        t.iter()
            .all(|x| x.abs() >= 0.75 * 448.0 - 32.0 && x.abs() <= 448.0)
    );
    let t = tensor(&mut r, InputClass::Denormal, 8, 64, 1.0, E4M3);
    assert!(
        t.iter()
            .all(|x| *x != 0.0 && x.abs() < crate::elem::pow2(-6))
    );
    let t = tensor(&mut r, InputClass::Denormal, 2, 8, 1.0, E2M1);
    assert!(t.iter().all(|x| x.abs() == 0.5));
    let t = tensor(&mut r, InputClass::ZeroRows, 8, 16, 1.0, BF16);
    for (i, row) in t.chunks(16).enumerate() {
        assert_eq!(row.iter().all(|x| *x == 0.0), i % 4 == 3, "row {i}");
    }
    let t = tensor(&mut r, InputClass::AllEqual, 3, 5, 1.0, BF16);
    assert!(t.iter().all(|x| *x == 0.75));
    let t = tensor(&mut r, InputClass::Outliers, 64, 64, 1.0, BF16);
    assert!(
        t.iter().any(|x| x.abs() > 16.0),
        "an outlier class without outliers"
    );
    for c in InputClass::all() {
        assert_eq!(InputClass::parse(c.name()), Some(c));
        let t = tensor(&mut r, c, 4, 32, 1.0, BF16);
        assert!(
            t.iter().all(|x| BF16.holds(*x)),
            "{} leaves the format",
            c.name()
        );
    }
}

#[test]
fn structural_indices_cover_every_edge() {
    let mut r = SplitMix64::new(9);
    let v = structural_indices(154_856, &[64, 51_619], 32, &mut r);
    for must in [0, 63, 64, 51_618, 51_619, 103_237, 103_238, 154_855] {
        assert!(v.binary_search(&must).is_ok(), "missing {must}");
    }
    assert!(v.windows(2).all(|w| w[0] < w[1]));
}
