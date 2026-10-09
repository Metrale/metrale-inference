// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The comparison measures the right ratio, treats a NaN kernel value as an
//! infinite ratio, and refuses every vacuous comparison.

use super::*;
use crate::bounded::Bounded;
use crate::elem::{BF16, F64};

fn b(v: f64, e: f64) -> Bounded {
    Bounded { v, e }
}

#[test]
fn ratios_and_refusals() {
    let r = bounded(
        &[1.0, 2.5, 3.0],
        &[b(1.0, 0.1), b(2.0, 1.0), b(3.0, 0.0)],
        F64,
    )
    .unwrap();
    assert!((r.max_ratio - 0.5).abs() < 1e-12);
    assert_eq!(r.worst, 1);
    assert!((r.max_err - 0.5).abs() < 1e-12);
    let nan = bounded(&[f64::NAN], &[b(1.0, 0.1)], F64).unwrap();
    assert!(nan.max_ratio.is_infinite());
    let exact_miss = bounded(&[1.0 + 1e-12], &[b(1.0, 0.0)], F64).unwrap();
    assert!(
        exact_miss.max_ratio.is_infinite(),
        "a zero bound admits no error"
    );
    assert_eq!(bounded(&[], &[], F64), Err(Vacuous::Empty));
    assert_eq!(
        bounded(&[1.0], &[b(1.0, 0.1), b(1.0, 0.1)], F64),
        Err(Vacuous::Length { got: 1, want: 2 })
    );
    assert_eq!(
        bounded(&[1.0], &[b(1.0, f64::INFINITY)], F64),
        Err(Vacuous::Unbounded(1))
    );
    assert_eq!(
        bounded(&[0.0, 0.0], &[b(0.0, 1.0), b(0.0, 1.0)], F64),
        Err(Vacuous::Trivial)
    );
}

#[test]
fn the_output_rounding_never_counts_against_the_kernel() {
    // 2026-10-09: v = 1 + 1/512 rounds to 1.0 or 1 + 1/128 in bf16 (spacing 1/128): either side
    // of the tie is a correct rounding, so both score zero even with a zero pre-rounding bound.
    let v = 1.0 + 1.0 / 256.0;
    assert_eq!(bounded(&[1.0], &[b(v, 0.0)], BF16).unwrap().max_ratio, 0.0);
    assert_eq!(
        bounded(&[1.0 + 1.0 / 128.0], &[b(v, 0.0)], BF16)
            .unwrap()
            .max_ratio,
        0.0
    );
    // 2026-10-09: One ulp further is not a rounding of v: its excess is half an ulp.
    let r = bounded(&[1.0 + 2.0 / 128.0], &[b(v, 1.0 / 512.0)], BF16).unwrap();
    assert_eq!(r.max_err, 1.0 / 128.0);
    assert_eq!(r.max_ratio, 4.0);
    // 2026-10-09: Below a power of two the spacing halves.
    assert_eq!(BF16.preimage(1.0), (1.0 - 1.0 / 512.0, 1.0 + 1.0 / 256.0));
    assert_eq!(
        BF16.preimage(-1.0),
        (-1.0 - 1.0 / 256.0, -1.0 + 1.0 / 512.0)
    );
}

#[test]
fn byte_comparison() {
    let d = bytes(&[1, 2, 3], &[1, 9, 3]).unwrap();
    assert_eq!((d.differing, d.first), (1, Some(1)));
    assert_eq!(bytes(&[1, 2], &[1, 2]).unwrap().differing, 0);
    assert_eq!(bytes(&[], &[]), Err(Vacuous::Empty));
    assert_eq!(bytes(&[0, 0], &[0, 0]), Err(Vacuous::Trivial));
    assert_eq!(
        bytes(&[1], &[1, 2]),
        Err(Vacuous::Length { got: 1, want: 2 })
    );
}
