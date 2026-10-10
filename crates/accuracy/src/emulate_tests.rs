// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The emulation's bracketing respects the depth it is given (so it is a
//! conforming kernel) and a narrower accumulator really is narrower.

use super::*;
use crate::bounded::{Bounded, sum};
use crate::elem::{BF16, F32};
use crate::inputs::SplitMix64;

#[test]
fn reduce_stays_inside_the_bound_of_its_depth() {
    let mut r = SplitMix64::new(21);
    for n in [1usize, 2, 3, 17, 320, 5120] {
        for depth in [n.ilog2() as u64 + 1, 29, n as u64] {
            let xs: Vec<f64> = (0..n).map(|_| F32.round(r.gaussian()).unwrap()).collect();
            let b = sum(
                &xs.iter().map(|&x| Bounded::exact(x)).collect::<Vec<_>>(),
                F32,
                depth,
            );
            for v in 0..VARIANTS {
                let got = reduce(&xs, F32, depth, v);
                assert!((got - b.v).abs() <= b.e, "n={n} depth={depth} variant={v}");
            }
        }
    }
}

#[test]
fn a_bf16_accumulator_loses_precision() {
    let mut r = SplitMix64::new(22);
    let xs: Vec<f64> = (0..4096)
        .map(|_| BF16.round(r.gaussian()).unwrap())
        .collect();
    let exact: f64 = xs.iter().sum();
    let f = (reduce(&xs, F32, 29, 0) - exact).abs();
    let b = (reduce(&xs, BF16, 29, 0) - exact).abs();
    assert!(b > 100.0 * f.max(1e-30), "bf16 err {b:e} vs f32 err {f:e}");
}
