// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Soundness and sensitivity of the bounded arithmetic. Soundness: real f32 kernels
//! that bracket a reduction in any tree no deeper than declared never exceed the bound, on
//! random and cancelling inputs. Sensitivity: a kernel that accumulates in bf16 where f32 is
//! declared does exceed it, so the bound is not vacuous.

use super::*;
use crate::elem::{BF16, E4M3, F32};
use crate::inputs::SplitMix64;

/// 2026-10-09: Sum `xs` in f32 by a random binary tree of depth at most `depth`; returns the
/// sum and the tree's depth.
fn f32_tree(xs: &[f32], rng: &mut SplitMix64, depth: u32) -> (f32, u32) {
    if xs.len() == 1 {
        return (xs[0], 0);
    }
    // 2026-10-09: A sequential leaf run once the remaining depth only allows a chain.
    if depth as usize >= xs.len() - 1 && rng.below(2) == 0 {
        let mut s = xs[0];
        for &x in &xs[1..] {
            s += x;
        }
        return (s, xs.len() as u32 - 1);
    }
    let cut = 1 + rng.below(xs.len() as u64 - 1) as usize;
    let half = depth.saturating_sub(1);
    let (a, da) = f32_tree(&xs[..cut], rng, half);
    let (b, db) = f32_tree(&xs[cut..], rng, half);
    (a + b, 1 + da.max(db))
}

#[test]
fn f32_reductions_in_any_tree_stay_inside_the_bound() {
    let mut rng = SplitMix64::new(7);
    let mut worst_ratio: f64 = 0.0;
    for trial in 0..400 {
        let n = 16 + rng.below(2048) as usize;
        let cancel = trial % 3 == 0;
        let xs: Vec<f32> = (0..n)
            .map(|i| {
                let g = rng.gaussian() as f32;
                if cancel && i % 2 == 1 { -g * 0.999 } else { g }
            })
            .collect();
        let terms: Vec<Bounded> = xs.iter().map(|&x| Bounded::exact(f64::from(x))).collect();
        let (got, depth) = f32_tree(&xs, &mut rng, 64);
        let b = sum(&terms, F32, u64::from(depth));
        let err = (f64::from(got) - b.v).abs();
        assert!(
            err <= b.e,
            "trial {trial}: n={n} depth={depth} err={err:e} bound={:e}",
            b.e
        );
        worst_ratio = worst_ratio.max(err / b.e);
    }
    assert!(
        worst_ratio > 0.0,
        "the trials must exercise rounding at all"
    );
}

#[test]
fn products_rounded_then_summed_stay_inside_the_bound() {
    // 2026-10-09: bf16 x bf16 products are exact in f32; E4M3 x bf16 too. Round the inputs, form
    // the products in f32, sum sequentially, round the result to bf16.
    let mut rng = SplitMix64::new(11);
    for _ in 0..300 {
        let k = 64 + rng.below(4096) as usize;
        let a: Vec<f64> = (0..k)
            .map(|_| BF16.round(rng.gaussian() * 3.0).unwrap())
            .collect();
        let w: Vec<f64> = (0..k)
            .map(|_| E4M3.round(rng.gaussian()).unwrap())
            .collect();
        let mut s = 0f32;
        for i in 0..k {
            s += (a[i] as f32) * (w[i] as f32);
        }
        let got = BF16.round(f64::from(s)).unwrap();
        let terms: Vec<Bounded> = (0..k)
            .map(|i| Bounded::exact(a[i]).mul(Bounded::exact(w[i])).round(F32))
            .collect();
        let b = sum(&terms, F32, k as u64).round(BF16);
        assert!(
            (got - b.v).abs() <= b.e,
            "k={k} err={:e} bound={:e}",
            (got - b.v).abs(),
            b.e
        );
    }
}

#[test]
fn bf16_accumulation_where_f32_is_declared_exceeds_the_bound() {
    // 2026-10-09: The sensitivity half of the contract: with a declared tree (16 sequential per
    // lane, 5 shuffle levels, 8 warps sequential) the bound is tight enough that a bf16
    // accumulator (rounding after every add) leaves it on most rows.
    let mut rng = SplitMix64::new(13);
    let (k, rows) = (4096usize, 64);
    let depth = 16 + 5 + 8;
    let mut caught = 0;
    for _ in 0..rows {
        let xs: Vec<f64> = (0..k)
            .map(|_| BF16.round(rng.gaussian()).unwrap())
            .collect();
        let mut s = 0f64;
        for &x in &xs {
            s = BF16.round(s + x).unwrap();
        }
        let terms: Vec<Bounded> = xs.iter().map(|&x| Bounded::exact(x)).collect();
        let b = sum(&terms, F32, depth);
        if (s - b.v).abs() > b.e {
            caught += 1;
        }
    }
    assert!(
        caught * 10 >= rows * 9,
        "bf16 accumulation caught on only {caught}/{rows} rows"
    );
}

#[test]
fn unstatable_bounds_are_infinite_not_optimistic() {
    let near_zero = Bounded { v: 1e-3, e: 2e-3 };
    assert!(Bounded::exact(1.0).div(near_zero).e.is_infinite());
    assert!(near_zero.rsqrt(0.0).e.is_infinite());
    assert!(gamma(1 << 30, BF16.unit_roundoff()).is_infinite());
}

#[test]
fn approximate_functions_cover_their_declared_error() {
    let mut rng = SplitMix64::new(17);
    for _ in 0..10_000 {
        let x = rng.gaussian() * 4.0;
        let xe = (x as f32) as f64;
        let b = Bounded::exact(x).round(F32).exp(2f64.powi(-22));
        // 2026-10-09: A kernel within the declared error: f32 exp of the f32-rounded input,
        // perturbed by just under the declared relative error.
        let k = (xe.exp() as f32) as f64 * (1.0 + 2f64.powi(-23));
        assert!((k - b.v).abs() <= b.e);
        let r = Bounded::exact(x.abs() + 0.1).rsqrt(2f64.powi(-22));
        let kr = 1.0 / (x.abs() + 0.1).sqrt() * (1.0 - 2f64.powi(-23));
        assert!((kr - r.v).abs() <= r.e);
    }
}
