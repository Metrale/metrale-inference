// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Headroom intervals on the toy model: n=1 without series is labelled and carries no
//! interval, a constant series gives a point interval and a noisy one brackets its mean (bootstrap
//! and delta agree), overlapping upsides are flagged tied, pooling keeps captures apart and refuses
//! mixed configurations, and a series that does not add up to its totals is refused.
//!
//! Owner: metrale-circuit tests.
//! Invariants: fixtures come from the parent `headroom_tests` module.

use super::*;

/// 2026-10-22: A capture of `n` steps of 100 us whose lm_head takes `lm(i)` us in step i (all
/// exclusive), and whose ffn norm takes `norm(i)` (all exclusive).
fn series_trace(n: usize, lm: impl Fn(usize) -> f64, norm: impl Fn(usize) -> f64) -> Trace {
    let row = |name: &str, f: &dyn Fn(usize) -> f64, calls: u64| {
        let v: Vec<f64> = (0..n).map(f).collect();
        let total: f64 = v.iter().sum();
        let mut k = kernel(name, None, calls * n as u64, total, total);
        k.step_total_us = v.clone();
        k.step_exclusive_us = v;
        k
    };
    let mut t = trace(vec![row("lm_head", &lm, 1), row("norm", &norm, 2)]);
    t.steps = n as u64;
    t.window_us = 100.0 * n as f64;
    t.step_us = vec![100.0; n];
    t
}

#[test]
fn a_capture_without_series_is_labelled_n1_with_no_interval() {
    let t = trace(vec![kernel("lm_head", None, 2, 20.0, 20.0)]);
    let h = run("fp4dev", Mode::Decode, 1, &inputs(Some(&t), None)).unwrap();
    assert!(
        h.uncertainty.starts_with("preliminary, n=1 (no CI)"),
        "{}",
        h.uncertainty
    );
    assert!(h.sites.iter().all(|r| r.ci.is_none() && !r.tied));
    assert!(h.totals.ci.is_none());
    let md = super::super::render_headroom(&h, "test");
    assert!(!md.contains("Cross-check"), "no interval, no cross-check");
}

#[test]
fn a_constant_series_has_a_point_interval_and_a_noisy_one_brackets_its_mean() {
    let flat = series_trace(64, |_| 10.0, |_| 2.0);
    let h = run("fp4dev", Mode::Decode, 1, &inputs(Some(&flat), None)).unwrap();
    let c = site(&h, "head [lm_head]")
        .ci
        .clone()
        .expect("series give an interval");
    assert_eq!((c.total_us.lo, c.total_us.hi), (10.0, 10.0));
    assert!(
        h.uncertainty
            .starts_with("preliminary, n=1 capture: 95% intervals"),
        "{}",
        h.uncertainty
    );
    let tc = h.totals.ci.as_ref().unwrap();
    // Residual = 100 - 10 - 2 every step: a point.
    assert!((tc.residual_us.lo - 88.0).abs() < 1e-9 && (tc.residual_us.hi - 88.0).abs() < 1e-9);
    // Noisy: 10 +- 4 alternating in runs of 4 steps. The interval brackets the mean, has width,
    // and the delta method agrees with the bootstrap within a factor of two in width.
    let noisy = series_trace(256, |i| if (i / 4) % 2 == 0 { 14.0 } else { 6.0 }, |_| 2.0);
    let h = run("fp4dev", Mode::Decode, 1, &inputs(Some(&noisy), None)).unwrap();
    let r = site(&h, "head [lm_head]");
    let c = r.ci.as_ref().unwrap();
    assert!(
        c.total_us.lo < 10.0 && 10.0 < c.total_us.hi,
        "{:?}",
        c.total_us
    );
    let p = r.pct_of_floor.unwrap();
    assert!(c.pct_of_floor.lo < p && p < c.pct_of_floor.hi);
    let (wb, wd) = (
        c.pct_of_floor.hi - c.pct_of_floor.lo,
        c.pct_of_floor_delta.hi - c.pct_of_floor_delta.lo,
    );
    assert!(
        wb > 0.0 && wd > 0.0 && wb / wd < 2.0 && wd / wb < 2.0,
        "bootstrap {wb} vs delta {wd}"
    );
    let u = r.upside_full.unwrap().saved_us;
    assert!(c.saved_full_us.lo <= u && u <= c.saved_full_us.hi);
}

#[test]
fn sites_whose_upside_intervals_overlap_are_flagged_as_tied() {
    // lm_head saves ~10 us, norm ~2 (two layers): with step noise of +-1 they do not overlap.
    let apart = series_trace(128, |i| 10.0 + if i % 3 == 0 { 1.0 } else { -0.5 }, |_| 2.0);
    let h = run("fp4dev", Mode::Decode, 1, &inputs(Some(&apart), None)).unwrap();
    assert!(h.sites.iter().all(|r| !r.tied));
    // Both near 2 us with noise: tied.
    let close = series_trace(
        128,
        |i| 2.0 + if (i / 2) % 2 == 0 { 0.5 } else { -0.5 },
        |i| 2.0 + if (i / 3) % 2 == 0 { 0.5 } else { -0.5 },
    );
    let h = run("fp4dev", Mode::Decode, 1, &inputs(Some(&close), None)).unwrap();
    assert!(site(&h, "head [lm_head]").tied && site(&h, "ffn [norm]").tied);
    let md = super::super::render_headroom(&h, "test");
    assert!(md.contains("| 1 ≈ |"), "{md}");
}

#[test]
fn pooling_keeps_captures_apart_and_refuses_different_configurations() {
    let a = series_trace(32, |_| 10.0, |_| 2.0);
    let mut b = series_trace(16, |_| 12.0, |_| 2.0);
    b.source = "second".into();
    let p = super::super::headroom_trace::pool(&[a.clone(), b.clone()]).unwrap();
    assert_eq!(
        (p.steps, p.segments.clone(), p.captures()),
        (48, vec![32, 16], 2)
    );
    let lm = p.kernels.iter().find(|k| k.kernel == "lm_head").unwrap();
    assert_eq!(lm.total_us, 32.0 * 10.0 + 16.0 * 12.0);
    let h = run("fp4dev", Mode::Decode, 1, &inputs(Some(&p), None)).unwrap();
    assert!(
        h.uncertainty.starts_with("n=2 captures pooled"),
        "{}",
        h.uncertainty
    );
    b.plan.kv_basis = Some("the second capture's own usage file".into());
    super::super::headroom_trace::pool(&[a.clone(), b.clone()]).expect("a basis is provenance");
    let mut c = b.clone();
    c.plan.execution.values_mut().for_each(|e| {
        e.by = super::super::headroom_trace::ExecutedBy::Plan;
        e.sites = vec!["*".into()];
    });
    let e = super::super::headroom_trace::pool(&[a.clone(), c]).unwrap_err();
    assert!(e.contains("would mix configurations"), "{e}");
    b.plan.kv_tokens = Some(7);
    let e = super::super::headroom_trace::pool(&[a, b]).unwrap_err();
    assert!(e.contains("would mix configurations"), "{e}");
}

#[test]
fn a_series_that_does_not_add_up_to_its_totals_is_refused() {
    let mut t = series_trace(8, |_| 10.0, |_| 2.0);
    t.kernels[0].step_total_us[3] = 99.0;
    let e = t.check().unwrap_err();
    assert!(e.contains("add up to its totals"), "{e}");
}

#[test]
fn the_step_interval_widens_for_autocorrelated_steps() {
    // 240 steps in runs of 20 slow (110 us) and 20 fast (90 us): neighbouring steps move
    // together, so the interval of the mean step must be far wider than the i.i.d. one
    // (2 * 1.96 * sd / sqrt(n) = 2.53 us). Resampling single steps would give about that.
    use super::super::headroom_ci::{Series, step_ci};
    let n = 240;
    let step: Vec<f64> = (0..n)
        .map(|i| if (i / 20) % 2 == 0 { 110.0 } else { 90.0 })
        .collect();
    let s = Series {
        segments: vec![n],
        step: step.clone(),
        sites: BTreeMap::new(),
        outside: vec![0.0; n],
    };
    let c = step_ci(&s, &BTreeMap::new(), 1.0);
    let iid = 2.0 * 1.959_964 * 10.0 / (n as f64).sqrt();
    let w = c.totals.step_us.hi - c.totals.step_us.lo;
    assert_eq!(c.block, 7, "block = ceil(240^(1/3))");
    assert!(
        w > 1.5 * iid,
        "step interval {w} vs i.i.d. {iid}: blocks were not used"
    );
    // The HAC delta interval on the residual (= the step here) agrees within a factor of 2.
    let d = c.totals.residual_delta.hi - c.totals.residual_delta.lo;
    assert!(
        d > 1.5 * iid && (w / d - 1.0).abs() < 1.0,
        "bootstrap {w} vs delta {d}"
    );
}
