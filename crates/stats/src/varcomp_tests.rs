// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: On synthetic data with known box, run and residual variances the components are
//! recovered and the grand-mean interval covers at the nominal rate; the pooled i.i.d. SE on the
//! same data does not (the negative control). Levels that cannot be separated are reported.

use alloc::vec::Vec;

use super::*;
use crate::rng::SplitMix64;
use crate::sim::{BAND, coverage};

const MU: f64 = 10.0;
const SA: f64 = 1.0;
const SB: f64 = 0.7;
const SE: f64 = 0.5;

fn dataset(r: &mut SplitMix64) -> Vec<(u32, u32, f64)> {
    let mut out = Vec::new();
    for b in 0..4u32 {
        let ab = SA * r.normal();
        for run in 0..3 + b % 3 {
            let br = SB * r.normal();
            for _ in 0..4 + (b + run) % 5 {
                out.push((b, run, MU + ab + br + SE * r.normal()));
            }
        }
    }
    out
}

#[test]
fn components_are_recovered_and_the_interval_covers() {
    let mut r = SplitMix64::new(41);
    let (mut sb, mut sr, mut se) = (0.0, 0.0, 0.0);
    let trials = 600;
    let c = coverage(trials, |_| {
        let v = nested(&dataset(&mut r)).unwrap();
        sb += v.var_box.unwrap();
        sr += v.var_run.unwrap();
        se += v.var_resid;
        v.interval(0.95).contains(MU)
    });
    let t = f(trials);
    assert!(c >= BAND.0 && c <= BAND.1, "varcomp coverage {c}");
    assert!((sb / t / (SA * SA) - 1.0).abs() < 0.20, "box {}", sb / t);
    assert!((sr / t / (SB * SB) - 1.0).abs() < 0.15, "run {}", sr / t);
    assert!((se / t / (SE * SE) - 1.0).abs() < 0.05, "resid {}", se / t);
}

/// 2026-10-22: The negative control: treating every sample as independent ignores the run and
/// box effects and under-covers badly.
#[test]
fn the_pooled_iid_se_under_covers() {
    let mut r = SplitMix64::new(41);
    let c = coverage(600, |_| {
        let d: Vec<f64> = dataset(&mut r).into_iter().map(|o| o.2).collect();
        let s = Summary {
            mean: mean(&d),
            se: libm::sqrt(var(&d) / f(d.len())),
            df: f(d.len() - 1),
        };
        s.interval(0.95, CiMethod::T).contains(MU)
    });
    assert!(c < 0.6, "pooled coverage {c}");
}

#[test]
fn one_box_reports_no_box_component_and_runs_carry_the_se() {
    let mut r = SplitMix64::new(42);
    let mut obs = Vec::new();
    for run in 0..3u32 {
        let br = SB * r.normal();
        for _ in 0..50 {
            obs.push((0u32, run, MU + br + SE * r.normal()));
        }
    }
    let v = nested(&obs).unwrap();
    assert_eq!(v.var_box, None);
    assert!(v.var_run.is_some());
    // 2026-10-22: Three runs: the df are near 2, not 149.
    assert!(v.df < 4.0, "df {}", v.df);
    let naive = libm::sqrt(var(&obs.iter().map(|o| o.2).collect::<Vec<_>>()) / 150.0);
    assert!(v.se_mean > naive);
}

#[test]
fn inseparable_levels_are_reported_merged() {
    let one_per_run = [
        (0, 0, 1.0),
        (0, 1, 2.0),
        (1, 0, 4.0),
        (1, 1, 5.0),
        (2, 0, 3.0),
        (2, 1, 3.5),
    ];
    let v = nested(&one_per_run).unwrap();
    assert_eq!(v.var_run, None);
    assert_eq!(v.merged, [(Level::Run, Level::Sample)]);
    let one_box_one_obs = [(0, 0, 1.0), (0, 1, 2.0), (0, 2, 4.0)];
    let v = nested(&one_box_one_obs).unwrap();
    assert_eq!((v.var_box, v.var_run), (None, None));
    assert!((v.mean - 7.0 / 3.0).abs() < 1e-12);
}

#[test]
fn a_negative_moment_estimate_is_truncated_and_reported() {
    // 2026-10-22: Run means identical within each box: the run mean square is far below the
    // residual one.
    let obs = [
        (0, 0, 0.0),
        (0, 0, 2.0),
        (0, 1, 2.0),
        (0, 1, 0.0),
        (1, 0, 5.0),
        (1, 0, 7.0),
        (1, 1, 7.0),
        (1, 1, 5.0),
    ];
    let v = nested(&obs).unwrap();
    assert_eq!(v.var_run, Some(0.0));
    assert!(v.truncated.contains(&Level::Run));
}

#[test]
fn too_few_or_invalid_observations_are_refused() {
    use metrale_error::ErrorCode;
    assert_eq!(
        nested(&[(0, 0, 1.0)]).unwrap_err().code(),
        ErrorCode::StatsInsufficient
    );
    assert_eq!(
        nested(&[(0, 0, 1.0), (0, 1, f64::NAN)]).unwrap_err().code(),
        ErrorCode::StatsInvalid
    );
}
