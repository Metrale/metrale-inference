// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The uncertainty of a headroom: 95% intervals on every measured site time, its
//! pct-of-floor and its upside, and on the step, its tok/s and the residual, by the moving-block
//! bootstrap over the capture's per-step series (steps are autocorrelated: a block keeps
//! neighbouring steps together; Künsch 1989), with the delta method on pct-of-floor and the
//! residual as a cross-check (Newey-West variance of the mean).
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - No series, no interval: a capture without per-step series gets none, and its report says
//!   "preliminary, n=1 (no CI)"; nothing here invents one.
//! - Every replicate resamples the same step indices for every series (the step, each site, the
//!   kernels outside the plan), so the joint statistics (upside, residual) keep their covariance.
//! - Deterministic: [`SEED`], [`REPS`] and the block length are stated in the report.

use crate::prelude::*;

use alloc::collections::BTreeMap;

use super::headroom_trace::Trace;
use super::stats::{
    Interval, SplitMix64, block_len, block_resample, hac_variance_of_mean, median, percentile_ci,
};

/// 2026-10-22: Bootstrap replicates.
pub const REPS: usize = 1000;
/// 2026-10-22: The resampling seed (`"headroom"` in ASCII).
pub const SEED: u64 = 0x6865_6164_726f_6f6d;
/// 2026-10-22: Interval level.
pub const LEVEL: f64 = 0.95;
/// 2026-10-22: The normal quantile of [`LEVEL`], for the delta method.
const Z: f64 = 1.959_963_984_540_054;

/// 2026-10-22: Per-step series of one step's parts, every one over the same steps.
#[derive(Debug, Clone)]
pub struct Series {
    /// 2026-10-22: Steps per capture.
    pub segments: Vec<usize>,
    /// 2026-10-22: Step durations.
    pub step: Vec<f64>,
    /// 2026-10-22: Per measured site (index into the sites): (total, exclusive) per step.
    pub sites: BTreeMap<usize, (Vec<f64>, Vec<f64>)>,
    /// 2026-10-22: Exclusive time of the kernels outside the plan, per step.
    pub outside: Vec<f64>,
}

/// 2026-10-22: Sum `rows`' series into one (total, exclusive) pair over `n` steps.
pub fn sum_series<'a>(
    rows: impl Iterator<Item = &'a super::headroom_trace::TraceKernel>,
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    let (mut t, mut e) = (vec![0.0; n], vec![0.0; n]);
    for k in rows {
        for i in 0..n {
            t[i] += k.step_total_us.get(i).copied().unwrap_or(0.0);
            e[i] += k.step_exclusive_us.get(i).copied().unwrap_or(0.0);
        }
    }
    (t, e)
}

/// 2026-10-22: Whether `t` carries series (checked by `Trace::check`).
pub fn has_series(t: &Trace) -> bool {
    !t.step_us.is_empty()
}

/// 2026-10-22: A measured site's floor and what its upside targets.
#[derive(Debug, Clone, Copy)]
pub struct SiteFloor {
    /// 2026-10-22: Floor, us per step.
    pub floor_us: f64,
    /// 2026-10-22: Target fraction of the floor, (0, 1].
    pub p: f64,
}

/// 2026-10-22: The intervals of one measured site.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SiteCi {
    /// 2026-10-22: Mean total per step.
    pub total_us: Interval,
    /// 2026-10-22: Mean exclusive per step.
    pub exclusive_us: Interval,
    /// 2026-10-22: Median of the per-step totals (robust to a slow step), and its interval.
    pub median_total_us: f64,
    /// 2026-10-22: Its interval.
    pub median_total_ci: Interval,
    /// 2026-10-22: pct of floor (bootstrap).
    pub pct_of_floor: Interval,
    /// 2026-10-22: pct of floor (delta method, HAC variance): the cross-check.
    pub pct_of_floor_delta: Interval,
    /// 2026-10-22: Saved at 100% of the floor.
    pub saved_full_us: Interval,
    /// 2026-10-22: tok/s with this site at 100%.
    pub tok_s_full: Interval,
    /// 2026-10-22: Saved at the target fraction.
    pub saved_target_us: Interval,
    /// 2026-10-22: tok/s at the target fraction.
    pub tok_s_target: Interval,
}

/// 2026-10-22: The step's intervals.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct TotalsCi {
    /// 2026-10-22: Mean step.
    pub step_us: Interval,
    /// 2026-10-22: tok/s.
    pub step_tok_s: Interval,
    /// 2026-10-22: Residual (bootstrap).
    pub residual_us: Interval,
    /// 2026-10-22: Residual (delta method on the per-step residual series).
    pub residual_delta: Interval,
}

/// 2026-10-22: The intervals of a step.
#[derive(Debug, Clone)]
pub struct StepCi {
    /// 2026-10-22: Per measured site.
    pub sites: BTreeMap<usize, SiteCi>,
    /// 2026-10-22: The step.
    pub totals: TotalsCi,
    /// 2026-10-22: Block length used.
    pub block: usize,
}

fn mean_at(xs: &[f64], idx: &[usize]) -> f64 {
    idx.iter().map(|&i| xs[i]).sum::<f64>() / idx.len() as f64
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn delta(point: f64, grad: f64, var: f64) -> Interval {
    let half = Z * grad.abs() * libm::sqrt(var);
    Interval {
        lo: point - half,
        hi: point + half,
    }
}

fn ci(reps: &[f64]) -> Interval {
    percentile_ci(reps, LEVEL).unwrap_or(Interval {
        lo: f64::NAN,
        hi: f64::NAN,
    })
}

/// 2026-10-22: Bootstrap the step's statistics. `floors` gives each measured site's floor;
/// `tokens` is tokens per step.
pub fn step_ci(s: &Series, floors: &BTreeMap<usize, SiteFloor>, tokens: f64) -> StepCi {
    let n = s.step.len();
    let block = block_len(n);
    let mut rng = SplitMix64::new(SEED);
    let keys: Vec<usize> = s.sites.keys().copied().collect();
    let mut reps: BTreeMap<usize, [Vec<f64>; 7]> = keys
        .iter()
        .map(|&k| (k, core::array::from_fn(|_| Vec::with_capacity(REPS))))
        .collect();
    let (mut step_r, mut tok_r, mut res_r) = (Vec::new(), Vec::new(), Vec::new());
    let mut scratch = Vec::with_capacity(n);
    for _ in 0..REPS {
        let idx = block_resample(&s.segments, block, &mut rng);
        let step = mean_at(&s.step, &idx);
        let mut excl = mean_at(&s.outside, &idx);
        for &k in &keys {
            let (tot, exc) = &s.sites[&k];
            let (t, e) = (mean_at(tot, &idx), mean_at(exc, &idx));
            excl += e;
            let f = floors[&k];
            let saved = |p: f64| (t - f.floor_us / p).clamp(0.0, e);
            scratch.clear();
            scratch.extend(idx.iter().map(|&i| tot[i]));
            let r = reps.get_mut(&k).expect("key");
            r[0].push(t);
            r[1].push(e);
            r[2].push(median(&scratch).unwrap_or(0.0));
            r[3].push(if t > 0.0 { 100.0 * f.floor_us / t } else { 0.0 });
            r[4].push(saved(1.0));
            r[5].push(saved(f.p));
            r[6].push(tokens * 1e6 / (step - saved(1.0)));
        }
        step_r.push(step);
        tok_r.push(tokens * 1e6 / step);
        res_r.push(step - excl);
    }
    let lag = block;
    let sites = keys
        .iter()
        .map(|&k| {
            let (tot, _) = &s.sites[&k];
            let r = &reps[&k];
            let f = floors[&k];
            let m = mean(tot);
            let var = hac_variance_of_mean(tot, lag).unwrap_or(0.0);
            let tok_target: Vec<f64> = r[0]
                .iter()
                .zip(&step_r)
                .zip(&r[5])
                .map(|((_, st), sv)| tokens * 1e6 / (st - sv))
                .collect();
            (
                k,
                SiteCi {
                    total_us: ci(&r[0]),
                    exclusive_us: ci(&r[1]),
                    median_total_us: median(tot).unwrap_or(0.0),
                    median_total_ci: ci(&r[2]),
                    pct_of_floor: ci(&r[3]),
                    pct_of_floor_delta: delta(
                        100.0 * f.floor_us / m,
                        100.0 * f.floor_us / (m * m),
                        var,
                    ),
                    saved_full_us: ci(&r[4]),
                    tok_s_full: ci(&r[6]),
                    saved_target_us: ci(&r[5]),
                    tok_s_target: ci(&tok_target),
                },
            )
        })
        .collect();
    let mut resid: Vec<f64> = s.step.iter().zip(&s.outside).map(|(a, b)| a - b).collect();
    for (_, e) in s.sites.values() {
        for (r, x) in resid.iter_mut().zip(e) {
            *r -= x;
        }
    }
    let rv = hac_variance_of_mean(&resid, lag).unwrap_or(0.0);
    StepCi {
        sites,
        totals: TotalsCi {
            step_us: ci(&step_r),
            step_tok_s: ci(&tok_r),
            residual_us: ci(&res_r),
            residual_delta: delta(mean(&resid), 1.0, rv),
        },
        block,
    }
}
