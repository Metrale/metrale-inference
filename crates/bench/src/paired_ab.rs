// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The verdict of a paired in-serve A/B run (`METRALE_PAIRED_AB`,
//! `metrale_model_layers::layers::ops::paired_ab`) from its `PAIRED_AB` log lines. GPU-free and
//! pure: the caller reads the log.
//!
//! Method. A run's steps split into requests where the KV length jumps (a new prompt). In a
//! request, the arms run in cycles of a palindromic order (A B B A for two arms, or its mirror,
//! picked per cycle by the instrument); the first `warm` cycles are dropped (each arm's first
//! steps capture its graph). Per cycle and per arm X != A the paired difference is
//! `d = mean(A steps) - mean(X steps)` (positive: X is faster), which cancels a linear drift
//! inside the cycle. Over a request's cycles:
//! - the location of `d` is Huber's M-estimate at `huber_k` (2026-10-23: the first A/A run's
//!   step times had a 3.5-5.6 ms SD from a heavy right tail, p99 48-57 ms on a 31-34 ms median,
//!   so the mean's 95% half-width was 0.21-0.46% of the step; Huber at 1.345 gave 0.12-0.16%
//!   on the same cycles). It is shift-equivariant, so a lever that moves every step by `x`
//!   moves it by `x`; the plain mean is reported beside it;
//! - its BCa interval from the stationary block bootstrap (Politis-White block,
//!   `metrale_stats::bootstrap::bca_series`), and a Wald summary with the bootstrap SE (half the
//!   95% width over 1.96) and the HAC df (`metrale_stats::hac`);
//! - TOST of |d| < margin (a share of the A step) and one-sided superiority d > 0 on it;
//! - Holm over every (request, arm) superiority p.
//!
//! Owner: metrale-bench.
//! Invariants:
//! - A cycle without exactly two steps of every arm (a dropped span) is dropped whole, never
//!   imputed.
//! - Lines that are not `PAIRED_AB step=` lines are ignored; a malformed one is an error.

use anyhow::{Context, Result, bail};
use metrale_stats::Summary;
use metrale_stats::bootstrap::bca_series;
use metrale_stats::compare::{holm, superiority, tost};
use metrale_stats::hac::{ess, mean_summary};
use metrale_stats::robust::huber;

/// 2026-10-23: One logged step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Step {
    pub step: u64,
    pub arm: usize,
    pub kv: usize,
    pub us: f64,
}

/// 2026-10-23: The steps of `text`'s `PAIRED_AB step=` lines, in log order.
pub fn parse(text: &str) -> Result<Vec<Step>> {
    let mut out = Vec::new();
    for line in text.lines() {
        let Some(at) = line.find("PAIRED_AB step=") else {
            continue;
        };
        let rest = &line[at + "PAIRED_AB ".len()..];
        let field = |k: &str| -> Result<&str> {
            rest.split_whitespace()
                .find_map(|kv| kv.strip_prefix(k).and_then(|v| v.strip_prefix('=')))
                .with_context(|| format!("PAIRED_AB line without `{k}=`: {line}"))
        };
        out.push(Step {
            step: field("step")?.parse()?,
            arm: field("arm")?.parse()?,
            kv: field("kv")?.parse()?,
            us: field("us")?.parse()?,
        });
    }
    Ok(out)
}

/// 2026-10-23: How to judge a run.
#[derive(Clone, Copy, Debug)]
pub struct Plan {
    /// 2026-10-23: Arms the run alternated (the spec's).
    pub arms: usize,
    /// 2026-10-23: Leading cycles of each request to drop.
    pub warm: usize,
    /// 2026-10-23: The TOST margin as a fraction of the A step (0.0025 is 0.25%).
    pub margin: f64,
    /// 2026-10-23: Huber's tuning constant for the location of the cycle differences.
    pub huber_k: f64,
    /// 2026-10-23: Bootstrap replicates and seed.
    pub reps: usize,
    pub seed: u64,
}

/// 2026-10-23: One arm of one request against A.
#[derive(Clone, Debug)]
pub struct ArmVerdict {
    /// 2026-10-23: The request's first KV length (its prompt, about).
    pub kv0: usize,
    pub arm: usize,
    pub cycles: usize,
    pub a_mean_us: f64,
    /// 2026-10-23: Huber location of A − X, µs per step (positive: X faster), its BCa 95%
    /// interval.
    pub d_us: f64,
    pub d_lo: f64,
    pub d_hi: f64,
    /// 2026-10-23: The plain mean of A − X, µs per step.
    pub d_mean_us: f64,
    pub ess: f64,
    pub tost_p: f64,
    pub equivalent: bool,
    pub superior_p: f64,
    /// 2026-10-23: Superiority after Holm over every verdict of the run.
    pub superior_holm: bool,
}

impl ArmVerdict {
    /// 2026-10-23: `x` as a percentage of the A step.
    pub fn pct(&self, x: f64) -> f64 {
        100.0 * x / self.a_mean_us
    }
}

/// 2026-10-23: The most a verify step advances its sequence's KV: a K-row verify keeps at most
/// K tokens, and the instrument runs K = 2 (slack for a deeper verify).
const MAX_KV_ADVANCE: usize = 8;

/// 2026-10-23: `steps` split into requests: a new request starts where the KV length does not
/// advance by 0..=`MAX_KV_ADVANCE` tokens from the step before (a new prompt was prefilled).
fn requests(steps: &[Step]) -> Vec<&[Step]> {
    let mut out = Vec::new();
    let mut start = 0;
    for i in 1..steps.len() {
        let (a, b) = (steps[i - 1].kv, steps[i].kv);
        if b < a || b - a > MAX_KV_ADVANCE {
            out.push(&steps[start..i]);
            start = i;
        }
    }
    if start < steps.len() {
        out.push(&steps[start..]);
    }
    out
}

/// 2026-10-23: Per cycle of `req`, the mean step of each arm; cycles without exactly two steps
/// of every arm dropped. Cycles are aligned on the instrument's global step counter.
fn cycles(req: &[Step], arms: usize) -> Vec<Vec<f64>> {
    let len = 2 * arms as u64;
    let mut by: std::collections::BTreeMap<u64, Vec<&Step>> = Default::default();
    for s in req {
        by.entry(s.step / len).or_default().push(s);
    }
    by.into_values()
        .filter_map(|c| {
            let complete = c.len() == 2 * arms
                && (0..arms).all(|a| c.iter().filter(|s| s.arm == a).count() == 2);
            complete.then(|| {
                (0..arms)
                    .map(|a| {
                        let v: Vec<f64> = c.iter().filter(|s| s.arm == a).map(|s| s.us).collect();
                        v.iter().sum::<f64>() / v.len() as f64
                    })
                    .collect()
            })
        })
        .collect()
}

/// 2026-10-23: The verdicts of `steps` under `plan`, one per (request, arm != A).
pub fn judge(steps: &[Step], plan: Plan) -> Result<Vec<ArmVerdict>> {
    if plan.arms < 2 {
        bail!("a paired run has at least two arms, not {}", plan.arms);
    }
    if let Some(s) = steps.iter().find(|s| s.arm >= plan.arms) {
        bail!(
            "step {} ran arm {} of a {}-arm run",
            s.step,
            s.arm,
            plan.arms
        );
    }
    let mut out = Vec::new();
    for req in requests(steps) {
        let cs: Vec<Vec<f64>> = cycles(req, plan.arms).into_iter().skip(plan.warm).collect();
        if cs.len() < 16 {
            continue;
        }
        let a_mean = cs.iter().map(|c| c[0]).sum::<f64>() / cs.len() as f64;
        for arm in 1..plan.arms {
            let d: Vec<f64> = cs.iter().map(|c| c[0] - c[arm]).collect();
            let k = plan.huber_k;
            // 2026-10-23: A replicate whose Huber iteration fails would bias the interval:
            // NaN, which `bca_series` refuses, rather than a silent substitute.
            let loc = |x: &[f64]| huber(x, k).unwrap_or(f64::NAN);
            let est = huber(&d, k)?;
            let ci = bca_series(&d, loc, plan.reps, 0.95, plan.seed)?;
            let hac: Summary = mean_summary(&d)?;
            let robust = Summary {
                mean: est,
                se: (ci.hi - ci.lo) / (2.0 * 1.959_963_985),
                df: hac.df,
            };
            let margin = plan.margin * a_mean;
            let t = tost(robust, margin, 0.05);
            let sup = superiority(robust, 0.0, 0.05);
            out.push(ArmVerdict {
                kv0: req[0].kv,
                arm,
                cycles: cs.len(),
                a_mean_us: a_mean,
                d_us: est,
                d_lo: ci.lo,
                d_hi: ci.hi,
                d_mean_us: hac.mean,
                ess: ess(&d)?,
                tost_p: t.p,
                equivalent: t.equivalent,
                superior_p: sup.p,
                superior_holm: false,
            });
        }
    }
    let ps: Vec<f64> = out.iter().map(|v| v.superior_p).collect();
    for (v, r) in out.iter_mut().zip(holm(&ps, 0.05)) {
        v.superior_holm = r;
    }
    Ok(out)
}

#[cfg(test)]
#[path = "paired_ab_tests.rs"]
mod tests;
