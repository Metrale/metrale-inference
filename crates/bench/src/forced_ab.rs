// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The verdict of serves that replayed ONE recorded token trajectory
//! (`METRALE_PAIRED_AB_TRAJ=replay:...`, `metrale_server::scheduler::forced_traj`). Every replay
//! runs the same steps (request, output index, KV length) over the same tokens, so a step of a
//! control serve and the same step of a lever serve differ only by their kernels; and each serve
//! logs what its own numerics would have accepted on that trajectory, so acceptance is compared
//! on identical drafts-and-contexts too. GPU-free and pure: the caller reads the logs.
//!
//! Step time. Within a (control, lever) serve pair, per request, the step-matched differences
//! `d = control − lever` (positive: the lever is faster) over the lever steps of the chosen arm,
//! after `warm` steps; Huber location with a stationary-bootstrap BCa interval (within-pair
//! precision). Across pairs the unit is the PAIR (a serve carries a boot-to-boot offset that no
//! within-serve interval sees, measured 2026-10-22 at ~0.8% of the step): mean of the pair
//! estimates, t interval with n−1 df, TOST at the margin and superiority, Holm over requests.
//!
//! Acceptance. Per step: control `own_acc` (its own draft equals its own row-0 pick) and lever
//! `own_acc`, on the same step; McNemar on the discordant steps, pooled over pairs.
//!
//! Owner: metrale-bench.
//! Invariants:
//! - Steps are matched by (request key, output index) and must agree on the KV length; a
//!   mismatch is an error (two serves did not replay the same record).
//! - A lever step counts for the arm only when it and the step before it ran that arm, so its
//!   own draft came from the same numerics (a paired run alternates arms).

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use metrale_stats::Summary;
use metrale_stats::bootstrap::bca_series;
use metrale_stats::compare::{holm, superiority, tost};
use metrale_stats::dist::t_ppf;
use metrale_stats::robust::huber;

use crate::paired_ab::Step;

/// 2026-10-23: One replayed K=2 verify (`PAIRED_TRAJ replay k2`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrajStep {
    pub key: u64,
    pub out: usize,
    pub kv: usize,
    pub forced_acc: bool,
    pub own_acc: bool,
    pub agree0: bool,
}

/// 2026-10-23: The `PAIRED_TRAJ replay k2` lines of a log, in order.
pub fn parse_traj(text: &str) -> Result<Vec<TrajStep>> {
    let mut out = Vec::new();
    for line in text.lines() {
        let Some(at) = line.find("PAIRED_TRAJ replay k2 ") else {
            continue;
        };
        let rest = &line[at..];
        let f = |k: &str| -> Result<&str> {
            rest.split_whitespace()
                .find_map(|kv| kv.strip_prefix(k).and_then(|v| v.strip_prefix('=')))
                .with_context(|| format!("PAIRED_TRAJ line without `{k}=`: {line}"))
        };
        out.push(TrajStep {
            key: u64::from_str_radix(f("key")?, 16)?,
            out: f("out")?.parse()?,
            kv: f("kv")?.parse()?,
            forced_acc: f("forced_acc")? == "1",
            own_acc: f("own_acc")? == "1",
            agree0: f("agree0")? == "1",
        });
    }
    Ok(out)
}

/// 2026-10-23: A replayed step with its timing: the arm it ran, the arm of the step before it
/// (`None` at a request's first), its device time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Timed {
    pub t: TrajStep,
    pub arm: usize,
    pub prev_arm: Option<usize>,
    pub us: f64,
}

/// 2026-10-23: `steps` (`PAIRED_AB`) joined to `traj` in log order on the KV length: both are
/// in step order, a dropped span leaves a trajectory step untimed (skipped).
pub fn join(steps: &[Step], traj: &[TrajStep]) -> Vec<Timed> {
    let mut by_kv: HashMap<usize, Vec<&Step>> = HashMap::new();
    for s in steps {
        by_kv.entry(s.kv).or_default().push(s);
    }
    let mut used: HashMap<usize, usize> = HashMap::new();
    let mut out: Vec<Timed> = Vec::new();
    for t in traj {
        let n = used.entry(t.kv).or_insert(0);
        let Some(s) = by_kv.get(&t.kv).and_then(|v| v.get(*n)) else {
            continue;
        };
        *n += 1;
        let prev_arm = out
            .last()
            .filter(|p| p.t.key == t.key && p.t.kv < t.kv)
            .map(|p| p.arm);
        out.push(Timed {
            t: *t,
            arm: s.arm,
            prev_arm,
            us: s.us,
        });
    }
    out
}

/// 2026-10-23: Per arm of one serve: steps, own acceptance, row-0 agreement with the record.
pub fn acceptance_by_arm(timed: &[Timed], arms: usize) -> Vec<(usize, usize, usize)> {
    (0..arms)
        .map(|a| {
            let v: Vec<&Timed> = timed
                .iter()
                .filter(|x| x.arm == a && x.prev_arm == Some(a))
                .collect();
            (
                v.len(),
                v.iter().filter(|x| x.t.own_acc).count(),
                v.iter().filter(|x| x.t.agree0).count(),
            )
        })
        .collect()
}

/// 2026-10-23: How to judge a forced A/B.
#[derive(Clone, Copy, Debug)]
pub struct ForcedPlan {
    /// 2026-10-23: The lever serves' arm under test; `None`: every arm (a lever serve whose
    /// arms are all the lever, e.g. an `aa` run with the lever set serve-wide).
    pub lever_arm: Option<usize>,
    /// 2026-10-23: Leading matched steps of each request to drop (graph capture).
    pub warm: usize,
    pub margin: f64,
    pub huber_k: f64,
    pub reps: usize,
    pub seed: u64,
}

/// 2026-10-23: One request's verdict over the serve pairs.
#[derive(Clone, Debug)]
pub struct ReqVerdict {
    pub kv0: usize,
    /// 2026-10-23: Per pair: matched steps, control mean us, Huber d us and its BCa interval.
    pub pairs: Vec<(usize, f64, f64, f64, f64)>,
    /// 2026-10-23: Mean over pairs of d (us) and its t interval (n−1 df); `None` with one pair.
    pub d_us: f64,
    pub ci: Option<(f64, f64)>,
    pub control_us: f64,
    pub tost_p: f64,
    pub superior_p: f64,
    pub superior_holm: bool,
    /// 2026-10-23: Acceptance over every pair's matched steps: control accepts, lever accepts,
    /// steps; McNemar discordant (control only, lever only) and its two-sided p.
    pub acc: (usize, usize, usize),
    pub discordant: (usize, usize),
    pub mcnemar_p: f64,
}

/// 2026-10-23: Two-sided McNemar p with continuity correction (chi-square, 1 df).
fn mcnemar(b: usize, c: usize) -> f64 {
    if b + c == 0 {
        return 1.0;
    }
    let x = ((b as f64 - c as f64).abs() - 1.0).max(0.0);
    let chi = x * x / (b + c) as f64;
    // 2026-10-23: chi-square(1) survival = 2 * (1 − Φ(sqrt(chi))).
    2.0 * (1.0 - metrale_stats::dist::norm_cdf(chi.sqrt()))
}

/// 2026-10-23: The verdict of `pairs` of (control, lever) joined serves under `plan`, one per
/// request key present in every serve.
pub fn judge_forced(
    pairs: &[(Vec<Timed>, Vec<Timed>)],
    plan: ForcedPlan,
) -> Result<Vec<ReqVerdict>> {
    if pairs.is_empty() {
        bail!("a forced A/B needs at least one (control, lever) pair");
    }
    let index = |v: &[Timed]| -> HashMap<(u64, usize), Timed> {
        v.iter().map(|x| ((x.t.key, x.t.out), *x)).collect()
    };
    let mut keys: Vec<(u64, usize)> = Vec::new();
    for x in &pairs[0].0 {
        if !keys.iter().any(|k| k.0 == x.t.key) {
            keys.push((x.t.key, x.t.kv));
        }
    }
    let mut out = Vec::new();
    for (key, kv0) in keys {
        let mut per_pair = Vec::new();
        let (mut acc, mut disc) = ((0, 0, 0), (0, 0));
        for (c, l) in pairs {
            let li = index(l);
            let mut cs: Vec<&Timed> = c.iter().filter(|x| x.t.key == key).collect();
            cs.sort_by_key(|x| x.t.out);
            let mut d = Vec::new();
            let mut cmean = 0.0;
            for x in cs.iter().skip(plan.warm) {
                let Some(y) = li.get(&(key, x.t.out)) else {
                    continue;
                };
                if y.t.kv != x.t.kv {
                    bail!(
                        "request {key:016x} out {}: kv {} vs {}: the serves replayed different records",
                        x.t.out,
                        x.t.kv,
                        y.t.kv
                    );
                }
                if let Some(a) = plan.lever_arm
                    && (y.arm != a || y.prev_arm != Some(a))
                {
                    continue;
                }
                d.push(x.us - y.us);
                cmean += x.us;
                acc.0 += usize::from(x.t.own_acc);
                acc.1 += usize::from(y.t.own_acc);
                acc.2 += 1;
                match (x.t.own_acc, y.t.own_acc) {
                    (true, false) => disc.0 += 1,
                    (false, true) => disc.1 += 1,
                    _ => {}
                }
            }
            if d.len() < 32 {
                continue;
            }
            let k = plan.huber_k;
            let loc = |s: &[f64]| huber(s, k).unwrap_or(f64::NAN);
            let est = huber(&d, k)?;
            let ci = bca_series(&d, loc, plan.reps, 0.95, plan.seed)?;
            per_pair.push((d.len(), cmean / d.len() as f64, est, ci.lo, ci.hi));
        }
        if per_pair.is_empty() {
            continue;
        }
        let n = per_pair.len() as f64;
        let mean = per_pair.iter().map(|p| p.2).sum::<f64>() / n;
        let control_us = per_pair.iter().map(|p| p.1).sum::<f64>() / n;
        let (ci, tost_p, superior_p) = if per_pair.len() >= 2 {
            let var = per_pair.iter().map(|p| (p.2 - mean).powi(2)).sum::<f64>() / (n - 1.0);
            let s = Summary {
                mean,
                se: (var / n).sqrt(),
                df: n - 1.0,
            };
            let h = t_ppf(0.975, s.df) * s.se;
            (
                Some((mean - h, mean + h)),
                tost(s, plan.margin * control_us, 0.05).p,
                superiority(s, 0.0, 0.05).p,
            )
        } else {
            (None, f64::NAN, f64::NAN)
        };
        out.push(ReqVerdict {
            kv0,
            pairs: per_pair,
            d_us: mean,
            ci,
            control_us,
            tost_p,
            superior_p,
            superior_holm: false,
            acc,
            discordant: disc,
            mcnemar_p: mcnemar(disc.0, disc.1),
        });
    }
    let ps: Vec<f64> = out
        .iter()
        .map(|v| {
            if v.superior_p.is_nan() {
                1.0
            } else {
                v.superior_p
            }
        })
        .collect();
    for (v, r) in out.iter_mut().zip(holm(&ps, 0.05)) {
        v.superior_holm = r;
    }
    Ok(out)
}

#[cfg(test)]
#[path = "forced_ab_tests.rs"]
mod tests;
