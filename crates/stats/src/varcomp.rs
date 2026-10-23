// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Random-effects variance components for samples within runs within boxes:
//! y = μ + a_box + b_run(box) + e. The interval of μ must carry the between-run and between-box
//! variance, not only the sample count: three runs of 1000 samples are three observations of the
//! run effect. Estimation is the ANOVA method of moments for unbalanced nested designs (Searle,
//! Casella & McCulloch 1992, §4.6): mean squares at each level equated to their expectations,
//! negative components truncated to 0 and reported in [`VarComp::truncated`]. μ̂ is the mean of
//! box means of run means (each box, and each run within its box, weighs equally), its variance
//! the matching combination of the components, and its df Satterthwaite's (1946) for that
//! combination of mean squares.
//!
//! A level that cannot be separated is reported, not invented: with one box `var_box` is `None`
//! (the SE reflects runs only, which is what "measured on one box" means); with one observation
//! per run the run and residual components are one, reported as `var_resid` with `var_run` `None`
//! and `(Run, Sample)` in [`VarComp::merged`] (also for a single run, whose run effect is not
//! estimable at all); with one run per box the run component merges into
//! the box component the same way.
//!
//! Owner: metrale-stats.
//! Invariants: at least two observations; components ≥ 0.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use metrale_error::{ErrorCode, error};

use crate::Result;
use crate::interval::{CiMethod, Interval, Level, Summary};
use crate::num::{f, mean, var};

/// 2026-10-22: The fitted components and the grand mean's standard error.
#[derive(Clone, Debug, PartialEq)]
pub struct VarComp {
    pub mean: f64,
    pub var_box: Option<f64>,
    pub var_run: Option<f64>,
    pub var_resid: f64,
    pub se_mean: f64,
    pub df: f64,
    pub n_box: usize,
    pub n_run: usize,
    pub n_obs: usize,
    /// 2026-10-22: Components whose moment estimate was negative and was set to 0.
    pub truncated: Vec<Level>,
    /// 2026-10-22: `(level, into)`: a level not separable from the one below it.
    pub merged: Vec<(Level, Level)>,
}

impl VarComp {
    /// 2026-10-22: The t interval of the grand mean at Satterthwaite df.
    #[must_use]
    pub fn interval(&self, level: f64) -> Interval {
        self.summary()
            .interval(level, CiMethod::VarCompSatterthwaite)
    }

    #[must_use]
    pub fn summary(&self) -> Summary {
        Summary {
            mean: self.mean,
            se: self.se_mean,
            df: self.df,
        }
    }
}

/// 2026-10-22: A one-way random model over `groups`: (σ²_group, σ²_within, μ̂ = mean of group
/// means, Var(μ̂), Satterthwaite df, whether σ²_group was truncated).
fn one_way(groups: &[Vec<f64>]) -> (f64, f64, f64, f64, f64, bool) {
    let g = groups.len();
    let n: usize = groups.iter().map(Vec::len).sum();
    let all: Vec<f64> = groups.iter().flatten().copied().collect();
    let grand = mean(&all);
    let means: Vec<f64> = groups.iter().map(|v| mean(v)).collect();
    let ss_g: f64 = groups
        .iter()
        .zip(&means)
        .map(|(v, m)| f(v.len()) * (m - grand) * (m - grand))
        .sum();
    let ss_w: f64 = groups
        .iter()
        .zip(&means)
        .flat_map(|(v, m)| v.iter().map(move |x| (x - m) * (x - m)))
        .sum();
    let (df_g, df_w) = (f(g - 1), f(n - g));
    let ms_g = ss_g / df_g;
    let ms_w = if df_w > 0.0 { ss_w / df_w } else { 0.0 };
    let sum_n2: f64 = groups.iter().map(|v| f(v.len()) * f(v.len())).sum();
    let k0 = (f(n) - sum_n2 / f(n)) / df_g;
    let raw = (ms_g - ms_w) / k0;
    let truncated = raw < 0.0;
    let s2g = raw.max(0.0);
    // 2026-10-22: Var(mean of group means) = σ²_g/G + σ²_w·Σ(1/n_i)/G².
    let ca = 1.0 / f(g);
    let cw: f64 = groups.iter().map(|v| 1.0 / f(v.len())).sum::<f64>() / (f(g) * f(g));
    // 2026-10-22: As a combination of mean squares: c_g·MS_g + c_w·MS_w.
    let (c_g, c_w) = if truncated {
        (0.0, cw)
    } else {
        (ca / k0, cw - ca / k0)
    };
    let v = (c_g * ms_g + c_w * ms_w).max(0.0);
    let mut den = 0.0;
    if df_g > 0.0 {
        den += (c_g * ms_g) * (c_g * ms_g) / df_g;
    }
    if df_w > 0.0 {
        den += (c_w * ms_w) * (c_w * ms_w) / df_w;
    }
    let df = if den > 0.0 {
        (v * v / den).max(1.0)
    } else {
        df_g.max(1.0)
    };
    (s2g, ms_w, mean(&means), v, df, truncated)
}

/// 2026-10-22: Fit the nested model to `(box, run, value)` observations (run ids are local to
/// their box).
pub fn nested(obs: &[(u32, u32, f64)]) -> Result<VarComp> {
    if obs.len() < 2 {
        return Err(error!(
            ErrorCode::StatsInsufficient,
            "varcomp::nested",
            2u32,
            obs.len()
        ));
    }
    if obs.iter().any(|o| !o.2.is_finite()) {
        return Err(error!(
            ErrorCode::StatsInvalid,
            "varcomp::nested", "a value is not finite"
        ));
    }
    let mut tree: BTreeMap<u32, BTreeMap<u32, Vec<f64>>> = BTreeMap::new();
    for &(b, r, v) in obs {
        tree.entry(b).or_default().entry(r).or_default().push(v);
    }
    let n_box = tree.len();
    let n_run: usize = tree.values().map(BTreeMap::len).sum();
    let n_obs = obs.len();
    let mut out = VarComp {
        mean: 0.0,
        var_box: None,
        var_run: None,
        var_resid: 0.0,
        se_mean: 0.0,
        df: 1.0,
        n_box,
        n_run,
        n_obs,
        truncated: Vec::new(),
        merged: Vec::new(),
    };
    let runs: Vec<Vec<f64>> = tree.values().flat_map(|b| b.values().cloned()).collect();
    if n_box == 1 && (n_run == n_obs || n_run == 1) {
        let all: Vec<f64> = runs.into_iter().flatten().collect();
        out.mean = mean(&all);
        out.var_resid = var(&all);
        out.se_mean = libm::sqrt(out.var_resid / f(n_obs));
        out.df = f(n_obs - 1);
        out.merged.push((Level::Run, Level::Sample));
        return Ok(out);
    }
    if n_box == 1 {
        let (s2r, s2e, m, v, df, t) = one_way(&runs);
        (out.mean, out.var_run, out.var_resid, out.se_mean, out.df) =
            (m, Some(s2r), s2e, libm::sqrt(v), df);
        if t {
            out.truncated.push(Level::Run);
        }
        return Ok(out);
    }
    if n_run == n_obs || n_run == n_box {
        let boxes: Vec<Vec<f64>> = tree
            .values()
            .map(|b| b.values().flatten().copied().collect())
            .collect();
        let (s2b, s2e, m, v, df, t) = one_way(&boxes);
        (out.mean, out.var_box, out.var_resid, out.se_mean, out.df) =
            (m, Some(s2b), s2e, libm::sqrt(v), df);
        out.merged.push(if n_run == n_obs {
            (Level::Run, Level::Sample)
        } else {
            (Level::Run, Level::Box)
        });
        if t {
            out.truncated.push(Level::Box);
        }
        return Ok(out);
    }
    three_level(&tree, out)
}

fn three_level(tree: &BTreeMap<u32, BTreeMap<u32, Vec<f64>>>, mut out: VarComp) -> Result<VarComp> {
    let (a, b, n) = (f(out.n_box), f(out.n_run), f(out.n_obs));
    let all: Vec<f64> = tree
        .values()
        .flat_map(|bx| bx.values().flatten().copied())
        .collect();
    let grand = mean(&all);
    let (mut ss_a, mut ss_b, mut ss_e) = (0.0, 0.0, 0.0);
    let (mut sum_ni2_over_ni, mut sum_nij2, mut sum_ni2) = (0.0, 0.0, 0.0);
    for bx in tree.values() {
        let vals: Vec<f64> = bx.values().flatten().copied().collect();
        let ni = f(vals.len());
        let mi = mean(&vals);
        ss_a += ni * (mi - grand) * (mi - grand);
        let mut sq = 0.0;
        for run in bx.values() {
            let nij = f(run.len());
            let mij = mean(run);
            ss_b += nij * (mij - mi) * (mij - mi);
            ss_e += run.iter().map(|x| (x - mij) * (x - mij)).sum::<f64>();
            sq += nij * nij;
        }
        sum_ni2_over_ni += sq / ni;
        sum_nij2 += sq;
        sum_ni2 += ni * ni;
    }
    let (df_a, df_b, df_e) = (a - 1.0, b - a, n - b);
    let ms = [ss_a / df_a, ss_b / df_b, ss_e / df_e];
    let dfs = [df_a, df_b, df_e];
    let k1 = (n - sum_ni2_over_ni) / df_b;
    let k2 = (sum_ni2_over_ni - sum_nij2 / n) / df_a;
    let k3 = (n - sum_ni2 / n) / df_a;
    // 2026-10-22: Each component as a combination of (MS_A, MS_B, MS_E).
    let ce = [0.0, 0.0, 1.0];
    let mut cb = [0.0, 1.0 / k1, -1.0 / k1];
    if dot(&cb, &ms) < 0.0 {
        cb = [0.0; 3];
        out.truncated.push(Level::Run);
    }
    let mut ca = [1.0 / k3, -k2 * cb[1] / k3, (-1.0 - k2 * cb[2]) / k3];
    if dot(&ca, &ms) < 0.0 {
        ca = [0.0; 3];
        out.truncated.push(Level::Box);
    }
    // 2026-10-22: Var(mean of box means of run means) = Ca·σ²_a + Cb·σ²_b + Ce·σ²_e.
    let wa = 1.0 / a;
    let mut wb = 0.0;
    let mut we = 0.0;
    let mut grand_mean = 0.0;
    for bx in tree.values() {
        let ri = f(bx.len());
        wb += 1.0 / ri;
        we += bx.values().map(|r| 1.0 / f(r.len())).sum::<f64>() / (ri * ri);
        grand_mean += bx.values().map(|r| mean(r)).sum::<f64>() / ri;
    }
    let (wb, we) = (wb / (a * a), we / (a * a));
    let c: Vec<f64> = (0..3)
        .map(|k| wa * ca[k] + wb * cb[k] + we * ce[k])
        .collect();
    let v = dot(&c, &ms).max(0.0);
    let den: f64 = (0..3)
        .map(|k| (c[k] * ms[k]) * (c[k] * ms[k]) / dfs[k])
        .sum();
    out.mean = grand_mean / a;
    out.var_box = Some(dot(&ca, &ms).max(0.0));
    out.var_run = Some(dot(&cb, &ms).max(0.0));
    out.var_resid = ms[2];
    out.se_mean = libm::sqrt(v);
    out.df = if den > 0.0 {
        (v * v / den).max(1.0)
    } else {
        df_a.max(1.0)
    };
    Ok(out)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
#[path = "varcomp_tests.rs"]
mod tests;
