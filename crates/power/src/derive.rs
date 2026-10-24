// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: A trace to its measurement row: the whole of the steady-state derivation, from the
//! trace alone (HPAR: `met bench kernel-energy --recompute <trace>` reproduces a row bitwise).
//!
//! Required trace metadata (PCND, no defaults): `platform`, `scope`, `policy`, `seed`, `order`,
//! `ci_level`, `bootstrap_reps`, `fingerprint`, `energy_source`.
//!
//! The estimate of record per quantity (J/launch total and dynamic, W): the mean of the repeats.
//! Its interval is the envelope of (a) the variance-components interval (samples within runs
//! within boxes, Satterthwaite df; between-run and between-box variance, not just n) and (b) the
//! hierarchical BCa bootstrap (`boot`), so neither the run structure nor the autocorrelation
//! inside a window is ignored. Every present source is stored; the platform's declared order
//! picks the headline; each pair of sources gets its signed energy delta, a Deming fit and
//! Bland-Altman limits over the windows; disagreement is flagged, never averaged away.
//!
//! Owner: metrale-power.
//! Invariants: a refusal of the headline channel refuses the row; a non-headline channel that
//! fails a rule is dropped with a flag naming the rule.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use metrale_abstract_io::power::SourceKind;
use metrale_error::{ErrorCode, error};
use metrale_stats::interval::Interval;

use crate::Result;
use crate::boot::{RepeatSeries, bca};
use crate::derive_pairs::{agreements, sensor_deltas};
use crate::integrate::{Channel, channels};
use crate::platform::{parse_scope, platform};
use crate::policy::WindowPolicy;
use crate::row::{Ci, EnergyRow, Label};
use crate::steady::{ChannelRepeat, LoadFacts, load_facts, measure, repeats};
use crate::trace::PowerTrace;

/// 2026-10-22: The row and what led to it.
#[derive(Clone, Debug, PartialEq)]
pub struct Derived {
    pub row: EnergyRow,
    /// 2026-10-22: Per channel (headline first), its repeats.
    pub channels: Vec<(Channel, Vec<ChannelRepeat>)>,
    /// 2026-10-22: (W, J_total, J_dyn): the variance-components interval and the BCa interval.
    pub varcomp: [Interval; 3],
    pub bca: [Interval; 3],
}

pub(crate) fn meta<'a>(t: &'a PowerTrace, k: &str) -> Result<&'a str> {
    t.meta.get(k).map(String::as_str).ok_or_else(|| {
        error!(
            ErrorCode::EnergyInputInvalid,
            format!("trace meta lacks `{k}`")
        )
    })
}

fn meta_num<T: core::str::FromStr>(t: &PowerTrace, k: &str) -> Result<T> {
    meta(t, k)?.parse().map_err(|_| {
        error!(
            ErrorCode::EnergyInputInvalid,
            format!("trace meta `{k}` is not a number")
        )
    })
}

/// 2026-10-22: The key a channel's reading is stored under: its source kind.
pub(crate) fn key(t: &PowerTrace, c: &Channel) -> &'static str {
    c.kind(t).as_str()
}

/// 2026-10-22: Derive the row of `t`, whose text hashes to `trace_sha256`.
pub fn derive(t: &PowerTrace, trace_sha256: &str) -> Result<Derived> {
    let policy = WindowPolicy::parse(meta(t, "policy")?)?;
    policy.check()?;
    let plat = platform(meta(t, "platform")?)?;
    let scope = parse_scope(meta(t, "scope")?)?;
    let level: f64 = meta_num(t, "ci_level")?;
    let reps: usize = meta_num(t, "bootstrap_reps")?;
    let seed: u64 = meta_num(t, "seed")?;
    let (pref, rails) = plat.scope(scope);
    let mut chans: Vec<Channel> = channels(t)
        .into_iter()
        .filter(|c| rails.contains(&c.rail))
        .collect();
    let present: Vec<SourceKind> = chans.iter().map(|c| c.kind(t)).collect();
    let (_, headline_kind) = pref.resolve(&present)?;
    let hi = chans
        .iter()
        .position(|c| c.kind(t) == headline_kind)
        .unwrap_or(0);
    let headline = chans.remove(hi);
    chans.insert(0, headline);
    let mut seen = Vec::new();
    chans.retain(|c| {
        let k = c.kind(t);
        let fresh = !seen.contains(&k);
        seen.push(k);
        fresh
    });

    let reps_all = repeats(t);
    if reps_all.is_empty() {
        return Err(error!(
            ErrorCode::EnergyInputInvalid,
            "the trace has no load window".to_string()
        ));
    }
    let mut facts: Vec<LoadFacts> = Vec::new();
    for r in &reps_all {
        facts.push(load_facts(t, &policy, &t.windows[r.load])?);
    }
    let mut flags: Vec<String> = Vec::new();
    let mut measured: Vec<(Channel, Vec<ChannelRepeat>)> = Vec::new();
    for (ci, c) in chans.iter().enumerate() {
        let one = |r| -> Result<ChannelRepeat> {
            let m = measure(t, &policy, *c, r)?;
            let drift = m.drift_pct()?;
            if drift > policy.max_baseline_drift_pct {
                return Err(error!(
                    ErrorCode::EnergyBaselineDrift,
                    m.repeat, drift, policy.max_baseline_drift_pct
                ));
            }
            m.j_dyn()?;
            Ok(m)
        };
        match reps_all.iter().map(one).collect::<Result<Vec<_>>>() {
            Ok(v) => measured.push((*c, v)),
            Err(e) if ci == 0 => return Err(e),
            Err(e) => flags.push(format!("dropped:{}:{}", key(t, c), e.code().name())),
        }
    }
    let (head, head_reps) = (measured[0].0, &measured[0].1);
    let w: Vec<f64> = head_reps.iter().map(|r| r.load.avg_w).collect();
    let jt: Vec<f64> = head_reps
        .iter()
        .map(|r| r.j_total())
        .collect::<Result<_>>()?;
    let jd: Vec<f64> = head_reps.iter().map(|r| r.j_dyn()).collect::<Result<_>>()?;
    let st: Vec<f64> = head_reps
        .iter()
        .map(|r| r.static_w())
        .collect::<Result<_>>()?;
    let ids: Vec<(u32, u32)> = head_reps.iter().map(|r| (r.box_id, r.repeat)).collect();
    let varcomp = [
        vc_interval(&ids, &w, level, &series_of(head_reps))?,
        vc_interval(&ids, &jt, level, &series_of(head_reps))?,
        vc_interval(&ids, &jd, level, &series_of(head_reps))?,
    ];
    let series: Vec<RepeatSeries<'_>> = head_reps
        .iter()
        .map(|r| RepeatSeries {
            box_id: r.box_id,
            load: &r.load,
            idle_a: r.idle_a.as_ref().expect("static_w checked both baselines"),
            idle_b: r.idle_b.as_ref().expect("static_w checked both baselines"),
            launches: r.launches.expect("j_total checked the launches") as f64,
        })
        .collect();
    let bca_ci = bca(&series, reps, level, seed)?;
    let env = |k: usize| Ci {
        lo: varcomp[k].lo.min(bca_ci[k].lo),
        hi: varcomp[k].hi.max(bca_ci[k].hi),
    };
    for (k, name) in ["w", "j_total", "j_dyn"].iter().enumerate() {
        if !metrale_stats::ratio::agree(&varcomp[k], &bca_ci[k], 0.5) {
            flags.push(format!("ci_methods_disagree:{name}"));
        }
    }
    let (deltas, delta_flags) = sensor_deltas(t, &measured, &policy);
    flags.extend(delta_flags);
    let agreement = agreements(t, &measured)?;
    for (r, rep) in head_reps.iter().enumerate() {
        let k = metrale_stats::changepoint::steady_start(&[&rep.load.series_w]);
        if k > 0 {
            flags.push(format!(
                "warmup_detected:window={}:updates={k}:untrimmed",
                reps_all[r].load
            ));
        }
    }
    let mut by = BTreeMap::new();
    let mut by_dyn = BTreeMap::new();
    for (c, v) in &measured {
        let jt: Vec<f64> = v.iter().map(|r| r.j_total()).collect::<Result<_>>()?;
        let jd: Vec<f64> = v.iter().map(|r| r.j_dyn()).collect::<Result<_>>()?;
        by.insert(key(t, c).to_string(), mean(&jt));
        by_dyn.insert(key(t, c).to_string(), mean(&jd));
    }
    let ess = head_reps
        .iter()
        .map(|r| metrale_stats::hac::ess(&r.load.series_w))
        .collect::<metrale_stats::Result<Vec<f64>>>()?
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    let outliers = if jt.len() >= 3 {
        metrale_stats::robust::flag_outliers(&jt, &metrale_stats::robust::OutlierRule::DECLARED)
            .len()
    } else {
        0
    };
    let mut boxes: Vec<u32> = ids.iter().map(|i| i.0).collect();
    boxes.sort_unstable();
    boxes.dedup();
    let n = head_reps.len() as u64;
    let opt_mean = |f: fn(&LoadFacts) -> Option<f64>| {
        let v: Vec<f64> = facts.iter().filter_map(f).collect();
        (!v.is_empty()).then(|| mean(&v))
    };
    let opt_max = |f: fn(&LoadFacts) -> Option<f64>| {
        facts
            .iter()
            .filter_map(f)
            .fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v))))
    };
    let row = EnergyRow {
        energy_j: mean(&jt),
        energy_dyn_j: mean(&jd),
        avg_w: mean(&w),
        static_w: mean(&st),
        energy_ci_j: env(1),
        energy_dyn_ci_j: env(2),
        avg_w_ci: env(0),
        energy_ci_level: level,
        energy_ci_method: "envelope(varcomp-satterthwaite, bca-hierarchical-stationary-bootstrap)"
            .to_string(),
        energy_method: "steady_state",
        energy_samples: n,
        energy_boxes: boxes.len() as u64,
        energy_launches: head_reps.iter().filter_map(|r| r.launches).sum(),
        energy_window_ms: head_reps.iter().map(|r| r.load.seconds * 1e3).sum(),
        energy_ess: ess,
        energy_outliers: outliers as u64,
        energy_seed: seed,
        energy_order: meta(t, "order")?.to_string(),
        energy_label: Label::of(n, boxes.len() as u64),
        sm_mhz: opt_mean(|f| f.sm_mhz),
        temp_c: opt_max(|f| f.temp_c),
        throttle_pct: opt_max(|f| f.throttle_pct),
        power_source: head.kind(t),
        power_rail: head.rail,
        energy_by_source: by,
        energy_dyn_by_source: by_dyn,
        sensor_delta_pct: deltas,
        energy_flags: flags,
        energy_agreement: agreement,
        energy_fingerprint: meta(t, "fingerprint")?.to_string(),
        energy_source: meta(t, "energy_source")?.to_string(),
        trace_sha256: trace_sha256.to_string(),
        energy_floor_class: None,
        energy_floor_j: None,
        pct_of_energy_floor: None,
    };
    Ok(Derived {
        row,
        channels: measured,
        varcomp,
        bca: bca_ci,
    })
}

pub(crate) fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

fn series_of(reps: &[ChannelRepeat]) -> Vec<Vec<f64>> {
    reps.iter().map(|r| r.load.series_w.clone()).collect()
}

/// 2026-10-22: The variance-components interval of the repeats' values; one repeat falls back to
/// the HAC t-interval of its load window, scaled from power to the quantity.
fn vc_interval(ids: &[(u32, u32)], v: &[f64], level: f64, series: &[Vec<f64>]) -> Result<Interval> {
    if v.len() >= 2 {
        let obs: Vec<(u32, u32, f64)> = ids.iter().zip(v).map(|(i, x)| (i.0, i.1, *x)).collect();
        return Ok(metrale_stats::varcomp::nested(&obs)?.interval(level));
    }
    let s = metrale_stats::hac::mean_summary(&series[0])?;
    let scale = if s.mean != 0.0 { v[0] / s.mean } else { 1.0 };
    let t = metrale_stats::dist::t_ppf(0.5 + level / 2.0, s.df.max(1.0));
    Ok(Interval {
        lo: v[0] - t * s.se * scale,
        hi: v[0] + t * s.se * scale,
        level,
        method: metrale_stats::interval::CiMethod::HacT,
    })
}
