// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The what-if spec (`whatif-spec/v1`: named rewrite sets, each with the serve
//! measurements it is backtested against), the backtest verdict per set, and the KV breakpoint
//! of a site between two analyses at different KV lengths.
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - A step change is `+` when the step gets slower (the ledger's tok/s losses are `+` here).
//! - The sign passes when the measurement's interval excludes 0 and the prediction has its
//!   sign, or when the interval includes 0 and the prediction lies inside it; the error passes
//!   at `|predicted - measured| <= 0.3%` of the measured step.
//! - The breakpoint is linear in KV between the two analyses (one capture each): a stated
//!   interpolation, never a fit.

use crate::prelude::*;

use metrale_error::{ErrorCode, error};
use serde::{Deserialize, Serialize};

use super::analyze::WhatIf;
use super::scenario::Rewrite;

/// 2026-10-23: One serve measurement of a rewrite set.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Measured {
    /// 2026-10-23: Which capture it applies to (`--tag`).
    pub tag: String,
    /// 2026-10-23: Step change, microseconds (+: slower).
    pub us: f64,
    /// 2026-10-23: Interval low, microseconds.
    pub lo: f64,
    /// 2026-10-23: Interval high.
    pub hi: f64,
    /// 2026-10-23: Where it comes from.
    pub basis: String,
}

/// 2026-10-23: A named rewrite set.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Set {
    /// 2026-10-23: Name.
    pub id: String,
    /// 2026-10-23: Ledger rows it is backtested against.
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// 2026-10-23: The rewrites, applied in order.
    pub rewrite: Vec<Rewrite>,
    /// 2026-10-23: Measurements.
    #[serde(default)]
    pub measured: Vec<Measured>,
}

/// 2026-10-23: The spec.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    /// 2026-10-23: `whatif-spec/v1`.
    pub schema: String,
    /// 2026-10-23: Rewrite sets.
    pub set: Vec<Set>,
}

/// 2026-10-23: Parse a spec.
pub fn parse_spec(text: &str) -> metrale_error::Result<Spec> {
    let s: Spec = toml::from_str(text)
        .map_err(|e| error!(ErrorCode::WhatIfInvalid, "spec", format!("{e}")))?;
    if s.schema != "whatif-spec/v1" {
        return Err(error!(
            ErrorCode::WhatIfInvalid,
            "spec",
            format!("schema `{}` (this build reads whatif-spec/v1)", s.schema)
        ));
    }
    for set in &s.set {
        if set.rewrite.is_empty() {
            return Err(error!(
                ErrorCode::WhatIfInvalid,
                set.id.clone(),
                "names no rewrite".to_string()
            ));
        }
        for m in &set.measured {
            if !(m.lo <= m.us && m.us <= m.hi) {
                return Err(error!(
                    ErrorCode::WhatIfInvalid,
                    set.id.clone(),
                    format!(
                        "measured {} outside its interval [{}, {}]",
                        m.us, m.lo, m.hi
                    )
                ));
            }
        }
    }
    Ok(s)
}

/// 2026-10-23: One backtest line.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Verdict {
    /// 2026-10-23: The set.
    pub id: String,
    /// 2026-10-23: Ledger rows.
    pub depends_on: Vec<String>,
    /// 2026-10-23: Simulated step change, microseconds.
    pub predicted_us: f64,
    /// 2026-10-23: Max-plus step change (no contention), microseconds.
    pub maxplus_us: f64,
    /// 2026-10-23: Measured, interval.
    pub measured_us: f64,
    /// 2026-10-23: Interval low.
    pub lo: f64,
    /// 2026-10-23: Interval high.
    pub hi: f64,
    /// 2026-10-23: `|predicted - measured|` as percent of the step.
    pub error_pct: f64,
    /// 2026-10-23: Sign rule.
    pub sign_ok: bool,
    /// 2026-10-23: Error within 0.3% of the step.
    pub error_ok: bool,
    /// 2026-10-23: The measurement's basis.
    pub basis: String,
}

/// 2026-10-23: The verdicts of `spec`'s sets measured under `tag`, from analysis `w`.
pub fn verdicts(spec: &Spec, w: &WhatIf, tag: &str) -> Vec<Verdict> {
    let mut out = Vec::new();
    for set in &spec.set {
        let (Some(m), Some(r)) = (
            set.measured.iter().find(|m| m.tag == tag),
            w.rewrites.iter().find(|r| r.id == set.id),
        ) else {
            continue;
        };
        let p = r.sim_delta_us;
        let sign_ok = if m.lo > 0.0 {
            p > 0.0
        } else if m.hi < 0.0 {
            p < 0.0
        } else {
            m.lo <= p && p <= m.hi
        };
        let error_pct = 100.0 * (p - m.us).abs() / w.measured_step_us;
        out.push(Verdict {
            id: set.id.clone(),
            depends_on: set.depends_on.clone(),
            predicted_us: p,
            maxplus_us: r.maxplus_delta_us,
            measured_us: m.us,
            lo: m.lo,
            hi: m.hi,
            error_pct,
            sign_ok,
            error_ok: error_pct <= 0.3,
            basis: m.basis.clone(),
        });
    }
    out
}

/// 2026-10-23: Where a site's critical-path time crosses another site's, linear in KV.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Crossing {
    /// 2026-10-23: The other site.
    pub other: String,
    /// 2026-10-23: KV tokens where the site's critical time equals the other's (`None`: the
    /// lines do not cross at a positive KV).
    pub kv: Option<f64>,
}

/// 2026-10-23: The breakpoint report of one site.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Breakpoint {
    /// 2026-10-23: `run · site`.
    pub site: String,
    /// 2026-10-23: `(kv, critical us/step, subgradient, min slack us)` at each analysis
    /// (`None`: unbounded slack).
    pub at: Vec<(f64, f64, f64, Option<f64>)>,
    /// 2026-10-23: KV where the site's critical time would reach zero slack if it had slack
    /// (`Some(kv)` only when the slack changes sign between the analyses); `None` when it is on
    /// the chain at both or off it at both.
    pub becomes_critical_kv: Option<f64>,
    /// 2026-10-23: Its rank by critical time at each analysis (1 = largest).
    pub rank: Vec<usize>,
    /// 2026-10-23: Crossings with the sites ranked above it at the lower KV.
    pub crossings: Vec<Crossing>,
}

fn key(s: &super::analyze::SiteCrit) -> String {
    format!("{} · {}", s.run, s.site)
}

/// 2026-10-23: The breakpoint of `site` (`run · site`) between analyses `a` and `b`.
pub fn breakpoint(a: &WhatIf, b: &WhatIf, site: &str) -> metrale_error::Result<Breakpoint> {
    let (lo, hi) = if a.kv_tokens <= b.kv_tokens {
        (a, b)
    } else {
        (b, a)
    };
    if (hi.kv_tokens - lo.kv_tokens).abs() < 1.0 {
        return Err(error!(
            ErrorCode::WhatIfInvalid,
            "breakpoint",
            "the two analyses attend the same KV length".to_string()
        ));
    }
    let find = |w: &WhatIf| w.sites.iter().find(|s| key(s) == site).cloned();
    let (Some(sl), Some(sh)) = (find(lo), find(hi)) else {
        return Err(error!(
            ErrorCode::WhatIfInvalid,
            "breakpoint",
            format!("site `{site}` is not in both analyses")
        ));
    };
    let rank = |w: &WhatIf, s: &str| {
        let mut v: Vec<&super::analyze::SiteCrit> = w.sites.iter().collect();
        v.sort_by(|x, y| y.critical_us.total_cmp(&x.critical_us));
        v.iter().position(|x| key(x) == s).map_or(0, |p| p + 1)
    };
    let line = |x0: f64, y0: f64, x1: f64, y1: f64| (y1 - y0) / (x1 - x0);
    let slope = line(lo.kv_tokens, sl.critical_us, hi.kv_tokens, sh.critical_us);
    let mut crossings = Vec::new();
    for o in &lo.sites {
        if o.critical_us <= sl.critical_us || key(o) == site {
            continue;
        }
        let Some(oh) = hi.sites.iter().find(|x| key(x) == key(o)) else {
            continue;
        };
        let so = line(lo.kv_tokens, o.critical_us, hi.kv_tokens, oh.critical_us);
        let kv = (slope > so)
            .then(|| lo.kv_tokens + (o.critical_us - sl.critical_us) / (slope - so))
            .filter(|k| *k > 0.0);
        crossings.push(Crossing { other: key(o), kv });
    }
    crossings.sort_by(|x, y| {
        x.kv.unwrap_or(f64::INFINITY)
            .total_cmp(&y.kv.unwrap_or(f64::INFINITY))
    });
    let crit = |s: &super::analyze::SiteCrit| s.slack_min_us.is_some_and(|x| x <= 1e-6);
    let becomes = match (sl.slack_min_us, sh.slack_min_us) {
        (Some(a), Some(b)) if crit(&sl) != crit(&sh) => {
            Some(lo.kv_tokens + a / (a - b) * (hi.kv_tokens - lo.kv_tokens))
        }
        _ => None,
    };
    Ok(Breakpoint {
        site: site.to_string(),
        at: [(lo, &sl), (hi, &sh)]
            .iter()
            .map(|(w, s)| (w.kv_tokens, s.critical_us, s.subgradient, s.slack_min_us))
            .collect(),
        becomes_critical_kv: becomes,
        rank: vec![rank(lo, site), rank(hi, site)],
        crossings,
    })
}
