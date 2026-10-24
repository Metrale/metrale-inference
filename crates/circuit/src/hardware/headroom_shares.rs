// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The two shares of a step the fuser/executor roadmap moves
//! (blueprint/DESIGN-fuser-megakernel.md §0, §5 "metric"): **plan-describable**, the exclusive
//! time of the launch sites the plan names, and **plan-executed**, the part of it the circuit
//! executor issued from the plan in the captured serve. Per run (`mode n=rows`) of the step.
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - Executed comes only from the capture's stated execution (`[plan.execution]` of the trace):
//!   every run of the step must be stated, and a stated run or site the step does not plan is an
//!   error, never ignored.
//! - Executed ⊆ describable: a run executed `plan` counts only its named sites (`*`: all).
//! - Without a capture there is no step, so no share (`[]`), not a zero.

use crate::prelude::*;

use serde::Serialize;

use super::headroom_model::{HeadroomError, SiteRow};
use super::headroom_trace::{ExecutedBy, Trace};

/// 2026-10-23: The shares of one run of the step.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PlanShare {
    /// 2026-10-23: `mode n=rows`.
    pub run: String,
    /// 2026-10-23: Who issued the run's launches in the captured serve.
    pub by: ExecutedBy,
    /// 2026-10-23: The capture's reason.
    pub why: String,
    /// 2026-10-23: Σ exclusive time of the run's measured sites, us per step.
    pub describable_us: f64,
    /// 2026-10-23: `describable_us` over the step, percent.
    pub describable_pct: f64,
    /// 2026-10-23: Σ exclusive time of the sites the executor issued, us per step.
    pub executed_us: f64,
    /// 2026-10-23: `executed_us` over the step, percent.
    pub executed_pct: f64,
}

/// 2026-10-23: The shares of each run in `runs`, from the measured `rows` and the capture's
/// stated execution; `[]` without a capture.
pub fn plan_shares(
    rows: &[SiteRow],
    runs: &[String],
    trace: Option<&Trace>,
) -> Result<Vec<PlanShare>, HeadroomError> {
    let Some(t) = trace else {
        return Ok(Vec::new());
    };
    let err = |m: String| Err(HeadroomError::Trace(m));
    if let Some(extra) = t.plan.execution.keys().find(|r| !runs.contains(r)) {
        return err(format!(
            "the trace states the execution of `{extra}`, which this step does not plan (runs: {})",
            runs.join(", ")
        ));
    }
    let step = t.step_us();
    let pct = |us: f64| 100.0 * us / step;
    let mut out = Vec::with_capacity(runs.len());
    for run in runs {
        let Some(e) = t.plan.execution.get(run) else {
            return err(format!(
                "the trace does not state how the serve executed `{run}` ([plan.execution])"
            ));
        };
        let mine: Vec<&SiteRow> = rows.iter().filter(|r| &r.run == run).collect();
        let all = e.sites.iter().any(|s| s == "*");
        if let Some(s) = e
            .sites
            .iter()
            .find(|s| *s != "*" && !mine.iter().any(|r| &r.site == *s))
        {
            return err(format!(
                "the trace says the executor issued site `{s}` of `{run}`, which the plan does not \
                 have"
            ));
        }
        let excl = |r: &&SiteRow| r.measured.as_ref().map_or(0.0, |m| m.exclusive_us);
        let describable_us: f64 = mine.iter().map(excl).sum();
        let executed_us: f64 = match e.by {
            ExecutedBy::Legacy => 0.0,
            ExecutedBy::Plan => mine
                .iter()
                .filter(|r| all || e.sites.contains(&r.site))
                .map(excl)
                .sum(),
        };
        out.push(PlanShare {
            run: run.clone(),
            by: e.by,
            why: e.why.clone(),
            describable_us,
            describable_pct: pct(describable_us),
            executed_us,
            executed_pct: pct(executed_us),
        });
    }
    Ok(out)
}
