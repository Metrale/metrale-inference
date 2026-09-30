// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: Runtime routes. A runtime route is a condition the engine checks at run time,
//! not a policy setting. When it holds, a step leaves the arm its plan names and runs another.
//! A class's FUSIONS.toml declares each route (`[[runtime]]`) with the settings it plans as. The
//! planner fuses the primary plan and, where a route applies, the route's plan beside it, so a
//! plan and its report show both arms.
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - A route applies to a run when its `modes` and `rows` contain the run and the policy states
//!   every `when` setting at that value. Its plan is fused under the policy with `plans_as`
//!   substituted, and is kept only when its digest differs from the primary plan's.
//! - `plans_as` changes only settings that `when` fixes, each to another value. A route
//!   re-plans an arm the policy selected; it never adds a setting.
//! - The executor's rule parser (`rules::parse_rules`) refuses the `[[runtime]]` table. A class
//!   the executor serves therefore declares no route until the executor runs both arms.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

use super::HwError;
use super::plan::Planned;
use crate::fuser::Policy;
use crate::rules::Mode;
use crate::venn::Run;

/// 2026-09-30: One `[[runtime]]` entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeRoute {
    /// 2026-09-30: Id, unique in the class's resolved rule set.
    pub id: String,
    /// 2026-09-30: The policy settings under which the engine checks the condition.
    pub when: BTreeMap<String, String>,
    /// 2026-09-30: Modes the check runs in.
    pub modes: BTreeSet<Mode>,
    /// 2026-09-30: Padded rows the check runs at, inclusive.
    pub rows: (u64, u64),
    /// 2026-09-30: The settings the route's arm is planned under.
    pub plans_as: BTreeMap<String, String>,
    /// 2026-09-30: When the engine takes the route, in words.
    pub why: String,
    /// 2026-09-30: Where the engine makes the check.
    pub cite: String,
}

/// 2026-09-30: A route that applies to a run, with its plan.
#[derive(Debug, Clone)]
pub struct RoutePlan {
    /// 2026-09-30: The route.
    pub route: RuntimeRoute,
    /// 2026-09-30: The route's plan (its own `routes` are empty).
    pub planned: Planned,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RouteFile {
    id: String,
    when: BTreeMap<String, String>,
    modes: Vec<String>,
    rows: [u64; 2],
    plans_as: BTreeMap<String, String>,
    why: String,
    cite: String,
}

/// 2026-09-30: The `[[runtime]]` entries of the FUSIONS.toml at `rel` (`None`: the file declares
/// none).
pub fn parse_routes(value: Option<toml::Value>, rel: &str) -> Result<Vec<RuntimeRoute>, HwError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let files: Vec<RouteFile> = value
        .try_into()
        .map_err(|e: toml::de::Error| HwError::Class(format!("{rel}: [[runtime]]: {e}")))?;
    let mut seen = BTreeSet::new();
    let mut out = Vec::with_capacity(files.len());
    for f in files {
        let bad = |detail: String| HwError::Class(format!("{rel}: runtime `{}`: {detail}", f.id));
        if !seen.insert(f.id.clone()) {
            return Err(bad("duplicate id".into()));
        }
        if f.plans_as.is_empty() {
            return Err(bad("`plans_as` is empty".into()));
        }
        for (k, v) in &f.plans_as {
            match f.when.get(k) {
                Some(w) if w != v => {}
                Some(_) => return Err(bad(format!("`plans_as` keeps `{k}` at its `when` value"))),
                None => {
                    return Err(bad(format!(
                        "`plans_as` sets `{k}`, which `when` does not fix"
                    )));
                }
            }
        }
        let modes = f
            .modes
            .iter()
            .map(|m| Mode::parse(m).ok_or_else(|| bad(format!("unknown mode `{m}`"))))
            .collect::<Result<BTreeSet<Mode>, HwError>>()?;
        if modes.is_empty() {
            return Err(bad("no modes".into()));
        }
        if f.rows[0] == 0 || f.rows[0] > f.rows[1] {
            return Err(bad(format!("rows {:?}", f.rows)));
        }
        if f.why.trim().is_empty() || f.cite.trim().is_empty() {
            return Err(bad("`why` and `cite` must be stated".into()));
        }
        out.push(RuntimeRoute {
            id: f.id,
            when: f.when,
            modes,
            rows: (f.rows[0], f.rows[1]),
            plans_as: f.plans_as,
            why: f.why,
            cite: f.cite,
        });
    }
    Ok(out)
}

/// 2026-09-30: `own` over `base`: a route of the same id replaces the base's, others are added.
pub fn overlay(base: &mut Vec<RuntimeRoute>, own: Vec<RuntimeRoute>) {
    for r in own {
        match base.iter_mut().find(|b| b.id == r.id) {
            Some(slot) => *slot = r,
            None => base.push(r),
        }
    }
}

impl RuntimeRoute {
    /// 2026-09-30: The engine makes this check on `run` under `policy`.
    pub fn applies(&self, policy: &Policy, run: Run) -> bool {
        self.modes.contains(&run.mode)
            && (self.rows.0..=self.rows.1).contains(&run.rows)
            && self
                .when
                .iter()
                .all(|(k, v)| policy.settings.get(k) == Some(v))
    }

    /// 2026-09-30: `policy` with `plans_as` substituted.
    pub fn policy(&self, policy: &Policy) -> Policy {
        let mut out = policy.clone();
        for (k, v) in &self.plans_as {
            out.settings.insert(k.clone(), v.clone());
        }
        out
    }

    /// 2026-09-30: `k=v` for each `plans_as` setting.
    pub fn plans_as_text(&self) -> String {
        let s: Vec<String> = self
            .plans_as
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        s.join(" ")
    }
}

#[cfg(test)]
#[path = "runtime_tests.rs"]
mod runtime_tests;
