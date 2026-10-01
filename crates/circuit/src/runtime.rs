// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: Runtime routes. A runtime route is a condition the engine checks at run time,
//! not a policy setting. When it holds, a step leaves the arm its plan names and runs another.
//! A FUSIONS.toml declares each route (`[[runtime]]`) with the settings its arm plans as. A
//! plan is rendered, estimated and executed with the arm of every route that applies beside it.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - A route applies to a run when its `modes` and `rows` contain the run and the policy states
//!   every `when` setting at that value. Its arm is fused under the policy with `plans_as`
//!   substituted, and is kept only when its digest differs from the primary plan's.
//! - `plans_as` changes only settings that `when` fixes, each to another value. A route
//!   re-plans an arm the policy selected; it never adds a setting.
//! - Route ids are unique and never a rule's id.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

use crate::fuser::{AvailableKernels, FuseError, FusionPlan, Policy, fuse};
use crate::ir::Circuit;
use crate::rules::{Mode, Rule, RuleError, parse_file};

/// 2026-09-30: One `[[runtime]]` entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeRoute {
    /// 2026-09-30: Id, unique in its rule set.
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

/// 2026-09-30: A FUSIONS.toml's rules and routes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleSet {
    /// 2026-09-30: The rules, in file order.
    pub rules: Vec<Rule>,
    /// 2026-09-30: The routes, in file order.
    pub runtime: Vec<RuntimeRoute>,
}

/// 2026-09-30: `[[runtime]]` as written.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RouteFile {
    pub(crate) id: String,
    when: BTreeMap<String, String>,
    modes: Vec<String>,
    rows: [u64; 2],
    plans_as: BTreeMap<String, String>,
    why: String,
    cite: String,
}

/// 2026-09-30: Parse FUSIONS.toml text into its rules and routes.
pub fn parse_rule_set(text: &str) -> Result<RuleSet, RuleError> {
    let (rules, files) = parse_file(text)?;
    let mut seen = BTreeSet::new();
    let mut runtime = Vec::with_capacity(files.len());
    for f in files {
        if !seen.insert(f.id.clone()) {
            return Err(bad(&f.id, "duplicate id".into()));
        }
        runtime.push(route(f)?);
    }
    check_ids(&rules, &runtime)?;
    Ok(RuleSet { rules, runtime })
}

/// 2026-09-30: No route shares an id with a rule.
pub fn check_ids(rules: &[Rule], runtime: &[RuntimeRoute]) -> Result<(), RuleError> {
    match runtime.iter().find(|r| rules.iter().any(|x| x.id == r.id)) {
        Some(r) => Err(bad(&r.id, "shares its id with a rule".into())),
        None => Ok(()),
    }
}

fn bad(id: &str, detail: String) -> RuleError {
    RuleError::Runtime {
        route: id.to_string(),
        detail,
    }
}

fn route(f: RouteFile) -> Result<RuntimeRoute, RuleError> {
    if f.plans_as.is_empty() {
        return Err(bad(&f.id, "`plans_as` is empty".into()));
    }
    for (k, v) in &f.plans_as {
        match f.when.get(k) {
            Some(w) if w != v => {}
            Some(_) => {
                return Err(bad(
                    &f.id,
                    format!("`plans_as` keeps `{k}` at its `when` value"),
                ));
            }
            None => {
                return Err(bad(
                    &f.id,
                    format!("`plans_as` sets `{k}`, which `when` does not fix"),
                ));
            }
        }
    }
    let modes = f
        .modes
        .iter()
        .map(|m| Mode::parse(m).ok_or_else(|| bad(&f.id, format!("unknown mode `{m}`"))))
        .collect::<Result<BTreeSet<Mode>, RuleError>>()?;
    if modes.is_empty() {
        return Err(bad(&f.id, "no modes".into()));
    }
    if f.rows[0] == 0 || f.rows[0] > f.rows[1] {
        return Err(bad(&f.id, format!("rows {:?}", f.rows)));
    }
    if f.why.trim().is_empty() || f.cite.trim().is_empty() {
        return Err(bad(&f.id, "`why` and `cite` must be stated".into()));
    }
    Ok(RuntimeRoute {
        id: f.id,
        when: f.when,
        modes,
        rows: (f.rows[0], f.rows[1]),
        plans_as: f.plans_as,
        why: f.why,
        cite: f.cite,
    })
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
    /// 2026-09-30: The engine makes this check on `mode` at `rows` under `policy`.
    pub fn applies(&self, policy: &Policy, mode: Mode, rows: u64) -> bool {
        self.modes.contains(&mode)
            && (self.rows.0..=self.rows.1).contains(&rows)
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

/// 2026-09-30: The arm of each route in `runtime` that applies to `mode` at `rows` and plans
/// differently from `primary`, fused with `rules` and `available` as the primary was.
pub fn route_arms(
    circuit: &Circuit,
    set: (&[Rule], &[RuntimeRoute]),
    available: &AvailableKernels,
    policy: &Policy,
    primary: &FusionPlan,
) -> Result<Vec<(RuntimeRoute, FusionPlan)>, FuseError> {
    let (rules, runtime) = set;
    let (mode, rows) = (primary.mode, primary.rows);
    let mut out = Vec::new();
    for r in runtime.iter().filter(|r| r.applies(policy, mode, rows)) {
        let arm = fuse(circuit, rules, available, &r.policy(policy), mode, rows)?;
        if arm.digest != primary.digest {
            out.push((r.clone(), arm));
        }
    }
    Ok(out)
}
