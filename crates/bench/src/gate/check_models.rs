// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The gate check with a model dimension (`--gate-base`): a gate's records are judged
//! per registered model, a change re-opens only the models it reaches (`model_scope`), at least
//! one reached model must certify before merge, and the reached models left uncertified are owed.
//!
//! Owner: bench gate.
//! Invariants:
//! - [`decide`] is pure; the rest reads records, BENCH.toml and `git diff`.
//! - Every registered model's covering record is judged against its own BENCH.toml entry, and
//!   one that fails fails the gate (D4), whether or not this change reaches it.
//! - A declared subject (`default = true`) that this change reaches must itself certify, and some
//!   declared subject must have a covering record, so the default-subject rule of
//!   `check::check_one` still holds.
//! - When git cannot diff the base, every registered model counts as reached (fail-closed).
//! - `owns` lists are read at the merge base, never at the change: a change that widens a list
//!   narrows its own reach only after it merges.
//! - A model with no committed record of a gate that is not a declared subject is dropped from
//!   that gate: neither reached nor owed (R8). A declared subject is never dropped.
//! - A committed owed manifest that is settled at the commit, or unreadable, is reported in
//!   `manifests`; the gate check fails until the change deletes or fixes it (R8 pruning).
//! - A group gate (`check_group`) has no model dimension yet: its verdict is `check_group`'s
//!   and it owes nothing.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::Result;
use serde::Serialize;

use super::check::{
    GateStatus, Standing, judge, record_is_for, record_standing_for, records_newest_first,
};
use super::check_paths::invalidating_paths;
use super::coverage::{GateCoverage, PROMOTION_CANDIDATES, REQUIRED};
use super::model_scope::{KernelMap, Subject, reaching};
use super::owns::Owners;
use super::record::{read_baseline, read_record};

/// 2026-10-15: One registered model's standing at the commit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "state", content = "why", rename_all = "lowercase")]
pub enum SubjectState {
    /// 2026-10-15: Its newest covering record passes every check.
    Pass,
    /// 2026-10-15: Its newest covering record fails: one entry per problem.
    Fail(Vec<String>),
    /// 2026-10-15: A record of it is committed and none covers the commit; the text says why.
    Stale(String),
    /// 2026-10-16: No record of it is committed for this gate. A model that is not a declared
    /// subject is then dropped from the gate: never reached, never owed (owner ruling R8; a gate
    /// variant is measured before it is declared, `gate-variants-cannot-bootstrap`).
    Unrecorded,
}

/// 2026-10-15: A (gate, model) a merge leaves uncertified. The merge sha is the caller's.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct Owed {
    pub gate: String,
    pub subject: Subject,
}

/// 2026-10-15: One registered model's standing on one gate, and whether the change reaches it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelStanding {
    pub subject: Subject,
    pub reached: bool,
    pub state: SubjectState,
}

/// 2026-10-15: Every required gate's verdict, every model's standing on it, and every owed
/// (gate, model).
#[derive(Debug)]
pub struct ModelCheck {
    pub statuses: BTreeMap<String, GateStatus>,
    pub owed: Vec<Owed>,
    pub standings: BTreeMap<String, Vec<ModelStanding>>,
    /// 2026-10-16: One line per committed owed manifest this change must delete (it is settled)
    /// or fix (it is unreadable); `owed_manifest` decides.
    pub manifests: Vec<String>,
}

fn name(s: &Subject) -> String {
    format!("{} on {}", s.checkpoint, s.hw)
}

/// 2026-10-15: One gate's verdict and the reached models it leaves owed, from each registered
/// model's state, the models the change reaches and the declared subjects.
pub fn decide(
    states: &BTreeMap<Subject, SubjectState>,
    reached: &BTreeSet<Subject>,
    defaults: &BTreeSet<Subject>,
) -> (GateStatus, Vec<Subject>) {
    let passes = |s: &Subject| matches!(states.get(s), Some(SubjectState::Pass));
    let why = |s: &Subject| match states.get(s) {
        Some(SubjectState::Stale(why)) => why.clone(),
        Some(SubjectState::Unrecorded) => "no record of it is committed".to_string(),
        _ => "not registered".to_string(),
    };
    // 2026-10-16: A model with no record of this gate that is not a declared subject is not on
    // the gate yet: it is neither reached nor owed.
    let dropped = |s: &Subject| {
        matches!(states.get(s), Some(SubjectState::Unrecorded)) && !defaults.contains(s)
    };
    let reached: BTreeSet<&Subject> = reached.iter().filter(|s| !dropped(s)).collect();
    let owed: Vec<Subject> = reached
        .iter()
        .filter(|s| !passes(s))
        .map(|s| (*s).clone())
        .collect();
    let failing: Vec<String> = states
        .iter()
        .filter_map(|(s, st)| match st {
            SubjectState::Fail(p) => Some(p.iter().map(move |p| format!("{}: {p}", name(s)))),
            _ => None,
        })
        .flatten()
        .collect();
    if !failing.is_empty() {
        return (GateStatus::Fail(failing), owed);
    }
    if let Some(d) = defaults.iter().find(|d| reached.contains(d) && !passes(d)) {
        return (
            GateStatus::Missing(format!(
                "this change reaches the declared subject {}, which must certify: {}",
                name(d),
                why(d)
            )),
            owed,
        );
    }
    if !defaults.iter().any(passes) {
        let list: Vec<String> = defaults
            .iter()
            .map(|d| format!("{} ({})", name(d), why(d)))
            .collect();
        return (
            GateStatus::Missing(format!(
                "no declared subject has a covering record: {}",
                if list.is_empty() {
                    "none is declared".to_string()
                } else {
                    list.join("; ")
                }
            )),
            owed,
        );
    }
    if !reached.is_empty() && !reached.iter().any(|s| passes(s)) {
        let list: Vec<String> = reached.iter().map(|s| name(s)).collect();
        return (
            GateStatus::Missing(format!(
                "this change reaches {} and none has a covering record; certify at least one",
                list.join(", ")
            )),
            owed,
        );
    }
    (GateStatus::Pass, owed)
}

/// 2026-10-15: A gate's measured BENCH.toml entries, and which are declared subjects.
pub fn subjects(root: &Path, gate: &str) -> Result<(Vec<Subject>, BTreeSet<Subject>)> {
    let mut all = Vec::new();
    let mut defaults = BTreeSet::new();
    for (target, entry) in super::bench::load_all(root)? {
        if entry.gate != gate || entry.status != "measured" {
            continue;
        }
        let s = Subject {
            hw: target.hardware.clone(),
            model: target.model.clone(),
            checkpoint: entry.checkpoint.clone(),
        };
        if entry.default {
            defaults.insert(s.clone());
        }
        all.push(s);
    }
    Ok((all, defaults))
}

/// 2026-10-15: `paths` filtered to those that reach `s`, in order.
fn reaching_one(paths: Vec<String>, s: &Subject, map: &KernelMap, owners: &Owners) -> Vec<String> {
    let keep = reaching(&paths, std::slice::from_ref(s), map, owners)
        .remove(s)
        .unwrap_or_default();
    paths.into_iter().filter(|p| keep.contains(p)).collect()
}

fn subject_state(
    root: &Path,
    sha: &str,
    gate: &GateCoverage,
    s: &Subject,
    map: &KernelMap,
    owners: &Owners,
) -> SubjectState {
    let baseline = match read_baseline(root, gate.id) {
        Ok(b) => b,
        Err(e) => return SubjectState::Stale(format!("baseline unreadable: {e:#}")),
    };
    let reach = |paths: Vec<String>| reaching_one(paths, s, map, owners);
    let mut newest: Option<Standing> = None;
    for path in records_newest_first(root, gate.id) {
        let Ok(record) = read_record(&path) else {
            continue;
        };
        if !record_is_for(&record, gate.id, &path)
            || record.target_model != s.checkpoint
            || record.hardware.gate_key() != s.hw
        {
            continue;
        }
        match record_standing_for(root, sha, &record, gate, &reach) {
            Standing::Stands => {
                return match judge(root, &record, &path, &baseline) {
                    GateStatus::Pass => SubjectState::Pass,
                    GateStatus::Fail(p) => SubjectState::Fail(p),
                    GateStatus::Missing(why) => SubjectState::Stale(why),
                };
            }
            other => {
                newest.get_or_insert(other);
            }
        }
    }
    let Some(newest) = newest else {
        return SubjectState::Unrecorded;
    };
    SubjectState::Stale(match newest {
        Standing::Invalidated(paths) => format!(
            "its newest record is invalidated by {}",
            super::check_fmt::summarize_paths(&paths)
        ),
        _ => "git cannot diff its newest record's commit against this one".to_string(),
    })
}

/// 2026-10-15: One gate's verdict and owed models for the change `base..sha`.
fn check_gate(
    root: &Path,
    sha: &str,
    base: Option<&str>,
    gate: &GateCoverage,
    map: &KernelMap,
    owners: &Owners,
) -> (GateStatus, Vec<Owed>, Vec<ModelStanding>) {
    if let Some(group) = super::group::find(gate.id) {
        return (
            super::check_group::check_group(root, group, sha),
            Vec::new(),
            Vec::new(),
        );
    }
    let (all, defaults) = match subjects(root, gate.id) {
        Ok(x) => x,
        Err(e) => {
            return (
                GateStatus::Missing(format!("BENCH.toml unreadable: {e:#}")),
                vec![],
                vec![],
            );
        }
    };
    let reached: BTreeSet<Subject> = match base.and_then(|b| invalidating_paths(root, sha, b, gate))
    {
        None => all.iter().cloned().collect(),
        Some(paths) => reaching(&paths, &all, map, owners)
            .into_iter()
            .filter(|(_, p)| !p.is_empty())
            .map(|(s, _)| s)
            .collect(),
    };
    let states: BTreeMap<Subject, SubjectState> = all
        .iter()
        .map(|s| (s.clone(), subject_state(root, sha, gate, s, map, owners)))
        .collect();
    let (status, owed) = decide(&states, &reached, &defaults);
    let owed = owed
        .into_iter()
        .map(|subject| Owed {
            gate: gate.id.to_string(),
            subject,
        })
        .collect();
    let standings = states
        .into_iter()
        .map(|(subject, state)| ModelStanding {
            reached: reached.contains(&subject),
            subject,
            state,
        })
        .collect();
    (status, owed, standings)
}

/// 2026-10-16: The committed owed manifests that are settled at this commit, or unreadable.
fn manifest_problems(root: &Path, standings: &BTreeMap<String, Vec<ModelStanding>>) -> Vec<String> {
    let dir = root.join(super::owed_manifest::DIR);
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
        Err(e) => return vec![format!("{}: unreadable: {e}", dir.display())],
    };
    let mut out = Vec::new();
    for entry in entries {
        let path = match entry {
            Ok(e) => e.path(),
            Err(e) => {
                out.push(format!("{}: unreadable entry: {e}", dir.display()));
                continue;
            }
        };
        let shown = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .display()
            .to_string();
        let parsed = std::fs::read_to_string(&path)
            .map_err(|e| e.to_string())
            .and_then(|t| super::owed_manifest::parse(&t));
        match parsed {
            Err(e) => out.push(format!("{shown}: unreadable owed manifest: {e}")),
            Ok(m) if super::owed_manifest::settled(&m, standings) => out.push(format!(
                "{shown}: every pair it lists is certified at this commit; delete it in this change"
            )),
            Ok(_) => {}
        }
    }
    out.sort();
    out
}

/// 2026-10-15: Every required gate's verdict for `sha` as a change from `base`, and what the
/// change leaves owed on the required gates and the promotion candidates.
pub fn check_models(root: &Path, sha: &str, base: &str, map: &KernelMap) -> ModelCheck {
    // 2026-10-15: No merge base reads as no base: every registered model is reached.
    let base = super::kernel_map::merge_base(root, base, sha).ok();
    let base = base.as_deref();
    let owners = base
        .map(|rev| {
            super::kernel_map::owners(&super::kernel_map::GitRepo {
                root: root.to_path_buf(),
                rev: rev.to_string(),
            })
        })
        .unwrap_or_default();
    let mut statuses = BTreeMap::new();
    let mut owed = Vec::new();
    let mut standings = BTreeMap::new();
    for gate in REQUIRED.iter() {
        let (status, o, s) = check_gate(root, sha, base, gate, map, &owners);
        statuses.insert(gate.id.to_string(), status);
        owed.extend(o);
        standings.insert(gate.id.to_string(), s);
    }
    for gate in PROMOTION_CANDIDATES {
        let (_, o, s) = check_gate(root, sha, base, gate, map, &owners);
        owed.extend(o);
        standings.insert(gate.id.to_string(), s);
    }
    owed.sort();
    let manifests = manifest_problems(root, &standings);
    ModelCheck {
        statuses,
        owed,
        standings,
        manifests,
    }
}

#[cfg(test)]
#[path = "check_models_tests.rs"]
mod check_models_tests;
