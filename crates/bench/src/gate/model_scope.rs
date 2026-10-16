// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Which of a gate's registered models a set of changed paths reaches: the model
//! dimension of coverage. `coverage::invalidates` decides whether a path re-opens a gate at all;
//! this decides which of the gate's models it re-opens.
//!
//! Owner: bench gate.
//! Invariants:
//! - Pure: the caller passes the paths, the registered subjects, the kernel map and the owners.
//! - A path outside `kernels/` that some BENCH.toml `owns` list holds (`owns`) reaches the
//!   subjects of the owning models alone; any other path outside `kernels/` reaches every
//!   subject (owner ruling Q1, 2026-10-15).
//! - A path under `kernels/` reaches the subjects `metrale_circuit`'s `affected` names for it.
//!   It reaches every subject when the map is unavailable or the path is unmapped, and every
//!   subject on its hardware that has no INSTANCES.toml entry, whose families are unknown
//!   (fail-closed).

use std::collections::{BTreeMap, BTreeSet};

use metrale_circuit::venn::affected::{InstanceUse, affected};
use metrale_circuit::venn::families::Families;
use serde::Serialize;

use super::owns::{Owners, owners_of};

/// 2026-10-15: One registered model of a gate: a measured BENCH.toml entry.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct Subject {
    /// 2026-10-15: Hardware directory, which is also the record's `gate_key`.
    pub hw: String,
    /// 2026-10-15: Model directory (`kernels/<hw>/<model>/`).
    pub model: String,
    /// 2026-10-15: Checkpoint id, the record's `target_model`.
    pub checkpoint: String,
}

/// 2026-10-15: The kernel → model map of the tree under test.
#[derive(Debug, Clone)]
pub enum KernelMap {
    /// 2026-10-15: Every hardware's family manifest and every instance's family usage.
    Derived {
        manifests: Vec<Families>,
        uses: Vec<InstanceUse>,
    },
    /// 2026-10-15: The map did not load; the text says why. Every path reaches every subject.
    Unavailable(String),
}

fn hw_of(path: &str) -> &str {
    path.strip_prefix("kernels/")
        .and_then(|r| r.split('/').next())
        .unwrap_or_default()
}

/// 2026-10-15: For each subject, the paths of `paths` that reach it (possibly none).
pub fn reaching(
    paths: &[String],
    subjects: &[Subject],
    map: &KernelMap,
    owners: &Owners,
) -> BTreeMap<Subject, BTreeSet<String>> {
    let mut out: BTreeMap<Subject, BTreeSet<String>> = subjects
        .iter()
        .map(|s| (s.clone(), BTreeSet::new()))
        .collect();
    let to_all = |p: &String, out: &mut BTreeMap<Subject, BTreeSet<String>>| {
        for set in out.values_mut() {
            set.insert(p.clone());
        }
    };
    let (kernel, other): (Vec<String>, Vec<String>) = paths
        .iter()
        .cloned()
        .partition(|p| p.starts_with("kernels/"));
    for p in &other {
        let by = owners_of(p, owners);
        if by.is_empty() {
            to_all(p, &mut out);
            continue;
        }
        for (s, set) in out.iter_mut() {
            if by.contains(s.model.as_str()) {
                set.insert(p.clone());
            }
        }
    }
    let (manifests, uses) = match map {
        KernelMap::Unavailable(_) => {
            for p in &kernel {
                to_all(p, &mut out);
            }
            return out;
        }
        KernelMap::Derived { manifests, uses } => (manifests, uses),
    };
    let a = affected(&kernel, manifests, uses);
    for p in &a.unmapped {
        to_all(p, &mut out);
    }
    for m in &a.models {
        for (s, set) in out.iter_mut() {
            if s.hw == m.hw && s.checkpoint == m.checkpoint {
                set.extend(m.reason.iter().map(|h| h.file.clone()));
            }
        }
    }
    let has_instance = |s: &Subject| {
        uses.iter().any(|u| {
            u.checkpoint == s.checkpoint && u.target.split('/').next() == Some(s.hw.as_str())
        })
    };
    for p in kernel.iter().filter(|p| !a.unmapped.contains(p)) {
        for (s, set) in out.iter_mut() {
            if s.hw == hw_of(p) && !has_instance(s) {
                set.insert(p.clone());
            }
        }
    }
    out
}

#[cfg(test)]
#[path = "model_scope_tests.rs"]
mod model_scope_tests;
