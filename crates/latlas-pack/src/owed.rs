// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The LKB queries over the bundle (queries part Q3, Q4, Q6 and `dangling`): the LKB
//! rows owed for an instance's kernels (the `lkb_every_campaign` rule, mechanised), measurements
//! that predate a change of their kernel's file, priority-decided critical pairs, and evidence
//! keys that name no measurement row.
//!
//! Owner: metrale-latlas-pack (the metrale pack; was metrale-kg-derive).
//! Invariants:
//! - A kernel is `landed` for an instance when one of its golden plans `LAUNCHES` it or one of its
//!   headroom sites `RUNS_KERNEL` it; every kernel compiled into the instance's target is checked
//!   too (`landed = false`), so an unclassified kernel the target builds is never missed.
//! - The four facets are exactly the design's: a family point, a taxonomy component, a measurement
//!   for the instance's model and hardware, an evidence row keyed to such a measurement.

use crate::prelude::*;
use crate::vocab::*;
use latlas_core::bundle::Bundle;
use latlas_error::{ErrorCode, Result, error};
use serde_json::Value;

/// 2026-10-23: One kernel's LKB facets for an instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Owed {
    pub kernel: NodeId,
    pub landed: bool,
    pub has_point: bool,
    pub classified: bool,
    pub measured: bool,
    pub evidenced: bool,
}

impl Owed {
    /// 2026-10-23: Whether any facet is missing.
    pub fn owes(&self) -> bool {
        !(self.has_point && self.classified && self.measured && self.evidenced)
    }
}

fn s<'a>(b: &'a Bundle, id: &NodeId, k: &str) -> &'a str {
    b.field(id, k).unwrap_or("")
}

/// 2026-10-23: Q3: the facets of every kernel of `instance` that owes at least one.
pub fn owed_lkb_rows(b: &Bundle, instance: &str) -> Result<Vec<Owed>> {
    let mut out: Vec<Owed> = lkb_facets(b, instance)?
        .into_iter()
        .filter(Owed::owes)
        .collect();
    out.sort_by(|a, b| (!a.landed, &a.kernel).cmp(&(!b.landed, &b.kernel)));
    Ok(out)
}

/// 2026-10-23: The facets of every kernel `instance` lands or compiles, owed or not (the subjects
/// of the `lkb` obligation rule), in kernel order.
pub fn lkb_facets(b: &Bundle, instance: &str) -> Result<Vec<Owed>> {
    let inst = NodeId::new(labels::Instance, instance)?;
    if b.node(&inst).is_none() {
        return Err(error!(ErrorCode::LatlasUnknownId, inst.to_string()));
    }
    let target = s(b, &inst, "target").to_string();
    let arch = NodeId::new(labels::Model, s(b, &inst, "arch"))?;
    let hw = NodeId::new(labels::Hardware, target.split('/').next().unwrap_or(""))?;
    let mut landed: BTreeSet<NodeId> = BTreeSet::new();
    for e in b.out(&inst, rels::Plans) {
        landed.extend(b.out(&e.to, rels::Launches).map(|l| l.to.clone()));
    }
    for site in b
        .of(labels::HeadroomSite)
        .filter(|n| n.p.get("instance").and_then(Value::as_str) == Some(instance))
    {
        landed.extend(b.out(&site.id, rels::RunsKernel).map(|l| l.to.clone()));
    }
    let compiled = b.of(labels::Kernel).filter(|k| {
        k.p.get("targets")
            .and_then(Value::as_array)
            .is_some_and(|t| t.iter().any(|x| x.as_str() == Some(target.as_str())))
    });
    let mut all: BTreeSet<NodeId> = compiled.map(|k| k.id.clone()).collect();
    all.extend(landed.iter().cloned());
    let mut out = Vec::new();
    for k in all {
        let has_point = b
            .out(&k, rels::Instantiates)
            .any(|e| e.to.label() == labels::FamilyPoint);
        let classified = b.node(&k).is_some_and(|n| {
            n.p.get("taxonomy_component")
                .is_some_and(|c| c.as_str().is_some_and(|c| !c.is_empty()))
        });
        let ms: Vec<&NodeId> = b
            .into(&k, rels::Measures)
            .map(|e| &e.from)
            .filter(|m| {
                b.out(m, rels::ForModel).any(|f| f.to == arch)
                    && b.out(m, rels::Targets).any(|t| t.to == hw)
            })
            .collect();
        let evidenced = ms.iter().any(|m| b.into(m, rels::KeyedTo).next().is_some());
        let measured = !ms.is_empty();
        let is_landed = landed.contains(&k);
        let row = Owed {
            landed: is_landed,
            kernel: k,
            has_point,
            classified,
            measured,
            evidenced,
        };
        out.push(row);
    }
    Ok(out)
}

/// 2026-10-23: Q4: `(measurement, kernel file, measured at, changed at)` for every measurement
/// whose file has a commit newer than the measurement's own, from the host's git facts
/// (`file → (sha, date)` of its last commit, full `sha → date`; a measurement's short sha is
/// matched as a prefix).
pub fn stale_measurements(
    b: &Bundle,
    file_last: &BTreeMap<String, (String, String)>,
    commit_date: &BTreeMap<String, String>,
) -> Vec<(NodeId, String, String, String)> {
    let mut out = Vec::new();
    for m in b.of(labels::Measurement) {
        let Some(c) = m.p.get("source_commit").and_then(Value::as_str) else {
            continue;
        };
        let file = m.p.get("file").and_then(Value::as_str).unwrap_or("");
        let date = commit_date
            .range(c.to_string()..)
            .next()
            .filter(|(full, _)| full.starts_with(c))
            .map(|(_, d)| d);
        let (Some((last, last_date)), Some(date)) = (file_last.get(file), date) else {
            continue;
        };
        let newer =
            matches!((last_date.parse::<u64>(), date.parse::<u64>()), (Ok(l), Ok(d)) if l > d);
        if newer && !last.starts_with(c) {
            out.push((m.id.clone(), file.to_string(), c.to_string(), last.clone()));
        }
    }
    out
}

/// 2026-10-23: Mark every stale measurement (Q4) with `stale_since` = the newer commit of its
/// kernel file, so `la_status owed_lkb` can show it without the host's git facts.
pub fn mark_stale(
    b: &mut Bundle,
    file_last: &BTreeMap<String, (String, String)>,
    commit_date: &BTreeMap<String, String>,
) {
    let stale = stale_measurements(b, file_last, commit_date);
    for (m, _, _, changed) in stale {
        if let Some(n) = b.nodes.iter_mut().find(|n| n.id == m) {
            n.p.insert("stale_since".into(), Value::String(changed));
        }
    }
}

/// 2026-10-23: Measurements whose `source` names no commit (Q4's `no_commit_in_source`).
pub fn uncommitted_measurements(b: &Bundle) -> Vec<NodeId> {
    b.of(labels::Measurement)
        .filter(|m| m.p.get("dangling_commit") == Some(&Value::Bool(true)))
        .map(|m| m.id.clone())
        .collect()
}

/// 2026-10-23: Evidence rows whose measurement key names no `[[m]]` row (an SSOT defect).
pub fn evidence_without_row(b: &Bundle) -> Vec<(NodeId, NodeId)> {
    let mut out = Vec::new();
    for e in b.of(labels::Evidence) {
        for k in b.out(&e.id, rels::KeyedTo) {
            if k.to.label() == labels::Unresolved {
                out.push((e.id.clone(), k.to.clone()));
            }
        }
    }
    out
}

/// 2026-10-23: Q6: critical pairs decided by priority, with their rules and priorities.
pub fn priority_pairs(b: &Bundle) -> Vec<(NodeId, Vec<(NodeId, i64)>)> {
    b.of(labels::CriticalPair)
        .filter(|cp| cp.p.get("resolved_by").and_then(Value::as_str) == Some("priority"))
        .map(|cp| {
            let rules = b
                .out(&cp.id, rels::Overlaps)
                .map(|e| {
                    (
                        e.to.clone(),
                        b.node(&e.to)
                            .and_then(|n| n.p.get("priority"))
                            .and_then(Value::as_i64)
                            .unwrap_or(0),
                    )
                })
                .collect();
            (cp.id.clone(), rules)
        })
        .collect()
}
