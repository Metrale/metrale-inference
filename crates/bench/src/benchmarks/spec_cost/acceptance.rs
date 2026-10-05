// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: The run's acceptance counts, for the drafter's confidence→acceptance
//! calibration: between the probe's `/metrics` page and the one after the last width, the
//! change in
//! - `metrale_spec_draft_confidence_total{le, accepted}`: drafts a verify step reached, by
//!   the drafter's top-1 log-probability bucket (upper edge `le`) and verdict; and
//! - `metrale_spec_verify_steps_total{drafts, accepted}`: verify steps by drafts checked and
//!   accepted, for the per-position priors.
//!
//! Both are whole-run totals, not per-width: acceptance is a property of the drafter and the
//! text, while the cost cells are per width. Pure: no I/O.
//!
//! Owner: bench, spec-cost.
//! Invariants:
//! - A series absent from the first page counts from 0 (the serve renders them once they
//!   fire); one that went backwards fails the run (the serve restarted).
//! - The record keys are written by `record` and read by [`read`] only.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};

use super::prom::Scrape;

const CONFIDENCE_METRIC: &str = "metrale_spec_draft_confidence_total";
const STEPS_METRIC: &str = "metrale_spec_verify_steps_total";

/// 2026-10-04: The run's acceptance counts, read back from a record.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AcceptanceCounts {
    /// 2026-10-04: Bucket upper edges, ascending, as the serve labelled them.
    pub edges: Vec<f32>,
    pub accepted: Vec<u64>,
    pub rejected: Vec<u64>,
    /// 2026-10-04: `(drafts, accepted)` → verify steps.
    pub steps: BTreeMap<(usize, usize), u64>,
}

fn delta(first: &Scrape, last: f64, name: &str, labels: &[(&str, &str)]) -> Result<u64> {
    let before = first.value(name, labels)?.unwrap_or(0.0);
    let d = last - before;
    if d < 0.0 || !d.is_finite() {
        bail!("{name}{labels:?} went from {before} to {last}: the serve restarted");
    }
    Ok(d as u64)
}

/// 2026-10-04: Writes the counts between `first` and `last` into `m`: per bucket `i`
/// (ascending edge) `conf{i}_le`, `conf{i}_accepted`, `conf{i}_rejected`; per verify shape
/// `steps_d{d}_a{a}`.
pub(crate) fn record(first: &Scrape, last: &Scrape, m: &mut BTreeMap<String, f64>) -> Result<()> {
    let mut buckets: BTreeMap<String, (f32, [u64; 2])> = BTreeMap::new();
    for s in last.series.iter().filter(|s| s.name == CONFIDENCE_METRIC) {
        let (Some(le), Some(acc)) = (s.labels.get("le"), s.labels.get("accepted")) else {
            bail!(
                "{CONFIDENCE_METRIC} without le/accepted labels: {:?}",
                s.labels
            );
        };
        let edge: f32 = le
            .parse()
            .with_context(|| format!("{CONFIDENCE_METRIC} le={le:?}"))?;
        let slot = match acc.as_str() {
            "1" => 1,
            "0" => 0,
            other => bail!("{CONFIDENCE_METRIC} accepted={other:?}"),
        };
        let d = delta(
            first,
            s.value,
            CONFIDENCE_METRIC,
            &[("le", le), ("accepted", acc)],
        )?;
        buckets.entry(le.clone()).or_insert((edge, [0, 0])).1[slot] = d;
    }
    let mut ordered: Vec<(f32, [u64; 2])> = buckets.into_values().collect();
    ordered.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (i, (edge, [rej, acc])) in ordered.into_iter().enumerate() {
        m.insert(format!("conf{i}_le"), f64::from(edge));
        m.insert(format!("conf{i}_accepted"), acc as f64);
        m.insert(format!("conf{i}_rejected"), rej as f64);
    }
    for s in last.series.iter().filter(|s| s.name == STEPS_METRIC) {
        let (Some(d), Some(a)) = (s.labels.get("drafts"), s.labels.get("accepted")) else {
            bail!(
                "{STEPS_METRIC} without drafts/accepted labels: {:?}",
                s.labels
            );
        };
        let n = delta(
            first,
            s.value,
            STEPS_METRIC,
            &[("drafts", d), ("accepted", a)],
        )?;
        m.insert(format!("steps_d{d}_a{a}"), n as f64);
    }
    Ok(())
}

/// 2026-10-04: The counts `record` wrote; `None` when the run saw no draft confidence (a
/// `k = 0` serve, or one that never batched a verify).
pub fn read(metrics: &BTreeMap<String, f64>) -> Result<Option<AcceptanceCounts>> {
    let mut c = AcceptanceCounts::default();
    while let Some(&le) = metrics.get(&format!("conf{}_le", c.edges.len())) {
        let i = c.edges.len();
        let get = |f: &str| {
            metrics
                .get(&format!("conf{i}_{f}"))
                .copied()
                .with_context(|| format!("acceptance record: bucket {i} has no {f}"))
        };
        c.accepted.push(get("accepted")? as u64);
        c.rejected.push(get("rejected")? as u64);
        c.edges.push(le as f32);
    }
    if c.edges.is_empty() {
        return Ok(None);
    }
    for (key, &v) in metrics {
        let Some(rest) = key.strip_prefix("steps_d") else {
            continue;
        };
        let parsed = rest
            .split_once("_a")
            .and_then(|(d, a)| Some((d.parse().ok()?, a.parse().ok()?)));
        let shape = parsed.with_context(|| format!("acceptance record: bad key {key:?}"))?;
        c.steps.insert(shape, v as u64);
    }
    Ok(Some(c))
}

#[cfg(test)]
#[path = "acceptance_tests.rs"]
mod tests;
