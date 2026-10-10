// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The row merge: per shape, decisions at row counts adjacent in the swept ladder
//! that are identical (kernel, family, default, numerics, enabled) become one `rows = [lo, hi]`
//! entry; then the file, with its sources.
//!
//! Owner: metrale-accuracy (envelope).
//! Invariants:
//! - Adjacency is in the shape's swept ladder ([`Selection::swept`]): a swept row with no
//!   decision (undecided) breaks a run, so a merged range never spans an unproven row count.
//! - A merged entry carries the times and provenance of its `hi` row, the largest launch.

use std::collections::BTreeMap;

use super::super::schedules::{SCHEMA, Schedule, Schedules, SchedulesError, Source, check};
use super::{Decision, Selection, shape_of};

fn entry(d: &Decision, lo: u64) -> Schedule {
    Schedule {
        op: d.cell.op.clone(),
        weight: d.cell.weight.clone(),
        activation: d.cell.activation.clone(),
        k: d.cell.k,
        n: d.cell.n,
        rows: [lo, d.cell.rows],
        kernel: d.kernel.clone(),
        family: d.family.clone(),
        default: d.default.clone().unwrap_or_default(),
        numerics: d.numerics,
        enabled: d.enabled,
        median_us: d.median_us,
        default_us: d.default_us.unwrap_or(0.0),
        floor_us: d.floor_us,
        measured: d.measured.clone(),
    }
}

fn same_decision(a: &Decision, b: &Decision) -> bool {
    a.kernel == b.kernel
        && a.family == b.family
        && a.default == b.default
        && a.numerics == b.numerics
        && a.enabled == b.enabled
}

/// 2026-10-10: The merged entries, in (shape, rows) order.
pub fn merge_rows(sel: &Selection) -> Vec<Schedule> {
    let mut by_shape: BTreeMap<_, Vec<&Decision>> = BTreeMap::new();
    for d in &sel.decisions {
        by_shape.entry(shape_of(&d.cell)).or_default().push(d);
    }
    let mut out = Vec::new();
    for (shape, mut ds) in by_shape {
        ds.sort_by_key(|d| d.cell.rows);
        let ladder: Vec<u64> = sel
            .swept
            .get(&shape)
            .map(|r| r.iter().copied().collect())
            .unwrap_or_default();
        let index = |rows: u64| ladder.iter().position(|&r| r == rows);
        let mut run: Option<(u64, &Decision)> = None;
        for d in ds {
            run = match run {
                Some((lo, prev))
                    if same_decision(prev, d)
                        && index(prev.cell.rows)
                            .zip(index(d.cell.rows))
                            .is_some_and(|(i, j)| j == i + 1) =>
                {
                    Some((lo, d))
                }
                Some((lo, prev)) => {
                    out.push(entry(prev, lo));
                    Some((d.cell.rows, d))
                }
                None => Some((d.cell.rows, d)),
            };
        }
        if let Some((lo, prev)) = run {
            out.push(entry(prev, lo));
        }
    }
    out
}

/// 2026-10-10: The file for `sel`: merged entries, the sources of their families (the caller
/// reads them with [`super::super::sources::sources_of`] for [`families`]), checked.
pub fn schedules(
    sel: &Selection,
    generated_by: &str,
    sources: BTreeMap<String, Source>,
) -> Result<Schedules, SchedulesError> {
    let s = Schedules {
        schema: SCHEMA,
        hardware: sel.hardware.clone(),
        generated_by: generated_by.to_string(),
        sources,
        schedule: merge_rows(sel),
    };
    check(&s)?;
    Ok(s)
}

/// 2026-10-10: The families whose sources the file records: those of the decisions, sorted.
pub fn families(sel: &Selection) -> Vec<String> {
    let set: std::collections::BTreeSet<&String> =
        sel.decisions.iter().map(|d| &d.family).collect();
    set.into_iter().cloned().collect()
}
