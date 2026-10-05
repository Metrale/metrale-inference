// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Reads a spec-cost run's record back into per-width step costs, the input of
//! `met benchmark spec-cost-table`. The writer is `cell::record`; the key names are shared
//! with it. Pure: no I/O.
//!
//! The cost a table carries is the WALL time of a step (`wall_ms`, the window over the step
//! count), split into its draft share (`draft_ms`, the timed propose phase) and the rest
//! (`verify_ms` = `wall_ms` − `draft_ms`, which includes the scheduler loop between steps).
//! That makes the step time agree with the energy, which is the whole window's joules over
//! the step count, and with `k = 0`, whose step time is wall time by construction.
//!
//! Owner: bench, spec-cost.
//! Invariants: a record is read only when every width it lists was measured with the rail
//! sampled; a vacuous width, or one without joules, refuses the whole record.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};

use super::cell::{
    KEY_DRAFT_J, KEY_DRAFT_MS, KEY_K, KEY_VACUOUS, KEY_VERIFY_J, KEY_WALL_MS, record_key,
};

/// 2026-10-04: One width's step cost at the record's draft depth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeasuredCell {
    pub n: usize,
    pub k: usize,
    pub verify_ms: f64,
    pub verify_j: f64,
    pub draft_ms: f64,
    pub draft_j: f64,
}

/// 2026-10-04: The cells of one spec-cost record (`frame.metrics` of a run), ascending in
/// width. Refused: no `k`, no width, a vacuous width, a width without joules (the rail was
/// not sampled), or a draft share larger than the step.
pub fn read(metrics: &BTreeMap<String, f64>) -> Result<Vec<MeasuredCell>> {
    let k = *metrics
        .get(KEY_K)
        .context("not a spec-cost record: no `k`")?;
    if !(k >= 0.0 && k.fract() == 0.0) {
        bail!("spec-cost record: k = {k} is not a draft count");
    }
    let suffix = format!("_{KEY_VACUOUS}");
    let mut widths: Vec<usize> = metrics
        .keys()
        .filter_map(|key| {
            key.strip_prefix('n')?
                .strip_suffix(suffix.as_str())?
                .parse()
                .ok()
        })
        .collect();
    widths.sort_unstable();
    if widths.is_empty() {
        bail!("spec-cost record at k = {k}: no width");
    }
    let field = |n: usize, f: &str| -> Result<f64> {
        metrics.get(&record_key(n, f)).copied().with_context(|| {
            format!("spec-cost record at k = {k}: width {n} has no {f} (rail not sampled?)")
        })
    };
    widths
        .into_iter()
        .map(|n| {
            if field(n, KEY_VACUOUS)? != 0.0 {
                bail!("spec-cost record at k = {k}: width {n} is vacuous; re-run it");
            }
            let (wall_ms, draft_ms) = (field(n, KEY_WALL_MS)?, field(n, KEY_DRAFT_MS)?);
            if draft_ms > wall_ms {
                bail!("spec-cost record at k = {k}: width {n} drafts {draft_ms} ms of a {wall_ms} ms step");
            }
            Ok(MeasuredCell {
                n,
                k: k as usize,
                verify_ms: wall_ms - draft_ms,
                verify_j: field(n, KEY_VERIFY_J)?,
                draft_ms,
                draft_j: field(n, KEY_DRAFT_J)?,
            })
        })
        .collect()
}

#[cfg(test)]
#[path = "table_input_tests.rs"]
mod tests;
