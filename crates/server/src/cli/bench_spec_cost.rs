// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: `met benchmark spec-cost-table`: the measured speculative-cost table from
//! spec-cost runs, one per draft depth `0..=K`, all of them on one box against one serve
//! configuration; and the drafter's acceptance calibration, fitted on the acceptance counts
//! of every run that has them (pooled). GPU-free.
//!
//! Owner: server CLI (`met benchmark`).
//! Invariants:
//! - The table is written only when it would load: `CostTable::render` validates the grid
//!   (every width at every depth `0..=K`, no duplicate, positive finite costs), and
//!   `table_input::read` refuses a record with a vacuous or unsampled width.
//! - The runs must agree on the box (machine id), the served model and every benchmark
//!   parameter other than `k`; a mix is refused, never merged. Repeats of a depth are combined
//!   into their per-field median; one file given twice is refused.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use metrale_bench::benchmarks::spec_cost::{DESCRIPTOR, acceptance, table_input};
use metrale_bench::history::RunRecord;
use metrale_speculative::spec_cost::{
    AcceptanceCalibration, Cell, CostTable, DrafterKey, SCHEMA, TableKey,
};

use super::bench_args::SpecCostTableArgs;

pub fn spec_cost_table_cmd(args: SpecCostTableArgs) -> Result<i32> {
    let mut records = Vec::with_capacity(args.results.len());
    for path in &args.results {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let record: RunRecord = serde_json::from_str(&text)
            .with_context(|| format!("{} is not a benchmark result", path.display()))?;
        records.push((path.display().to_string(), record));
    }
    let plan_digests =
        metrale_model_layers::circuit_exec::spec_key::spec_cost_plan_digests(&args.recipe)?;
    let key = TableKey {
        schema: SCHEMA,
        box_class: args.box_class,
        recipe: args.recipe,
        plan_digests,
    };
    let drafter = DrafterKey {
        weights_sha256: args.drafter_weights_sha256,
        vocab: args.mtp_vocab,
        quantization: args.mtp_quantization,
        context: args.mtp_context,
    };
    let table = assemble(&key, &records)?;
    let calibration = calibrate(drafter, &records)?;
    for (path, text) in [(&args.out, &table), (&args.calibration_out, &calibration)] {
        std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
        println!("wrote {} ({} runs)", path.display(), records.len());
    }
    Ok(0)
}

/// 2026-10-04: The calibration text for `drafter`, fitted on the pooled acceptance counts of
/// every run that has them; their bucket edges must agree. Pure.
pub(crate) fn calibrate(drafter: DrafterKey, records: &[(String, RunRecord)]) -> Result<String> {
    let mut pooled: Option<acceptance::AcceptanceCounts> = None;
    for (label, record) in records {
        let Some(c) = acceptance::read(&record.frame.metrics).with_context(|| label.clone())?
        else {
            continue;
        };
        let Some(p) = pooled.as_mut() else {
            pooled = Some(c);
            continue;
        };
        if p.edges != c.edges {
            bail!(
                "{label}: confidence buckets {:?}, other runs {:?}",
                c.edges,
                p.edges
            );
        }
        for i in 0..p.edges.len() {
            p.accepted[i] += c.accepted[i];
            p.rejected[i] += c.rejected[i];
        }
        for (shape, n) in c.steps {
            *p.steps.entry(shape).or_default() += n;
        }
    }
    let Some(p) = pooled else {
        bail!(
            "no run carries draft-confidence counts: run spec-cost at k >= 1 with a width of 2 \
             or more against a serve that exports metrale_spec_draft_confidence_total"
        );
    };
    AcceptanceCalibration::fit(drafter, &p.edges, &p.accepted, &p.rejected, &p.steps)
        .and_then(|c| c.render())
        .map_err(anyhow::Error::msg)
}

/// 2026-10-04: The table text for `key` from `(label, record)` runs. Repeats of a depth
/// (interleaved runs) are combined field by field into their median. Pure.
pub(crate) fn assemble(key: &TableKey, records: &[(String, RunRecord)]) -> Result<String> {
    let mut repeats: BTreeMap<(usize, usize), Vec<table_input::MeasuredCell>> = BTreeMap::new();
    let mut first: Option<(&str, Identity)> = None;
    for (i, (label, record)) in records.iter().enumerate() {
        if records[..i].iter().any(|(l, _)| l == label) {
            bail!("{label} is given twice");
        }
        if record.benchmark_id != DESCRIPTOR.id {
            bail!(
                "{label}: a {} result, not {}",
                record.benchmark_id,
                DESCRIPTOR.id
            );
        }
        let identity = Identity::of(record);
        match &first {
            None => first = Some((label, identity)),
            Some((first_label, want)) if *want != identity => bail!(
                "{label} and {first_label} were not measured alike (box, model or parameters \
                 other than k differ): {identity:?} vs {want:?}"
            ),
            Some(_) => {}
        }
        for c in table_input::read(&record.frame.metrics).with_context(|| label.clone())? {
            repeats.entry((c.n, c.k)).or_default().push(c);
        }
    }
    let cells: Vec<Cell> = repeats
        .into_iter()
        .map(|((n, k), reps)| {
            let field =
                |f: fn(&table_input::MeasuredCell) -> f64| median(reps.iter().map(f).collect());
            Cell {
                n,
                k,
                verify_ms: field(|c| c.verify_ms),
                verify_j: field(|c| c.verify_j),
                draft_ms: field(|c| c.draft_ms),
                draft_j: field(|c| c.draft_j),
            }
        })
        .collect();
    CostTable::render(key, &cells).map_err(anyhow::Error::msg)
}

/// 2026-10-04: The median of a non-empty list; the mean of the middle two for an even count.
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        (v[m - 1] + v[m]) / 2.0
    }
}

/// 2026-10-04: What must be equal across the runs of one table.
#[derive(Debug, PartialEq, Eq)]
struct Identity {
    machine_id: Option<String>,
    model: String,
    params: BTreeMap<String, String>,
}

impl Identity {
    fn of(record: &RunRecord) -> Self {
        let mut params = record.params.clone();
        params.remove("k");
        Self {
            machine_id: record
                .frame
                .hardware_state
                .as_ref()
                .and_then(|h| h.before.machine.machine_id.clone()),
            model: record.target_model.clone(),
            params,
        }
    }
}

#[cfg(test)]
#[path = "bench_spec_cost_tests.rs"]
mod tests;
