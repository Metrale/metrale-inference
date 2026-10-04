// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: `met benchmark spec-cost-table`: the measured speculative-cost table from
//! spec-cost runs, one per draft depth `0..=K`, all of them on one box against one serve
//! configuration. GPU-free.
//!
//! Owner: server CLI (`met benchmark`).
//! Invariants:
//! - The table is written only when it would load: `CostTable::render` validates the grid
//!   (every width at every depth `0..=K`, no duplicate, positive finite costs), and
//!   `table_input::read` refuses a record with a vacuous or unsampled width.
//! - The runs must agree on the box (machine id), the served model and every benchmark
//!   parameter other than `k`; a mix is refused, never merged.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use metrale_bench::benchmarks::spec_cost::{DESCRIPTOR, table_input};
use metrale_bench::history::RunRecord;
use metrale_speculative::spec_cost::{Cell, CostTable, SCHEMA, TableKey};

/// 2026-10-04: `met benchmark spec-cost-table` options.
#[derive(clap::Args, Debug)]
pub struct SpecCostTableArgs {
    /// A `met benchmark run spec-cost --format json` result. Repeat it, once per draft depth
    /// 0..=K.
    #[arg(long = "result", required = true)]
    pub results: Vec<PathBuf>,
    /// The HARDWARE.toml box class the runs were measured on, e.g. gb10.
    #[arg(long)]
    pub box_class: String,
    /// The recipe id the measured serve ran.
    #[arg(long)]
    pub recipe: String,
    /// MODE=DIGEST: the plan digest of one measured mode. Repeat it per mode.
    #[arg(long = "plan-digest", value_parser = super::bench_args::parse_kv, required = true)]
    pub plan_digests: Vec<(String, String)>,
    /// Where to write the table.
    #[arg(long)]
    pub out: PathBuf,
}

pub fn spec_cost_table_cmd(args: SpecCostTableArgs) -> Result<i32> {
    let mut records = Vec::with_capacity(args.results.len());
    for path in &args.results {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let record: RunRecord = serde_json::from_str(&text)
            .with_context(|| format!("{} is not a benchmark result", path.display()))?;
        records.push((path.display().to_string(), record));
    }
    let mut plan_digests = BTreeMap::new();
    for (mode, digest) in args.plan_digests {
        if plan_digests.insert(mode.clone(), digest).is_some() {
            bail!("--plan-digest {mode} is given twice");
        }
    }
    let key = TableKey {
        schema: SCHEMA,
        box_class: args.box_class,
        recipe: args.recipe,
        plan_digests,
    };
    let text = assemble(&key, &records)?;
    std::fs::write(&args.out, &text).with_context(|| format!("writing {}", args.out.display()))?;
    println!("wrote {} ({} runs)", args.out.display(), records.len());
    Ok(0)
}

/// 2026-10-04: The table text for `key` from `(label, record)` runs. Pure.
pub(crate) fn assemble(key: &TableKey, records: &[(String, RunRecord)]) -> Result<String> {
    let mut cells: Vec<Cell> = Vec::new();
    let mut first: Option<(&str, Identity)> = None;
    for (label, record) in records {
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
        let read = table_input::read(&record.frame.metrics).with_context(|| label.clone())?;
        cells.extend(read.into_iter().map(|c| Cell {
            n: c.n,
            k: c.k,
            verify_ms: c.verify_ms,
            verify_j: c.verify_j,
            draft_ms: c.draft_ms,
            draft_j: c.draft_j,
        }));
    }
    CostTable::render(key, &cells).map_err(anyhow::Error::msg)
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
