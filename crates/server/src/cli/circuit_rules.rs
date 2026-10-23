// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `met circuit rules --critical-pairs`: the I/O side of `metrale_circuit::critical`.
//! It plans every golden instance at every checked-in mode and row count on its own class, and
//! every roadmap-matrix cell (`met circuit plan --matrix`) at the report runs, resolves each
//! critical pair of the class's rules both ways, writes or checks the report, and runs the
//! shrink-only allowlist gate.
//!
//! Owner: server CLI.
//! Invariants:
//! - Nothing here matches, resolves or classifies; the crate does, over [`FsTree`].
//! - The gate runs in both modes: a priority-decided rule pair without an allowlist entry, a
//!   stale entry, or a list past [`CEILING`] is an error.
//! - `--check` compares and never writes; a stale or missing report is an error.
//! - Golden plans with a row table (`verify_batch`) are not covered: the hardware planner fuses
//!   modes and row counts only.

use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use metrale_circuit::critical_report::{PairRow, PlanPairs, gate, parse_allowlist, render};
use metrale_circuit::hardware::critical_hw::{critical_pairs_on, floor_us, site};
use metrale_circuit::hardware::model::{model_of, policy_on_class};
use metrale_circuit::hardware::plan::{report_runs, resolve};
use metrale_circuit::hardware::{
    CircuitSource, ModelSpec, ModelUnderPlan, PrecisionChoice, Registry,
};
use metrale_circuit::parse_instances;
use metrale_circuit::venn::{Repo, Run};

use super::circuit_hw::{
    FsTree, MATRIX_CONFIGS, MATRIX_DEVICES, MATRIX_MODELS, precision_of, registry, source,
    write_or_check,
};
use super::circuit_venn::find_root;

/// 2026-10-23: The committed report.
pub(crate) const REPORT: &str = "kernels/circuits/plans/critical/CRITICAL-PAIRS.md";
/// 2026-10-23: The allowlist of rule pairs settled by priority alone.
pub(crate) const ALLOWLIST: &str = "kernels/circuits/CRITICAL_PAIRS.toml";
/// 2026-10-23: The allowlist's seeded size. It may only fall: growing it is a visible edit here.
pub(crate) const CEILING: usize = 32;

/// `met circuit rules` options.
#[derive(clap::Args, Debug, Clone)]
pub struct CircuitRulesArgs {
    /// Report the critical pairs of the fusion rules: overlapping matches resolved both ways.
    #[arg(long, required = true)]
    pub critical_pairs: bool,
    /// Compare with the committed report instead of writing it; a stale one is an error.
    #[arg(long)]
    pub check: bool,
    /// The repository root (default: the directory above the working one that holds
    /// kernels/circuits/INSTANCES.toml).
    #[arg(long)]
    pub root: Option<PathBuf>,
}

fn plans_of(
    reg: &Registry,
    tree: &FsTree,
    model: &ModelUnderPlan,
    device: &str,
    runs: &[Run],
) -> Result<Vec<PlanPairs>> {
    let r = resolve(reg, device, tree, model)?;
    let (policy, _) = policy_on_class(&model.policy, model.settings_class.as_deref(), &r.chain[0])?;
    let c = &model.circuit;
    let mut out = Vec::new();
    for &run in runs {
        let (today, matches, pairs) =
            critical_pairs_on(&r, c, &policy, run).with_context(|| {
                format!(
                    "{} on {device} {} n={}",
                    model.label,
                    run.mode.name(),
                    run.rows
                )
            })?;
        out.push(PlanPairs {
            model: model.label.clone(),
            device: device.to_string(),
            run,
            matches,
            floor_us: floor_us(&r, c, &policy, &today)?,
            rows: pairs
                .into_iter()
                .map(|pair| PairRow {
                    site_a: site(c, &pair.a),
                    site_b: site(c, &pair.b),
                    pair,
                })
                .collect(),
        });
    }
    Ok(out)
}

/// 2026-10-23: The critical pairs of every covered plan.
pub(crate) fn all_plans(root: &std::path::Path) -> Result<Vec<PlanPairs>> {
    let tree = FsTree::new(root.to_path_buf());
    let reg = registry(&tree)?;
    let mut out = Vec::new();
    let mut seen: BTreeSet<(String, String, Run)> = BTreeSet::new();
    let mut add = |plans: Vec<PlanPairs>, out: &mut Vec<PlanPairs>| {
        for p in plans {
            if seen.insert((p.model.clone(), p.device.clone(), p.run)) {
                out.push(p);
            }
        }
    };
    let text = Repo::read(&tree, "kernels/circuits/INSTANCES.toml").map_err(anyhow::Error::msg)?;
    for inst in parse_instances(&text)?.iter().filter(|i| i.golden) {
        let m = model_of(&tree, inst, "recipe".into(), PrecisionChoice::Recipe)?;
        let device = inst.target.split('/').next().context("instance target")?;
        let runs: Vec<Run> = inst
            .plans
            .iter()
            .flat_map(|(&mode, rows)| rows.iter().map(move |&rows| Run { mode, rows }))
            .collect();
        add(plans_of(&reg, &tree, &m, device, &runs)?, &mut out);
    }
    for mm in MATRIX_MODELS {
        let dir = root
            .join(MATRIX_CONFIGS)
            .join(mm.checkpoint.replacen('/', "--", 1));
        let read = |f: &str| std::fs::read_to_string(dir.join(f)).ok();
        let (config, hf_quant) = (read("config.json"), read("hf_quant_config.json"));
        let m = source(&tree).model(&ModelSpec {
            checkpoint: mm.checkpoint,
            config_json: config.as_deref(),
            hf_quant: hf_quant.as_deref(),
            precision: precision_of(mm.precision),
        })?;
        for device in MATRIX_DEVICES {
            add(plans_of(&reg, &tree, &m, device, &report_runs())?, &mut out);
        }
    }
    Ok(out)
}

/// 2026-10-23: Run `met circuit rules`.
pub(crate) fn run(a: CircuitRulesArgs) -> Result<()> {
    if !a.critical_pairs {
        bail!("`met circuit rules` needs --critical-pairs");
    }
    let root = match &a.root {
        Some(r) => r.clone(),
        None => find_root(&std::env::current_dir()?)?,
    };
    let plans = all_plans(&root)?;
    let allowed = parse_allowlist(
        &std::fs::read_to_string(root.join(ALLOWLIST))
            .with_context(|| format!("reading {ALLOWLIST}"))?,
    )
    .map_err(|e| anyhow::anyhow!("{ALLOWLIST}: {e}"))?;
    let text = render(&plans, &allowed, "met circuit rules --critical-pairs");
    write_or_check(&root, REPORT, &text, a.check)?;
    let problems = gate(&plans, &allowed, CEILING);
    if !problems.is_empty() {
        bail!("{ALLOWLIST} gate:\n  {}", problems.join("\n  "));
    }
    eprintln!(
        "{REPORT}: {} ({} plans); allowlist gate passed",
        if a.check { "current" } else { "written" },
        plans.len()
    );
    Ok(())
}

#[cfg(test)]
#[path = "circuit_rules_tests.rs"]
mod circuit_rules_tests;
