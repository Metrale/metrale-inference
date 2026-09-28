// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `met circuit show|display`. The I/O side of metrale-circuit: the circuit TOMLs,
//! precision tables and FUSIONS.toml this binary was built from (embedded, so the view matches
//! the kernels compiled in), the recipe's instance, and, when the checkpoint is in the local
//! cache, a check of the instance's shape against its `config.json`.
//!
//! Owner: server CLI.
//! Invariants:
//! - Nothing here decides a kernel: planning and drawing are metrale-circuit's pure functions.
//! - An unknown recipe, mode rows the instance cannot give, and a cached checkpoint whose
//!   shape disagrees with INSTANCES.toml are errors, never a silent fallback.

use std::io::{IsTerminal, Write};

use anyhow::{Context, Result, bail};
use metrale_circuit::display::{DisplayOpts, Expand, Glyphs};
use metrale_circuit::{AvailableKernels, Instance, Mode};

use super::{CircuitAction, CircuitArgs, CircuitMode, CircuitPlanArgs, circuit_paint};

// 2026-09-28: The embedded texts and the instance lookups live with the executor, so the view
// and the executor read one copy.
pub(crate) use metrale_model_layers::circuit_exec::sources::{
    arch_shape, instance, shape_drift, sources,
};

fn mode_of(m: CircuitMode) -> Mode {
    match m {
        CircuitMode::Decode => Mode::Decode,
        CircuitMode::MultiSeq => Mode::MultiSeq,
        CircuitMode::Verify => Mode::Verify,
        CircuitMode::Draft => Mode::Draft,
    }
}

/// 2026-09-28: `--rows`, or one row for the modes that plan one.
pub(crate) fn rows_of(inst: &Instance, args: &CircuitPlanArgs) -> Result<u64> {
    let mode = mode_of(args.mode);
    match (args.rows, mode) {
        (Some(0), _) => bail!("--rows must be at least 1"),
        (Some(r), _) => Ok(r),
        (None, Mode::Decode | Mode::Draft) => Ok(1),
        (None, _) => {
            let listed = inst.plans.get(&mode).cloned().unwrap_or_default();
            bail!(
                "--mode {} needs --rows; the checked-in plans cover {listed:?}",
                mode.name()
            )
        }
    }
}

fn check_shape(inst: &Instance) -> Result<()> {
    let Ok(dir) = crate::model_resolver::resolve_model_dir(&inst.checkpoint, None) else {
        eprintln!(
            "note: {} is not in the local cache; the shape is INSTANCES.toml's, unchecked",
            inst.checkpoint
        );
        return Ok(());
    };
    let text = std::fs::read_to_string(dir.join("config.json"))
        .with_context(|| format!("reading {}", dir.join("config.json").display()))?;
    let cfg = metrale_config::parse_config(&text)?;
    let drift = shape_drift(&inst.shape, &arch_shape(&cfg)?);
    if !drift.is_empty() {
        bail!(
            "kernels/circuits/INSTANCES.toml disagrees with {}'s config.json:\n  {}",
            inst.checkpoint,
            drift.join("\n  ")
        );
    }
    Ok(())
}

/// 2026-09-28: Run `met circuit`.
pub(crate) fn dispatch(args: CircuitArgs) -> Result<()> {
    let plan_args = match &args.action {
        CircuitAction::Show(p) => p.clone(),
        CircuitAction::Display(d) => d.plan.clone(),
        CircuitAction::Diff(_) => {
            let CircuitAction::Diff(d) = args.action else {
                unreachable!("matched above")
            };
            return tokio::task::block_in_place(|| super::circuit_diff::run_diff(*d));
        }
    };
    let inst = instance(&plan_args.recipe)?;
    let rows = rows_of(&inst, &plan_args)?;
    let mode = mode_of(plan_args.mode);
    check_shape(&inst)?;
    let loaded = metrale_circuit::load(&inst, sources(&inst)?)?;
    // 2026-09-28: Offline, no target is probed: every kernel a rule names counts as built.
    let avail = AvailableKernels::all_named_by(&loaded.rules);
    // 2026-09-28: The plan digest covers what the plan runs; the rule set it was chosen from
    // is attested separately (and by the closure hash).
    eprintln!("rules: FUSIONS.toml sha256 {}", loaded.rules_digest);
    let text = match args.action {
        CircuitAction::Show(_) => metrale_circuit::render_plan(&inst, &loaded, &avail, mode, rows)?,
        CircuitAction::Diff(_) => unreachable!("returned above"),
        CircuitAction::Display(d) => {
            let tty = std::io::stdout().is_terminal();
            let width = if tty {
                crossterm::terminal::size().map_or(100, |(w, _)| w as usize)
            } else {
                100
            };
            let expand = match (d.layer, d.all_layers) {
                (Some(n), _) => Expand::Layer(n),
                (None, true) => Expand::AllLayers,
                (None, false) => Expand::Summary,
            };
            let glyphs = if d.ascii {
                Glyphs::Ascii
            } else {
                Glyphs::Unicode
            };
            let opts = DisplayOpts {
                width,
                glyphs,
                expand,
            };
            let doc = metrale_circuit::display_plan(&inst, &loaded, &avail, mode, rows, &opts)?;
            let depth = circuit_paint::resolve_depth(
                d.color,
                tty,
                std::env::var("NO_COLOR").ok().as_deref(),
                std::env::var("COLORTERM").ok().as_deref(),
            );
            circuit_paint::paint(&doc, depth)
        }
    };
    // 2026-09-28: A reader that closes early (`| head`) has what it asked for; that is not
    // a failure of the command.
    match std::io::stdout().lock().write_all(text.as_bytes()) {
        Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
        other => Ok(other?),
    }
}

#[cfg(test)]
#[path = "circuit_tests.rs"]
mod circuit_tests;
