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
use metrale_circuit::hardware::CircuitSource;
use metrale_circuit::{AvailableKernels, Instance, Mode};

use super::{
    CircuitAction, CircuitArgs, CircuitDisplayArgs, CircuitMode, CircuitPlanArgs, circuit_paint,
};

// 2026-09-28: The embedded texts and the instance lookups live with the executor, so the view
// and the executor read one copy.
pub(crate) use metrale_model_layers::circuit_exec::sources::{instance, shape_drift, sources};

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
    if let metrale_circuit::PrecisionSpec::Policy {
        checkpoint_plan, ..
    } = &inst.precision
    {
        let cached: serde_json::Value = serde_json::from_str(&text)?;
        let fixture = metrale_model_layers::circuit_exec::sources::lookup(
            &metrale_model_layers::circuit_exec::sources::CHECKPOINTS,
            checkpoint_plan,
            "checkpoint plan",
        )?;
        let table: toml::Table = toml::from_str(fixture)?;
        let stored: serde_json::Value = serde_json::from_str(
            table
                .get("quantization_config")
                .and_then(|v| v.as_str())
                .context("checkpoint plan has no quantization_config")?,
        )?;
        if cached.get("quantization_config") != Some(&stored) {
            bail!(
                "kernels/circuits/checkpoints/{checkpoint_plan}.toml's quantization_config differs \
                 from {}'s config.json",
                inst.checkpoint
            );
        }
    }
    let drift = shape_drift(&inst.shape, &metrale_circuit::map_checkpoint(&text)?.shape);
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
        CircuitAction::Venn(_) => {
            let CircuitAction::Venn(v) = args.action else {
                unreachable!("matched above")
            };
            return super::circuit_venn::run(*v);
        }
        CircuitAction::Plan(_) => {
            let CircuitAction::Plan(p) = args.action else {
                unreachable!("matched above")
            };
            return super::circuit_hw::run(*p);
        }
        CircuitAction::Memory(_) => {
            let CircuitAction::Memory(m) = args.action else {
                unreachable!("matched above")
            };
            return super::circuit_memory::run(*m);
        }
        CircuitAction::Precision(_) => {
            let CircuitAction::Precision(p) = args.action else {
                unreachable!("matched above")
            };
            return super::circuit_precision::run(*p);
        }
    };
    let inst = instance(&plan_args.recipe)?;
    let rows = rows_of(&inst, &plan_args)?;
    let mode = mode_of(plan_args.mode);
    check_shape(&inst)?;
    // 2026-10-05: `--hardware`: the device's class plans it from the working tree, for `show`
    // and `display` alike (the plan `met circuit plan --format plan` prints).
    if let Some(hw) = &plan_args.device.hardware {
        return emit(&device_text(&args.action, &inst, hw, (mode, rows))?);
    }
    let loaded = metrale_circuit::load(&inst, sources(&inst)?)?;
    // 2026-09-28: Offline, no target is probed: every kernel a rule names counts as built.
    let avail = AvailableKernels::all_named_by(&loaded.rules);
    // 2026-09-28: The plan digest covers what the plan runs; the rule set it was chosen from
    // is attested separately (and by the closure hash).
    eprintln!("rules: FUSIONS.toml sha256 {}", loaded.rules_digest);
    let text = match args.action {
        CircuitAction::Show(_) => {
            let families = families_for(&inst)?;
            metrale_circuit::render_plan(&inst, &loaded, &avail, mode, rows, &families)?
        }
        CircuitAction::Diff(_)
        | CircuitAction::Venn(_)
        | CircuitAction::Plan(_)
        | CircuitAction::Memory(_)
        | CircuitAction::Precision(_) => {
            unreachable!("returned above")
        }
        CircuitAction::Display(d) => {
            let doc = metrale_circuit::display_plan(
                &inst,
                &loaded,
                &avail,
                (mode, rows),
                &display_opts(&d),
                &families_for(&inst)?,
            )?;
            paint(&d, &doc)
        }
    };
    emit(&text)
}

/// 2026-10-05: `show` or `display` of `inst` planned on device `hw`: `show` prints the plan
/// `met circuit plan --format plan` prints, `display` draws it.
pub(crate) fn device_text(
    action: &CircuitAction,
    inst: &Instance,
    hw: &str,
    (mode, rows): (Mode, u64),
) -> Result<String> {
    let (tree, reg, _) = super::circuit_hw::tree_here()?;
    // 2026-10-05: The model `met circuit plan --checkpoint <recipe> --precision recipe` plans.
    let model = super::circuit_hw::source(&tree).model(&metrale_circuit::hardware::ModelSpec {
        checkpoint: &inst.recipe,
        config_json: None,
        hf_quant: None,
        precision: metrale_circuit::hardware::PrecisionChoice::Recipe,
    })?;
    let run = metrale_circuit::venn::Run { mode, rows };
    let one = metrale_circuit::hardware::plan_one(&reg, hw, &tree, &model, run)?;
    Ok(match action {
        CircuitAction::Display(d) => {
            let doc = metrale_circuit::hardware::display_on(&model, &one, &display_opts(d))?;
            paint(d, &doc)
        }
        _ => metrale_circuit::hardware::plan_text(&model.circuit, &one),
    })
}

/// 2026-10-05: The terminal drawing options `display` asks for.
fn display_opts(d: &CircuitDisplayArgs) -> DisplayOpts {
    let width = if std::io::stdout().is_terminal() {
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
    DisplayOpts {
        width,
        glyphs,
        expand,
    }
}

/// 2026-10-05: Colour a drawing as `--color`, the terminal and the environment allow.
fn paint(d: &CircuitDisplayArgs, doc: &metrale_circuit::display::Document) -> String {
    let depth = circuit_paint::resolve_depth(
        d.color,
        std::io::stdout().is_terminal(),
        std::env::var("NO_COLOR").ok().as_deref(),
        std::env::var("COLORTERM").ok().as_deref(),
    );
    circuit_paint::paint(doc, depth)
}

/// 2026-09-28: Write the view. A reader that closes early (`| head`) has what it asked for;
/// that is not a failure of the command.
fn emit(text: &str) -> Result<()> {
    match std::io::stdout().lock().write_all(text.as_bytes()) {
        Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
        other => Ok(other?),
    }
}

/// 2026-10-02: KERNEL_FAMILIES.toml per hardware, embedded beside FUSIONS.toml so `show` names
/// the compute unit of each group (`metrale_circuit::venn::compute`) as this binary declares it.
const FAMILIES: [(&str, &str); 1] = [(
    "gb10",
    include_str!("../../../../kernels/gb10/common/KERNEL_FAMILIES.toml"),
)];

/// 2026-10-02: The kernel families of the instance's hardware (the first `target` segment).
pub(crate) fn families_for(inst: &Instance) -> Result<metrale_circuit::venn::Families> {
    let hw = inst.target.split('/').next().unwrap_or_default();
    let text = FAMILIES
        .iter()
        .find(|(h, _)| *h == hw)
        .map(|(_, t)| *t)
        .with_context(|| format!("no KERNEL_FAMILIES.toml embedded for hardware `{hw}`"))?;
    Ok(metrale_circuit::venn::parse_families(text)?)
}

#[cfg(test)]
#[path = "circuit_tests.rs"]
mod circuit_tests;
