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

use anyhow::{Context, Result, anyhow, bail};
use metrale_circuit::display::{DisplayOpts, Expand, Glyphs};
use metrale_circuit::{ArchShape, AvailableKernels, Instance, LayerKind, Mode, Sources};

use super::{CircuitAction, CircuitArgs, CircuitMode, CircuitPlanArgs, circuit_paint};

/// 2026-09-28: kernels/circuits/INSTANCES.toml as built.
const INSTANCES: &str = include_str!("../../../../kernels/circuits/INSTANCES.toml");

/// 2026-09-28: Every circuit, block library, precision table and FUSIONS.toml an instance can
/// name, as built.
/// `every_instance_source_is_embedded_and_loads` fails when INSTANCES.toml names one missing here.
const CIRCUITS: [(&str, &str); 2] = [
    (
        "qwen3_5",
        include_str!("../../../../kernels/circuits/qwen3_5.toml"),
    ),
    (
        "qwen3_6_moe",
        include_str!("../../../../kernels/circuits/qwen3_6_moe.toml"),
    ),
];
const PRECISION: [(&str, &str); 2] = [
    (
        "qwen3.8-27b-nvfp4-unsloth",
        include_str!("../../../../kernels/circuits/precision/qwen3.8-27b-nvfp4-unsloth.toml"),
    ),
    (
        "qwen3.6-35b-a3b-fp8-bf16head",
        include_str!("../../../../kernels/circuits/precision/qwen3.6-35b-a3b-fp8-bf16head.toml"),
    ),
];
const BLOCKS: [(&str, &str); 1] = [(
    "qwen3_hybrid",
    include_str!("../../../../kernels/circuits/blocks/qwen3_hybrid.toml"),
)];
const FUSIONS: [(&str, &str); 1] = [(
    "gb10",
    include_str!("../../../../kernels/gb10/common/FUSIONS.toml"),
)];

fn lookup<'a>(table: &[(&str, &'a str)], key: &str, what: &str) -> Result<&'a str> {
    table
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, v)| *v)
        .ok_or_else(|| anyhow!("{what} `{key}` is not built into this binary"))
}

/// 2026-09-28: The embedded texts `instance` is built from.
pub(crate) fn sources(instance: &Instance) -> Result<Sources<'static>> {
    let hw = instance.target.split('/').next().unwrap_or_default();
    Ok(Sources {
        circuit: lookup(&CIRCUITS, &instance.arch, "circuit")?,
        precision: lookup(&PRECISION, &instance.precision, "precision table")?,
        rules: lookup(&FUSIONS, hw, "FUSIONS.toml for hardware")?,
        blocks: &BLOCKS,
    })
}

/// 2026-09-28: The instance serving `recipe`.
pub(crate) fn instance(recipe: &str) -> Result<Instance> {
    let all = metrale_circuit::parse_instances(INSTANCES)?;
    let known: Vec<String> = all.iter().map(|i| i.recipe.clone()).collect();
    all.into_iter().find(|i| i.recipe == recipe).ok_or_else(|| {
        anyhow!(
            "no circuit instance for recipe `{recipe}`; kernels/circuits/INSTANCES.toml has: {}",
            known.join(", ")
        )
    })
}

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

/// 2026-09-28: The arch shape of a parsed `config.json`, in the dim names the circuits read.
pub(crate) fn arch_shape(cfg: &metrale_config::ModelConfig) -> Result<ArchShape> {
    let mut layer_kinds = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        layer_kinds.push(match cfg.layer_type(i) {
            metrale_config::LayerType::LinearAttention => LayerKind::LinearAttention,
            metrale_config::LayerType::FullAttention => LayerKind::FullAttention,
            other => bail!("layer {i} is {other:?}, which no circuit models"),
        });
    }
    let dims = [
        ("hidden", cfg.hidden_size),
        ("inter", cfg.intermediate_size),
        ("vocab", cfg.vocab_size),
        ("q_heads", cfg.num_attention_heads),
        ("kv_heads", cfg.num_key_value_heads),
        ("head_dim", cfg.head_dim),
        ("lin_k_heads", cfg.linear_num_key_heads),
        ("lin_k_dim", cfg.linear_key_head_dim),
        ("lin_v_heads", cfg.linear_num_value_heads),
        ("lin_v_dim", cfg.linear_value_head_dim),
        ("experts", cfg.num_experts),
        ("top_k", cfg.num_experts_per_tok),
        ("moe_inter", cfg.moe_intermediate_size),
        ("shared_inter", cfg.shared_expert_intermediate_size),
    ]
    .into_iter()
    .filter(|(_, v)| *v > 0)
    .map(|(k, v)| (k.to_string(), v as u64))
    .collect();
    Ok(ArchShape { layer_kinds, dims })
}

/// 2026-09-28: Every way `from_config` disagrees with the instance's stated shape.
pub(crate) fn shape_drift(stated: &ArchShape, from_config: &ArchShape) -> Vec<String> {
    let mut out = Vec::new();
    if stated.layer_kinds != from_config.layer_kinds {
        out.push(format!(
            "layer kinds: INSTANCES.toml has {} layers, config.json {} (or the kinds differ)",
            stated.layer_kinds.len(),
            from_config.layer_kinds.len()
        ));
    }
    for (k, v) in &stated.dims {
        match from_config.dims.get(k) {
            Some(c) if c == v => {}
            other => out.push(format!("{k}: INSTANCES.toml {v}, config.json {other:?}")),
        }
    }
    out
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
