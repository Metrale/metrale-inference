// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Architecture circuits. A model is a graph of ops (`kernels/circuits/<arch>.toml`)
//! whose edges the fuser marks fused or materialised by applying the hardware's fusion rules
//! (`kernels/<hw>/common/FUSIONS.toml`) for one mode and row count.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Pure: no GPU, no file I/O, no environment, no clock and no randomness. Callers read the
//!   TOMLs and pass their text in ([`load`]).
//! - A plan is deterministic, and its digest is recorded beside the closure hash.

pub mod checkpoint;
pub mod circuit_toml;
pub mod config_map;
pub mod declared_precision;
pub mod digest;
pub mod dims;
pub mod display;
pub mod format;
pub mod fuser;
pub mod hardware;
pub mod instances;
pub mod instantiate;
pub mod ir;
pub mod model_buffer;
pub mod planner;
pub mod precision;
pub mod precision_policy;
pub mod render;
pub mod rules;
pub mod runtime;
pub mod state;
pub mod state_ops;
pub mod venn;

#[cfg(test)]
mod test_toy;

pub use checkpoint::{
    CheckpointError, QuantMetadata, ResolvedCheckpoint, ServePrecision,
    instantiate_from_checkpoint, map_checkpoint, recurrent_states, resolve_checkpoint,
};
pub use circuit_toml::{CircuitError, includes_of};
pub use format::{Format, Scale};
pub use fuser::{AvailableKernels, EdgeState, FuseError, FusionPlan, Group, Policy, fuse};
pub use instances::{Instance, InstanceError, PrecisionSpec, parse_instances};
pub use instantiate::instantiate;
pub use ir::{ArchShape, Circuit, LayerKind, LinearRole, OpKind, Section};
pub use precision::{EdgePrecision, LinearFormats, PrecisionError, PrecisionTable};
pub use rules::{KernelId, Mode, Numerics, Rule, RuleError, parse_rules};
pub use runtime::{RuleSet, RuntimeRoute, parse_rule_set};

/// 2026-09-28: Any failure between the TOML texts and a rendered plan.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LoadError {
    /// 2026-09-28: The circuit.
    #[error(transparent)]
    Circuit(#[from] CircuitError),
    /// 2026-09-28: FUSIONS.toml.
    #[error(transparent)]
    Rules(#[from] RuleError),
    /// 2026-09-28: The precision table.
    #[error(transparent)]
    Precision(#[from] PrecisionError),
    /// 2026-09-28: Fusion.
    #[error(transparent)]
    Fuse(#[from] FuseError),
    /// 2026-09-28: The display renderer.
    #[error(transparent)]
    Display(#[from] display::DisplayError),
    /// 2026-09-28: The buffer planner.
    #[error(transparent)]
    Plan(#[from] planner::PlanError),
    /// 2026-09-28: The precision table describes another checkpoint.
    #[error("precision table is for `{table}`, the instance serves `{instance}`")]
    CheckpointMismatch {
        /// 2026-09-28: The table's checkpoint.
        table: String,
        /// 2026-09-28: The instance's checkpoint.
        instance: String,
    },
}

/// 2026-09-28: The texts one instance is built from.
#[derive(Debug, Clone, Copy)]
pub struct Sources<'a> {
    /// 2026-09-28: `kernels/circuits/<arch>.toml`.
    pub circuit: &'a str,
    /// 2026-09-28: The file the instance's [`PrecisionSpec`] names: a precision table or a
    /// checkpoint plan fixture (`kernels/circuits/checkpoints/<name>.toml`).
    pub precision: &'a str,
    /// 2026-09-28: `kernels/<hw>/common/FUSIONS.toml`.
    pub rules: &'a str,
    /// 2026-09-28: The block libraries the circuit includes, by name:
    /// `kernels/circuits/blocks/<name>.toml`.
    pub blocks: &'a [(&'a str, &'a str)],
}

/// 2026-09-28: An instance's circuit and rules, ready to fuse.
#[derive(Debug, Clone)]
pub struct Loaded {
    /// 2026-09-28: The instantiated circuit.
    pub circuit: Circuit,
    /// 2026-09-28: The rules, in file order.
    pub rules: Vec<Rule>,
    /// 2026-09-30: The runtime routes, in file order.
    pub runtime: Vec<RuntimeRoute>,
    /// 2026-09-28: SHA-256 of the FUSIONS.toml text ([`digest::rules_digest`]).
    pub rules_digest: String,
}

/// 2026-09-28: Parse and instantiate `instance` from `src`, with the precision its spec names.
pub fn load(instance: &Instance, src: Sources<'_>) -> Result<Loaded, LoadError> {
    let mismatch = |table: String| LoadError::CheckpointMismatch {
        table,
        instance: instance.checkpoint.clone(),
    };
    match &instance.precision {
        PrecisionSpec::Table(_) => {
            let table = PrecisionTable::parse(src.precision)?;
            if table.checkpoint != instance.checkpoint {
                return Err(mismatch(table.checkpoint));
            }
            load_with(instance, src, &table)
        }
        PrecisionSpec::Policy {
            tier, caps, engine, ..
        } => {
            let plan = precision_policy::CheckpointPlan::parse(src.precision)?;
            if plan.checkpoint != instance.checkpoint {
                return Err(mismatch(plan.checkpoint));
            }
            let policy = metrale_config::WeightQuantPolicy::new(
                precision_policy::tier_named(tier)?,
                &plan.plan,
                precision_policy::caps_named(caps)?,
            );
            load_with(
                instance,
                src,
                &precision_policy::PolicyPrecision::new(policy, engine),
            )
        }
    }
}

/// 2026-09-28: Parse and instantiate `instance` from `src` with `precision`: the executor
/// passes the served model's own policy here, the checked-in fixture being only the offline
/// copy of it.
pub fn load_with(
    instance: &Instance,
    src: Sources<'_>,
    precision: &dyn EdgePrecision,
) -> Result<Loaded, LoadError> {
    let circuit = instantiate(src.circuit, src.blocks, &instance.shape, precision)?;
    let RuleSet { rules, runtime } = parse_rule_set(src.rules)?;
    Ok(Loaded {
        circuit,
        rules,
        runtime,
        rules_digest: digest::rules_digest(src.rules),
    })
}

/// 2026-09-28: The header lines a rendering of `instance` carries.
pub fn header(instance: &Instance) -> render::Header {
    let mut h = vec![
        ("recipe".to_string(), instance.recipe.clone()),
        ("checkpoint".to_string(), instance.checkpoint.clone()),
        ("target".to_string(), instance.target.clone()),
    ];
    let settings: Vec<String> = instance
        .policy
        .settings
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    h.push(("settings".to_string(), settings.join(" ")));
    let levers: Vec<&str> = instance
        .policy
        .opt_in_levers
        .iter()
        .map(String::as_str)
        .collect();
    let levers = if levers.is_empty() {
        "none".to_string()
    } else {
        levers.join(" ")
    };
    h.push(("opt-in levers".to_string(), levers));
    h
}

/// 2026-09-28: Fuse and render one plan of `instance`.
pub fn render_plan(
    instance: &Instance,
    loaded: &Loaded,
    available: &AvailableKernels,
    mode: Mode,
    rows: u64,
) -> Result<String, LoadError> {
    let plan = fuse(
        &loaded.circuit,
        &loaded.rules,
        available,
        &instance.policy,
        mode,
        rows,
    )?;
    let head = header(instance);
    let mut text = render::render(&loaded.circuit, &plan, &head);
    let set = (loaded.rules.as_slice(), loaded.runtime.as_slice());
    for (route, arm) in
        runtime::route_arms(&loaded.circuit, set, available, &instance.policy, &plan)?
    {
        let h = render::with_settings(&head, &route.policy(&instance.policy));
        text.push_str(&render::route_section(&loaded.circuit, &route, &arm, &h));
    }
    Ok(text)
}

/// 2026-09-28: Fuse, lay out and draw one plan of `instance`: the view `met circuit display`
/// prints, over the same plan [`render_plan`] renders.
pub fn display_plan(
    instance: &Instance,
    loaded: &Loaded,
    available: &AvailableKernels,
    mode: Mode,
    rows: u64,
    opts: &display::DisplayOpts,
) -> Result<display::Document, LoadError> {
    let plan = fuse(
        &loaded.circuit,
        &loaded.rules,
        available,
        &instance.policy,
        mode,
        rows,
    )?;
    let buffers = planner::plan_buffers(&loaded.circuit, &plan, rows)?;
    let info = display::DisplayInfo {
        checkpoint: instance.checkpoint.clone(),
        recipe: instance.recipe.clone(),
        bytes: Some((buffers.materialized_bytes, buffers.arena_bytes)),
    };
    Ok(display::display(&loaded.circuit, &plan, &info, opts)?)
}
