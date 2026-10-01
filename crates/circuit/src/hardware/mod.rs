// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The hardware axis of the circuit (`met circuit plan --hardware <device>`): point a
//! model and a target device at it and get the fused plan the device's kernel class can run,
//! the gap report against that class's kernel set ranked by the device's roofline, decode and
//! prefill estimates, and a memory-fit check.
//!
//! - [`device`]: `kernels/DEVICES.toml`, the SKU registry (roofline, native instruction kinds).
//! - [`class`]: a class's chain, rules and families, inherited along `HARDWARE.toml` `inherits`.
//! - [`avail`]: which kernels the device can run, read from the sources' guards.
//! - [`exec`]: how a node's declared formats execute on the device (never a silent upcast).
//! - [`plan`] / [`gaps`] / [`estimate`]: fuse with gaps marked, classify, estimate.
//! - [`model`]: the model seam ([`model::CircuitSource`]).
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - Pure: every file arrives through [`sources::KernelTree`] (SBIO); nothing reads the disk,
//!   the environment or a clock.
//! - Deterministic: the same inputs render the same report bytes (the `--check` tests rely on
//!   it).

pub mod avail;
pub mod class;
pub mod device;
pub mod estimate;
pub mod exec;
pub mod gaps;
pub mod model;
pub mod model_checkpoint;
pub mod plan;
mod render;
mod render_matrix;
mod render_routes;
pub mod sources;

#[cfg(test)]
mod fp4_costing_tests;
#[cfg(test)]
mod hardware_tests;
#[cfg(test)]
#[path = "runtime_tests.rs"]
mod runtime_tests;
#[cfg(test)]
mod test_fixture;

use std::collections::BTreeMap;

use crate::fuser::Policy;
use crate::render::Header;
use crate::rules::KernelId;
use crate::venn::Class;
use crate::venn::roofline::CostError;

pub use device::{Device, Registry, parse_devices};
pub use model::{CircuitSource, InstancesSource, ModelSpec, ModelUnderPlan, PrecisionChoice};
pub use model_checkpoint::CheckpointSource;
pub use render::render_report;
pub use render_matrix::{PortList, port_lists, summary_row};
pub use render_routes::plan_text;
pub use sources::{ClassSources, KernelTree, Module};

/// 2026-09-30: Why a hardware plan could not be built.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum HwError {
    /// 2026-09-30: `kernels/DEVICES.toml`.
    #[error("DEVICES.toml: {0}")]
    Registry(String),
    /// 2026-09-30: A device id the registry does not list.
    #[error("unknown device `{id}` (kernels/DEVICES.toml lists: {})", .known.join(", "))]
    UnknownDevice {
        /// 2026-09-30: The id asked for.
        id: String,
        /// 2026-09-30: The ids listed.
        known: Vec<String>,
    },
    /// 2026-09-30: A class's HARDWARE.toml, FUSIONS.toml, KERNEL_FAMILIES.toml or sources.
    #[error("{0}")]
    Class(String),
    /// 2026-09-30: The class's build flags and the device's instruction kinds disagree.
    #[error(
        "kernels/{class}/HARDWARE.toml {} -D{macro_name}, but device `{device}` {} {requires}: \
         the build and the registry contradict each other",
        if *.defined { "defines" } else { "does not define" },
        if *.defined { "claims" } else { "lacks" }
    )]
    BuildContradictsDevice {
        /// 2026-09-30: Device id.
        device: String,
        /// 2026-09-30: Class.
        class: String,
        /// 2026-09-30: Guard macro.
        macro_name: String,
        /// 2026-09-30: The instruction kind it stands for.
        requires: String,
        /// 2026-09-30: The class defines the macro.
        defined: bool,
    },
    /// 2026-09-30: The model.
    #[error("{0}")]
    Model(String),
    /// 2026-09-30: Fusing.
    #[error("{0}")]
    Plan(String),
    /// 2026-09-30: The roofline.
    #[error(transparent)]
    Cost(#[from] CostError),
}

/// 2026-09-30: A built hardware report.
#[derive(Debug, Clone)]
pub struct HwReport {
    /// 2026-09-30: The model.
    pub model: ModelUnderPlan,
    /// 2026-09-30: The device's class, resolved.
    pub resolved: plan::Resolved,
    /// 2026-09-30: The policy planned with.
    pub policy: Policy,
    /// 2026-09-30: Settings the class's defaults changed from the model's source.
    pub class_settings: BTreeMap<String, String>,
    /// 2026-09-30: Plan header.
    pub header: Header,
    /// 2026-09-30: One table per report run ([`plan::report_runs`]).
    pub tables: Vec<gaps::GapTable>,
    /// 2026-09-30: The runtime routes that apply to those runs, estimated.
    pub routes: Vec<render_routes::RouteRow>,
    /// 2026-09-30: Prefill estimates: (tokens, microseconds).
    pub prefill: Vec<(u64, f64)>,
    /// 2026-09-30: Weights, KV and state.
    pub footprint: estimate::Footprint,
    /// 2026-09-30: Weight bytes one decode row reads.
    pub weight_floor: f64,
    /// 2026-09-30: (declared pair, weight, activation, execution) to node count.
    pub exec: BTreeMap<(String, String, String, exec::Exec), usize>,
    /// 2026-09-30: Kernels of the rules this model's ops and policy could select that the
    /// device cannot run.
    pub absent: Vec<(KernelId, avail::Absence)>,
    /// 2026-09-30: Every registry device's usable bytes, for the "does not fit" advice.
    pub others: Vec<(String, f64)>,
    /// 2026-09-30: The regenerating command.
    pub command: String,
}

impl HwReport {
    /// 2026-09-30: Share of the step at `table` covered by shared kernels (measured, or any).
    pub fn covered(&self, table: usize, measured_only: bool) -> f64 {
        let classes: &[Class] = if measured_only {
            &[Class::Shared]
        } else {
            &[Class::Shared, Class::SharedUnmeasured]
        };
        self.tables.get(table).map_or(0.0, |t| t.share_of(classes))
    }
}

/// 2026-09-30: One fused plan of `model` on `device_id` at `run`, with the policy and plan
/// header it was made under: the plan `met circuit plan --format plan` prints and
/// `met circuit display --hardware` draws.
#[derive(Debug, Clone)]
pub struct OnePlan {
    /// 2026-09-30: The device's class, resolved.
    pub resolved: plan::Resolved,
    /// 2026-09-30: The policy after the class's defaults.
    pub policy: Policy,
    /// 2026-09-30: The plan header.
    pub header: Header,
    /// 2026-09-30: The plan.
    pub planned: plan::Planned,
}

/// 2026-09-30: Plan `model` on `device_id` at `run`.
pub fn plan_one(
    registry: &Registry,
    device_id: &str,
    tree: &dyn KernelTree,
    model: &ModelUnderPlan,
    run: crate::venn::Run,
) -> Result<OnePlan, HwError> {
    let resolved = plan::resolve(registry, device_id, tree, model)?;
    let (policy, _) = model::policy_on_class(
        &model.policy,
        model.settings_class.as_deref(),
        &resolved.chain[0],
    )?;
    let header =
        crate::render::with_settings(&model::header_on(model, &resolved.device.class), &policy);
    let planned = plan::fuse_on(&resolved, &model.circuit, &policy, run)?;
    Ok(OnePlan {
        resolved,
        policy,
        header,
        planned,
    })
}

/// 2026-09-30: Draw `one` as `met circuit display` does (placeholder groups show their
/// `novel` emitter).
pub fn display_on(
    model: &ModelUnderPlan,
    one: &OnePlan,
    opts: &crate::display::DisplayOpts,
) -> Result<crate::display::Document, crate::LoadError> {
    let plan = &one.planned.plan;
    let buffers = crate::planner::plan_buffers(&model.circuit, plan, plan.rows)?;
    let info = crate::display::DisplayInfo {
        checkpoint: model.checkpoint.clone(),
        recipe: format!("{} on {}", model.label, one.resolved.device.id),
        bytes: Some((buffers.materialized_bytes, buffers.arena_bytes)),
    };
    Ok(crate::display::display(&model.circuit, plan, &info, opts)?)
}

/// 2026-09-30: Some node of `c` has the op (and, for a linear, a role) the rule's first pattern
/// op names: the rule could apply to this model at all.
fn heads_a_node(r: &crate::rules::Rule, c: &crate::ir::Circuit) -> bool {
    r.pattern.first().is_some_and(|p| {
        c.nodes.iter().any(|n| match n.op {
            crate::ir::OpKind::Linear(role) if !p.roles.is_empty() => p.roles.contains(&role),
            op => op == p.op,
        })
    })
}

/// 2026-09-30: Prompt lengths every report estimates.
pub const PREFILL_TOKENS: [u64; 2] = [4096, 32768];

/// 2026-09-30: Build the report of `model` on `device_id`.
pub fn build_report(
    registry: &Registry,
    device_id: &str,
    tree: &dyn KernelTree,
    model: ModelUnderPlan,
    command: String,
) -> Result<HwReport, HwError> {
    let resolved = plan::resolve(registry, device_id, tree, &model)?;
    let (policy, class_settings) = model::policy_on_class(
        &model.policy,
        model.settings_class.as_deref(),
        &resolved.chain[0],
    )?;
    let header =
        crate::render::with_settings(&model::header_on(&model, &resolved.device.class), &policy);
    let c = &model.circuit;
    let mut tables = Vec::new();
    for run in plan::report_runs() {
        let planned = plan::fuse_on(&resolved, c, &policy, run)?;
        tables.push(gaps::gap_table(
            &resolved,
            c,
            &policy.settings,
            &model.label,
            planned,
        )?);
    }
    let routes = render_routes::route_rows(&resolved, c, &policy, &model.label, &tables)?;
    let rf = |n| resolved.roofline_of(n);
    let prefill = PREFILL_TOKENS
        .iter()
        .map(|&t| Ok((t, estimate::prefill_us(c, &policy.settings, &rf, t)?)))
        .collect::<Result<Vec<_>, HwError>>()?;
    let footprint = estimate::footprint(c, &policy.settings).map_err(HwError::Model)?;
    let weight_floor = estimate::weight_floor_bytes(c).map_err(HwError::Model)?;
    let mut exec = BTreeMap::new();
    for n in &c.nodes {
        let (Some(w), Some(a)) = (n.weight, estimate::activation_of(c, n)) else {
            continue;
        };
        let e = exec::exec_of(&resolved.device, w, a);
        *exec
            .entry((exec::pair_name(w, a), w.name(), a.name(), e))
            .or_insert(0) += 1;
    }
    let absent = resolved
        .rule_list()
        .iter()
        .filter(|r| {
            r.when
                .iter()
                .all(|(k, v)| policy.settings.get(k) == Some(v))
                && heads_a_node(r, c)
        })
        .flat_map(|r| r.kernels.iter())
        .filter_map(|k| {
            resolved
                .availability
                .absent
                .get(k)
                .map(|a| (k.clone(), a.clone()))
        })
        .collect::<BTreeMap<_, _>>()
        .into_iter()
        .collect();
    Ok(HwReport {
        model,
        resolved,
        policy,
        class_settings,
        header,
        tables,
        routes,
        prefill,
        footprint,
        weight_floor,
        exec,
        absent,
        others: registry
            .devices
            .iter()
            .map(|d| (d.id.clone(), d.memory_bytes * d.usable_fraction))
            .collect(),
        command,
    })
}
