// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: The kernel Venn diagram of a target model against the models we already support
//! (`met circuit venn`): every target node, in every mode and row count asked for, classified
//! against the kernel families the compared models run, ranked by estimated share of step time.
//!
//! Classes, decided per node from the family manifest ([`families`]):
//! - **Shared**: a family the compared models run, at the same compile-time and policy point,
//!   with a microbench record at that point and row count.
//! - **Shared, unmeasured**: the same, without the record. Also a point the family already
//!   realises by an instantiation or a runtime branch while the compared models run another
//!   (the differing parameters are still listed).
//! - **Parameterization opportunity**: the same family where declared parameters differ and
//!   one of them is compile-time; the target point is missing or exists only as a file copy.
//! - **Policy variant**: the same, where every differing parameter is a policy.
//! - **Novel**: no family implements the op with the node's formats at that row count.
//!
//! Owner: metrale-circuit (venn).
//! Invariants:
//! - Pure: no I/O. The caller passes the circuits, plans, manifest and measurements.
//! - Deterministic: ties break by family order in the manifest, then by circuit order.
//! - A compared node whose plan kernel belongs to no family, and a parameter a node cannot
//!   supply, are typed errors: the report is never built from a guess.

pub mod checkpoint;
pub mod classify;
mod cli_args;
pub mod compute;
pub mod discover;
pub mod families;
pub mod measurements;
mod render;
mod render_summary;
pub mod repo;
pub mod report;
pub mod roofline;

use std::collections::BTreeMap;

use crate::fuser::FusionPlan;
use crate::ir::Circuit;
use crate::rules::Mode;

pub use cli_args::VennArgs;
pub use families::{Families, FamilyError, ParamKind, parse_families};
pub use measurements::{Measurements, parse_measurements};
pub use render::render;
pub use repo::{Repo, checkpoint_id_of, load_instance, report_text, resolve};
pub use report::{VennInputs, VennReport, build};

/// 2026-09-29: One model on either side of the diagram, at one mode and row count.
#[derive(Debug, Clone, Copy)]
pub struct Subject<'a> {
    /// 2026-09-29: Recipe id.
    pub recipe: &'a str,
    /// 2026-09-29: The instantiated circuit.
    pub circuit: &'a Circuit,
    /// 2026-09-29: The instance's policy settings (`kv_cache_dtype`, ...).
    pub settings: &'a BTreeMap<String, String>,
    /// 2026-09-29: Its fused plan at this mode and row count; `None` for a target no rule set
    /// covers yet, whose nodes are matched to families by op.
    pub plan: Option<&'a FusionPlan>,
}

/// 2026-09-29: A node's class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Class {
    /// 2026-09-29: Reuse; optimized at this point.
    Shared,
    /// 2026-09-29: Reuse, then microbench.
    SharedUnmeasured,
    /// 2026-09-29: A policy template (or a split).
    PolicyVariant,
    /// 2026-09-29: A template parameter (or a split), with the stability gate.
    ParameterizationOpportunity,
    /// 2026-09-29: Build it.
    Novel,
}

impl Class {
    /// 2026-09-29: The report spelling.
    pub fn name(self) -> &'static str {
        match self {
            Class::Shared => "Shared",
            Class::SharedUnmeasured => "Shared, unmeasured",
            Class::PolicyVariant => "Policy variant",
            Class::ParameterizationOpportunity => "Param. opportunity",
            Class::Novel => "Novel",
        }
    }
}

/// 2026-09-29: A parameter whose value differs from the compared point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diff {
    /// 2026-09-29: Parameter.
    pub param: String,
    /// 2026-09-29: Its kind.
    pub kind: ParamKind,
    /// 2026-09-29: The target's value.
    pub target: String,
    /// 2026-09-29: The compared value.
    pub other: String,
}

/// 2026-09-29: What a target node was compared with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Compared {
    /// 2026-09-29: A node a compared model runs, and the kernels its plan group launches.
    Model {
        /// 2026-09-29: Recipe.
        recipe: String,
        /// 2026-09-29: Node id.
        node: String,
        /// 2026-09-29: `module::function` list, or the emitter of a kernel-less group.
        kernels: String,
    },
    /// 2026-09-29: An instantiated point no compared model runs.
    Point {
        /// 2026-09-29: How it is realised.
        how: families::How,
        /// 2026-09-29: Its sources.
        files: Vec<String>,
    },
}

/// 2026-09-29: A node's classification against one family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// 2026-09-29: Family id.
    pub family: String,
    /// 2026-09-29: Class.
    pub class: Class,
    /// 2026-09-29: The target's point in the family.
    pub point: families::Values,
    /// 2026-09-29: The comparison.
    pub compared: Compared,
    /// 2026-09-29: Parameters that differ from the comparison, runtime ones included.
    pub diffs: Vec<Diff>,
    /// 2026-09-29: How the target point is realised when the family already has it.
    pub instantiated: Option<families::How>,
    /// 2026-09-29: Evidence records at the target point and row count.
    pub evidence: Vec<families::EvidenceSource>,
}

/// 2026-09-29: Why a Venn table could not be built.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VennError {
    /// 2026-09-29: A compared node whose plan group's kernels no family lists.
    #[error("{recipe}: node `{node}` ({op}) runs {kernels}, which no family in the manifest lists")]
    UnmappedKernel {
        /// 2026-09-29: Recipe.
        recipe: String,
        /// 2026-09-29: Node id.
        node: String,
        /// 2026-09-29: Op name.
        op: String,
        /// 2026-09-29: Kernels.
        kernels: String,
    },
    /// 2026-09-29: A parameter a node cannot supply, and no stated `absent` value.
    #[error("family `{family}` parameter `{param}`: node `{node}` gives no value")]
    MissingValue {
        /// 2026-09-29: Family id.
        family: String,
        /// 2026-09-29: Parameter.
        param: String,
        /// 2026-09-29: Node id.
        node: String,
    },
    /// 2026-09-29: An evidence record naming a measurements.toml row that does not exist.
    #[error("family `{family}` cites measurement `{key}`, which measurements.toml does not have")]
    UnknownMeasurement {
        /// 2026-09-29: Family id.
        family: String,
        /// 2026-09-29: The key.
        key: String,
    },
    /// 2026-09-29: A node the roofline cannot estimate.
    #[error(transparent)]
    Cost(#[from] roofline::CostError),
    /// 2026-09-29: A mode the target circuit has no section for, or a row count it cannot run.
    #[error("{0}")]
    Run(String),
    /// 2026-09-29: Loading or fusing a subject.
    #[error("{0}")]
    Load(String),
    /// 2026-09-29: The family manifest disagrees with the kernel sources.
    #[error("KERNEL_FAMILIES.toml has drifted from the sources:\n  {}", .0.join("\n  "))]
    Drift(Vec<String>),
    /// 2026-09-29: The target instance disagrees with the checkpoint directory it was named by.
    #[error("{0}")]
    Checkpoint(String),
}

/// 2026-09-29: One mode and row count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Run {
    /// 2026-09-29: Mode.
    pub mode: Mode,
    /// 2026-09-29: Padded rows.
    pub rows: u64,
}

#[cfg(test)]
mod venn_tests;
