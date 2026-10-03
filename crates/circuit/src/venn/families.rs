// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: The kernel-family manifest, `kernels/<hw>/common/KERNEL_FAMILIES.toml`: which
//! circuit ops each family implements, the parameters it varies (runtime, compile-time or
//! policy), the points it instantiates, the evidence envelope (the points with microbench
//! records) and the rules that rediscover the instantiated points from the kernel sources.
//!
//! Owner: metrale-circuit (venn).
//! Invariants:
//! - A parameter is an axis the family's code is structured to vary (a runtime argument, a
//!   template or macro parameter, a policy or a per-point file copy), even when only one value
//!   is instantiated. A format the code cannot vary without a rewrite is a constraint on the
//!   op (`[[family.op]]`), not a parameter.
//! - Every point states every compile-time and policy parameter, and nothing else: a runtime
//!   parameter never defines a point. Evidence sits on an instantiated point.
//! - Nothing defaults: an op, format, parameter, extractor or mode the loader does not know is a
//!   typed error, and a parameter a node cannot supply is an error unless the manifest states
//!   its `absent` value.

use std::collections::{BTreeMap, BTreeSet};

use super::compute::{ComputeUnit, FamilyCompute};
use crate::format::Format;
use crate::ir::{LayerKind, LinearRole, OpKind};
use crate::rules::{KernelId, Mode};

/// 2026-09-29: How a parameter reaches the kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ParamKind {
    /// 2026-09-29: A kernel argument: strides, counts, eps, scale pointers. Never defines a point.
    Runtime,
    /// 2026-09-29: Sizes registers, shared memory or unrolling: a template parameter or a
    /// per-point copy (head_dim, group size, tiles).
    Compile,
    /// 2026-09-29: A policy the code plugs in: weight format, scale layout, activation
    /// quantizer, activation epilogue, routing scoring.
    Policy,
}

impl ParamKind {
    /// 2026-09-29: The manifest and report spelling.
    pub fn name(self) -> &'static str {
        match self {
            ParamKind::Runtime => "runtime",
            ParamKind::Compile => "compile-time",
            ParamKind::Policy => "policy",
        }
    }
}

/// 2026-09-29: Where a parameter's value is read from, for one node of one circuit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Extract {
    /// 2026-09-29: `dim:<name>`: the circuit's dim.
    Dim(String),
    /// 2026-09-29: `weight`: the node's weight format.
    Weight,
    /// 2026-09-29: `activation`: the format of the node's first input.
    Activation,
    /// 2026-09-29: `output`: the format of the node's first output.
    Output,
    /// 2026-09-29: `op`: the node's op name.
    Op,
    /// 2026-09-29: `consumer_op`: the op of the first node reading the node's first output.
    ConsumerOp,
    /// 2026-09-29: `param:<key>`: the node's template parameter.
    Param(String),
    /// 2026-09-29: `setting:<key>`: the instance's policy setting.
    Setting(String),
    /// 2026-09-29: `in_dim` / `out_dim`: the first input's / output's feature dimension.
    InDim,
    /// 2026-09-29: See [`Extract::InDim`].
    OutDim,
}

impl Extract {
    fn parse(s: &str) -> Option<Self> {
        let (head, arg) = match s.split_once(':') {
            Some((h, a)) if !a.is_empty() => (h, Some(a.to_string())),
            Some(_) => return None,
            None => (s, None),
        };
        match (head, arg) {
            ("dim", Some(a)) => Some(Extract::Dim(a)),
            ("param", Some(a)) => Some(Extract::Param(a)),
            ("setting", Some(a)) => Some(Extract::Setting(a)),
            ("weight", None) => Some(Extract::Weight),
            ("activation", None) => Some(Extract::Activation),
            ("output", None) => Some(Extract::Output),
            ("op", None) => Some(Extract::Op),
            ("consumer_op", None) => Some(Extract::ConsumerOp),
            ("in_dim", None) => Some(Extract::InDim),
            ("out_dim", None) => Some(Extract::OutDim),
            _ => None,
        }
    }
}

/// 2026-09-29: One parameter of a family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    /// 2026-09-29: Name, e.g. `head_dim`.
    pub name: String,
    /// 2026-09-29: How it reaches the kernel.
    pub kind: ParamKind,
    /// 2026-09-29: Op base name to extractor. Ops of the family missing here do not carry it.
    pub from: BTreeMap<String, Extract>,
    /// 2026-09-29: The value when the extractor finds nothing (a dim or parameter the circuit
    /// does not state), with the manifest's reason in a comment; `None` makes that an error.
    pub absent: Option<String>,
}

/// 2026-09-29: An op a family implements, with the constraints a node must meet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpSpec {
    /// 2026-09-29: Op base name (`linear`, `paged_attention`, ...). 2026-10-02: or a quantizer
    /// with its output format (`act_quant:nvfp4/g16`).
    pub op: String,
    /// 2026-09-29: For `linear`: the roles it takes; empty is every role.
    pub roles: BTreeSet<LinearRole>,
    /// 2026-09-29: Allowed weight formats; empty is any.
    pub weight: BTreeSet<Format>,
    /// 2026-09-29: Allowed first-input formats; empty is any.
    pub activation: BTreeSet<Format>,
    /// 2026-09-29: Node template parameters that must hold these values.
    pub params: Values,
    /// 2026-09-29: Ops one of which must read the node's first output, by base name
    /// (`l2_norm`) or qualified name (`linear:q`); empty is no constraint. A kernel that fuses
    /// its consumer (a conv with its L2 norm) implements the op only where that consumer
    /// follows.
    pub feeds: BTreeSet<String>,
    /// 2026-09-29: Ops one of which must produce the node's first input, spelled as `feeds`;
    /// empty is no constraint (a snapshot of the conv window, not of the SSM state).
    pub after: BTreeSet<String>,
    /// 2026-09-29: Ops one of which must also read the node's first input, spelled as
    /// `feeds`; empty is no constraint. A conv kernel that writes its window snapshot and the
    /// L2 norm of its output implements the snapshot only where the L2 norm reads beside it.
    pub beside: BTreeSet<String>,
}

impl OpSpec {
    /// 2026-10-02: The spec names `op`: by base name, or by its qualified name (an `act_quant`
    /// with its format).
    pub fn names(&self, op: &OpKind) -> bool {
        self.op == op.base_name() || self.op == op.name()
    }
}

/// 2026-09-29: How an instantiated point is realised in the sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum How {
    /// 2026-09-29: A template or macro instantiation.
    Instantiation,
    /// 2026-09-29: A per-point copy of the source file (the parameterization target).
    Copy,
    /// 2026-09-29: A runtime branch inside one entry point.
    Branch,
}

impl How {
    /// 2026-09-29: The manifest spelling.
    pub fn name(self) -> &'static str {
        match self {
            How::Instantiation => "instantiation",
            How::Copy => "copy",
            How::Branch => "branch",
        }
    }
}

/// 2026-09-29: Parameter name to value.
pub type Values = BTreeMap<String, String>;

/// 2026-09-29: One instantiated point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Point {
    /// 2026-09-29: Every compile-time and policy parameter's value.
    pub values: Values,
    /// 2026-09-29: How it is realised.
    pub how: How,
    /// 2026-09-29: Repo-relative sources that realise it.
    pub files: Vec<String>,
    /// 2026-10-02: Its compute unit where it differs from the family's (`None`: the family's).
    pub compute: Option<ComputeUnit>,
    /// 2026-10-02: Its own pipeline declarations, where they differ from the family's
    /// ([`crate::pipeline::declare`]).
    pub pipeline: crate::pipeline::declare::ByOp,
}

/// 2026-09-29: Where an evidence record lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceSource {
    /// 2026-09-29: A `docs/kernel-perf/measurements.toml` row, keyed `<kernel> @ <regime>`.
    Measurement(String),
    /// 2026-09-29: A microbench not in measurements.toml: its method and result in words.
    Microbench(String),
}

/// 2026-09-29: A microbench record at one point and the row counts it measured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Evidence {
    /// 2026-09-29: The instantiated point measured.
    pub point: Values,
    /// 2026-09-29: The row counts measured; a record says nothing about any other count.
    pub rows: BTreeSet<u64>,
    /// 2026-09-29: The record.
    pub source: EvidenceSource,
}

/// 2026-09-29: A rule that rediscovers instantiated points from the sources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Discover {
    /// 2026-09-29: Each file matching `glob` realises `values` as a copy.
    File {
        /// 2026-09-29: Repo-relative glob (`*` matches any run).
        glob: String,
        /// 2026-09-29: The point's values.
        values: Values,
    },
    /// 2026-09-29: Each `name(...)` invocation line in `file` realises one point: parameter
    /// `p` is argument `args[p]` (0-based), mapped through `map[p]` when present.
    Macro {
        /// 2026-09-29: Repo-relative source.
        file: String,
        /// 2026-09-29: Macro or template name.
        name: String,
        /// 2026-09-29: Parameter to argument index.
        args: BTreeMap<String, usize>,
        /// 2026-09-29: Parameter to (argument text to value).
        map: BTreeMap<String, BTreeMap<String, String>>,
    },
}

/// 2026-09-29: One kernel family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Family {
    /// 2026-09-29: Id.
    pub id: String,
    /// 2026-09-29: One line.
    pub description: String,
    /// 2026-09-29: Entry points, as FUSIONS.toml names them.
    pub kernels: Vec<KernelId>,
    /// 2026-09-29: FUSIONS.toml emitters of kernel-less groups (host copies) it stands for.
    pub emitters: Vec<String>,
    /// 2026-09-29: Inclusive range of rows one launch covers.
    pub rows: (u64, u64),
    /// 2026-10-03: The plan modes it runs in; empty for every mode.
    pub modes: std::collections::BTreeSet<crate::rules::Mode>,
    /// 2026-09-29: Ops it implements.
    pub ops: Vec<OpSpec>,
    /// 2026-09-29: Its parameter space.
    pub params: Vec<Param>,
    /// 2026-09-29: Instantiated points.
    pub points: Vec<Point>,
    /// 2026-09-29: The evidence envelope.
    pub evidence: Vec<Evidence>,
    /// 2026-09-29: Discovery rules.
    pub discover: Vec<Discover>,
    /// 2026-10-02: The compute units it runs on ([`super::compute`]).
    pub compute: FamilyCompute,
    /// 2026-10-02: The numeric pipeline each op runs at, the family's and its kernels'
    /// ([`crate::pipeline::declare`]).
    pub pipeline: crate::pipeline::declare::FamilyPipelines,
    /// 2026-10-02: Device scratch a launch needs beyond its edges (`crate::memory`).
    pub workspace: Vec<Workspace>,
}

/// 2026-10-02: One workspace of a family (`[[family.workspace]]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Workspace {
    /// 2026-10-02: Name, unique in the family.
    pub name: String,
    /// 2026-10-02: Byte expressions over the circuit's dims, `n` (rows), `k` (the node's input
    /// width) and `sm_count`; the largest is the workspace.
    pub bytes: Vec<crate::dims::DimExpr>,
    /// 2026-10-02: Held inside the legacy buffer arena.
    pub arena: bool,
    /// 2026-10-02: Where the engine allocates it, and why it has this size.
    pub why: String,
}

impl Family {
    /// 2026-10-03: It runs only in prefill modes, which the Venn and the hardware gap report do
    /// not classify yet (their runs are decode, multi-sequence, verify and draft).
    pub fn prefill_only(&self) -> bool {
        !self.modes.is_empty() && self.modes.iter().all(|m| m.is_prefill())
    }

    /// 2026-09-29: One launch covers more than one row.
    pub fn multi_row(&self) -> bool {
        self.rows.1 > 1
    }

    /// 2026-09-29: The parameter named `name`.
    pub fn param(&self, name: &str) -> Option<&Param> {
        self.params.iter().find(|p| p.name == name)
    }

    /// 2026-10-02: Each point's values with its own pipeline declarations.
    pub fn point_pipelines(&self) -> Vec<(&Values, &crate::pipeline::declare::ByOp)> {
        self.points
            .iter()
            .map(|p| (&p.values, &p.pipeline))
            .collect()
    }
}

/// 2026-09-29: A fact about a legacy layer implementation the circuit cannot show: a mode in
/// which the layer has no multi-row path and loops per sequence (or per row).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacyPath {
    /// 2026-09-29: Circuit arch.
    pub arch: String,
    /// 2026-09-29: Layer kind.
    pub layer_kind: LayerKind,
    /// 2026-09-29: Modes whose rows above one loop per sequence.
    pub per_sequence: Vec<Mode>,
    /// 2026-09-29: The sites (`block.node`) that loop; empty is every node of the layer kind.
    pub sites: Vec<String>,
    /// 2026-09-29: The loop runs only above this many rows (a batched path covers the rest).
    pub rows_above: u64,
    /// 2026-09-29: `path:line` of the loop or the trait default it falls back to.
    pub cite: String,
    /// 2026-09-29: Text the cited line holds (checked by the tests).
    pub holds: String,
    /// 2026-09-29: One line.
    pub note: String,
}

/// 2026-09-29: The roofline constants of the hardware.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Roofline {
    /// 2026-09-29: DRAM bandwidth, GB/s.
    pub dram_gbps: f64,
    /// 2026-09-29: Dense BF16 tensor peak, TFLOPS.
    pub bf16_tflops: f64,
    /// 2026-09-29: FP8 tensor peak, TFLOPS.
    pub fp8_tflops: f64,
    /// 2026-09-29: NVFP4 tensor peak, TFLOPS.
    pub nvfp4_tflops: f64,
    /// 2026-09-29: KV context each sequence attends over in the estimate.
    pub context_tokens: u64,
}

/// 2026-09-29: A parsed manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct Families {
    /// 2026-09-29: Hardware directory.
    pub hardware: String,
    /// 2026-09-29: Estimate constants.
    pub roofline: Roofline,
    /// 2026-09-29: Families, in file order.
    pub families: Vec<Family>,
    /// 2026-09-29: Legacy per-sequence facts.
    pub legacy: Vec<LegacyPath>,
}

impl Families {
    /// 2026-10-02: The compute unit `kernel` runs on: its family's answer; `None` when no
    /// family lists it.
    pub fn compute_of(&self, kernel: &KernelId) -> Option<&ComputeUnit> {
        self.families
            .iter()
            .find(|f| f.kernels.contains(kernel))
            .map(|f| f.compute.of(kernel))
    }

    /// 2026-09-29: The family that ran a node of `op` in a plan group launching `kernels`
    /// (or, for a kernel-less group, emitted by `emitter`): the first, in manifest order, that
    /// lists one of them and implements the op (and, for `linear`, the role). Formats are not
    /// checked: a plan runs what its rules chose. `None` when no family lists them.
    pub fn of_group(&self, kernels: &[KernelId], emitter: &str, op: &OpKind) -> Option<&Family> {
        self.families.iter().find(|f| {
            let listed = if kernels.is_empty() {
                f.emitters.iter().any(|e| e == emitter)
            } else {
                kernels.iter().any(|k| f.kernels.contains(k))
            };
            listed
                && f.ops.iter().any(|s| {
                    s.names(op)
                        && match op {
                            OpKind::Linear(r) => s.roles.is_empty() || s.roles.contains(r),
                            _ => true,
                        }
                })
        })
    }
}

/// 2026-09-29: Why the manifest did not load.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FamilyError {
    /// 2026-09-29: Not TOML, or not the schema.
    #[error("KERNEL_FAMILIES.toml: {0}")]
    Parse(String),
    /// 2026-09-29: A field of one family.
    #[error("family `{family}`: {detail}")]
    Field {
        /// 2026-09-29: Family id.
        family: String,
        /// 2026-09-29: What was wrong.
        detail: String,
    },
    /// 2026-09-29: An op outside the circuit vocabulary.
    #[error("family `{family}`: unknown op `{op}`")]
    UnknownOp {
        /// 2026-09-29: Family id.
        family: String,
        /// 2026-09-29: The op.
        op: String,
    },
    /// 2026-09-29: A point, evidence record or discovery rule naming a parameter the family
    /// does not declare, or a runtime one.
    #[error("family `{family}`: `{param}` is not a compile-time or policy parameter of it")]
    UnknownParam {
        /// 2026-09-29: Family id.
        family: String,
        /// 2026-09-29: The parameter.
        param: String,
    },
}

#[path = "families_compute.rs"]
mod compute_file;
#[path = "families_file.rs"]
mod file;
#[path = "families_legacy.rs"]
mod legacy_file;
#[path = "families_workspace.rs"]
mod workspace_file;

/// 2026-09-29: Parse the manifest text.
pub fn parse_families(text: &str) -> Result<Families, FamilyError> {
    file::parse(text)
}

#[cfg(test)]
#[path = "families_tests.rs"]
mod families_tests;
