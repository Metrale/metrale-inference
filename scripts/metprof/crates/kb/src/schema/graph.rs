// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the kernel-side kinds (DESIGN §2.2): kernels and their profiler
//! symbols, instantiations, profile runs and their samples, fusion rules,
//! circuit plans and bounds. `Option` marks a field the table marks `?` or a
//! value the source may not know; it is written as `null`, never omitted.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::port::{FileRef, Identity};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelClass {
    Compute,
    Comm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeUnit {
    TensorCore,
    CudaCore,
    Memory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Decode,
    MultiSeq,
    Verify,
    Draft,
    VerifyBatch,
    Prefill,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MappedBy {
    Plan,
    Families,
    Manual,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Kernel {
    pub module: Option<String>,
    pub func: Option<String>,
    pub family: Option<String>,
    pub group: Option<String>,
    pub class: KernelClass,
    pub compute: Option<ComputeUnit>,
    pub mma: Option<String>,
    pub rows: Option<[u64; 2]>,
    pub ops: Vec<String>,
    pub emitters: Vec<String>,
    pub file_hint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelSymbol {
    pub name: String,
    pub grid_hint: Option<String>,
    /// `None` for a name no plan, family table or person mapped.
    pub mapped_by: Option<MappedBy>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShapeEnvelope {
    pub rows: [u64; 2],
    pub isl_bucket: Option<String>,
    pub dims: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Instantiation {
    pub layer_family: String,
    pub block: Option<String>,
    pub mode: Mode,
    pub shape_envelope: ShapeEnvelope,
    pub circuit_group_idx: Option<u64>,
    pub weight_format: String,
    pub act_format: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileRun {
    pub command: String,
    /// The argv with every file argument replaced by its sha256 and basename.
    pub args: Vec<crate::port::Arg>,
    /// The metprof result map, verbatim.
    pub result: Map<String, Value>,
    /// The whole `metprof.kernel_tree/1` JSON, for `tree`.
    pub kernel_tree: Option<Value>,
    /// The tree's provenance block, when the run has a tree.
    pub provenance: Option<Map<String, Value>>,
    pub files: Vec<FileRef>,
    pub metprof_version: String,
    pub metprof_git_sha: String,
    pub families_digest: String,
    pub capture_sha256: String,
    pub identity: Identity,
    pub coverage: Option<f64>,
    pub total_time: Option<f64>,
    pub time_unit: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CircuitJoin {
    pub group: u64,
    pub rule: String,
    pub plan_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelSample {
    pub node_path: String,
    pub depth: u64,
    pub total_time: f64,
    pub self_time: f64,
    pub launches: u64,
    pub mean_us: Option<f64>,
    pub min_us: Option<f64>,
    pub max_us: Option<f64>,
    /// Per-launch p10, median and p90 over the subtree (`None` without launches).
    pub p10_us: Option<f64>,
    pub median_us: Option<f64>,
    pub p90_us: Option<f64>,
    pub union_ms: Option<f64>,
    pub fuses: Vec<String>,
    pub circuit: Option<CircuitJoin>,
    /// A leaf's launch identities (empty on an interior node).
    pub variants: Vec<Variant>,
    /// Pre-order position among the run's samples.
    pub ord: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FusionRule {
    /// The `Rule` fields of FUSIONS.toml, verbatim.
    pub rule: Map<String, Value>,
    pub inherited_from: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CircuitPlan {
    pub arch: String,
    pub mode: String,
    pub rows: u64,
    pub policy: String,
    pub recipe: String,
    pub rules_digest: String,
    pub groups_count: u64,
    pub launches: u64,
    pub edges_fused: u64,
    pub edges_materialized: u64,
    pub render_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundClass {
    Dram,
    Compute,
    Latency,
    Launch,
    InstructionFetch,
    Occupancy,
    CommExposed,
    Host,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Overlap {
    Exposed,
    Hidden,
    Partial,
    NA,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundMethod {
    #[serde(rename = "metprof floor")]
    MetprofFloor,
    Microbench,
    Manual,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Achieved {
    pub gbps: Option<f64>,
    pub tflops: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bound {
    pub bound_class: BoundClass,
    pub share_of_union: f64,
    pub floor_ratio: f64,
    pub measured_peak_ref: String,
    pub achieved: Achieved,
    pub removable_fraction: f64,
    pub critical_path_share: Option<f64>,
    pub overlap_behaviour: Overlap,
    pub method: BoundMethod,
}

/// One launch identity of a tree leaf (metprof `kernel_tree/1`): the
/// demangled name and launch configuration, their digests, and the launches'
/// times. `kernel_id` = SHA-256 of the name, a newline and the configuration
/// text; `config_sha256` = SHA-256 of that text (both computed by metprof).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Variant {
    pub demangled: String,
    pub grid: [i64; 3],
    pub block: [i64; 3],
    pub registers_per_thread: i64,
    pub static_shared_bytes: i64,
    pub dynamic_shared_bytes: i64,
    pub config_sha256: String,
    pub kernel_id: String,
    pub launches: u64,
    pub total_time: f64,
    pub p10_us: Option<f64>,
    pub median_us: Option<f64>,
    pub p90_us: Option<f64>,
}
