// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the context kinds (DESIGN §2.2): campaigns, hardware classes,
//! models, builds, measurements, gates, boxes, evidence and documents.
//! Nested blocks the table leaves open (`{…}`) are JSON objects kept verbatim.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Competitor {
    pub engine: String,
    pub version: String,
    pub image_digest: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreRow {
    pub cell: String,
    pub ours: Option<f64>,
    pub theirs: Option<f64>,
    pub unit: String,
    pub direction: String,
    pub measured_at: String,
    pub n: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Campaign {
    pub title: String,
    pub goal_metric: String,
    pub target_cells: Vec<String>,
    pub started: String,
    pub ended: Option<String>,
    pub boxes: Vec<String>,
    pub status: String,
    pub competitor: Option<Competitor>,
    pub scoreboard_rows: Vec<ScoreRow>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareClass {
    pub arch: String,
    pub compute_capability: String,
    pub sm_count: u64,
    pub memory_bandwidth_gbps: f64,
    pub memory_type: String,
    pub memory_gb: f64,
    pub inherits: Option<String>,
    pub measured_peaks: Map<String, Value>,
    pub devices: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub arch: String,
    pub family: String,
    pub moe: bool,
    pub params: String,
    pub precision_tier: String,
    pub layer_kinds: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Build {
    pub repo: String,
    pub branch: Option<String>,
    pub git_dirty: bool,
    pub binary_sha256: String,
    pub env_sha256: String,
    pub recipe_sha256: String,
    pub built_at: Option<String>,
    pub perf_env: Map<String, Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpreadKind {
    Halfrange,
    Sd,
    Iqr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spread {
    pub kind: SpreadKind,
    pub value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Instrument {
    MetBench,
    Metprof,
    Nvml,
    Ladder,
    MlperfEdge,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Engine {
    pub kind: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Measurement {
    pub metric: String,
    pub cell: Map<String, Value>,
    pub value: f64,
    pub unit: String,
    pub direction: String,
    pub n: u64,
    pub spread: Option<Spread>,
    pub boots: Option<u64>,
    pub cool_wait_s: Option<f64>,
    pub thermal: Option<Map<String, Value>>,
    pub clocks: Option<Map<String, Value>>,
    pub instrument: Instrument,
    pub instrument_sha256: String,
    pub draw_sha256: Option<String>,
    pub competitor: bool,
    pub engine: Engine,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gate {
    pub stage: u8,
    pub cost_minutes: f64,
    pub automated: bool,
    pub tool: String,
    pub description: String,
}

/// A machine, by alias only: never a hostname, an address or a machine id in
/// clear (`machine_id_sha256` is the digest).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Box {
    pub alias: String,
    pub hardware_class: String,
    pub device_id: String,
    pub role: String,
    pub machine_id_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    NsysCapture,
    ServeLog,
    BenchJson,
    StatusSection,
    PrComment,
    Screenshot,
    ReleaseAsset,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Evidence {
    pub kind: EvidenceKind,
    pub sha256: String,
    pub size: u64,
    pub private_locator_ref: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocSource {
    StatusMd,
    Playbook,
    MemoryNote,
    Mining,
    Paper,
    VendorDoc,
    UpstreamKernel,
    Bible,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Doc {
    pub title: String,
    pub text: String,
    pub source: DocSource,
    pub url_or_ref: String,
    pub cited_kernels: Vec<String>,
}
