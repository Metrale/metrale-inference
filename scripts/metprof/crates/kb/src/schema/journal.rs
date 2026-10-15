// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the improvement-loop kinds (DESIGN §2.2, §2.5, §2.6): patterns,
//! levers (the journal v2 record, verbatim), candidates, predictions, A/B
//! records, process metrics and playbook revisions.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatternKind {
    Pattern,
    AntiPattern,
    Gate,
    TimeSink,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scope {
    pub hardware_classes: Vec<String>,
    pub models: Vec<String>,
    pub regimes: Vec<String>,
    pub modes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pattern {
    pub kind: PatternKind,
    pub title: String,
    pub trigger_symptom: String,
    pub first_method: String,
    pub lever_classes: Vec<String>,
    pub scope: Scope,
    pub evidence_counts: Map<String, Value>,
    pub confidence: f64,
    pub qualifications: Vec<String>,
    pub retired: bool,
    pub origin_doc: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Revision {
    pub source_ref: String,
    pub content_sha256: String,
    pub verdict: String,
    pub date: String,
}

/// A lever attempt: the journal v2 record verbatim (its vocabulary is checked
/// by the journal lint, not here), plus the union's other versions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lever {
    pub journal: Map<String, Value>,
    pub revisions: Vec<Revision>,
    pub conflict: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    Kb,
    Research,
    FirstPrinciples,
    Recombination,
    Novel,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Score {
    pub target_ms: f64,
    pub removable_fraction: f64,
    pub k: f64,
    pub p_engage: f64,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub origin: Origin,
    pub hypothesis: String,
    pub mechanism: String,
    pub sources: Vec<String>,
    pub kb_precedents: Vec<String>,
    pub score: Score,
    pub mde_ms: f64,
    pub chosen: bool,
    pub rejected_why: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Delta {
    pub value: f64,
    pub unit: String,
    pub direction: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Agent {
    pub model: String,
    pub role: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prediction {
    pub projected_delta: Delta,
    pub estimate_kind: String,
    pub basis: String,
    pub by_agent: Agent,
    pub measured_delta: Option<Delta>,
    pub ratio: Option<f64>,
    pub scored_at: Option<String>,
}

/// A metprof `verdict` record verbatim, plus the derived verdict fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ABRecord {
    pub record: Map<String, Value>,
    pub verdict: String,
    pub delta: f64,
    pub ratio: f64,
    pub improvement: f64,
    pub control_spread: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Window {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessMetric {
    pub kind: String,
    pub scope: Map<String, Value>,
    pub value: f64,
    pub n: u64,
    pub window: Window,
    pub method_version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlaybookRevision {
    pub date: String,
    pub files_changed: Vec<String>,
    pub diff_sha256: String,
    pub rationale: String,
    pub metrics_cited: Vec<String>,
    pub proposed_by: Agent,
    pub approved_by: String,
    pub pr_url: Option<String>,
}
