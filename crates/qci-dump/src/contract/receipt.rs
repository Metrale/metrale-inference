// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: `CompatibilityReceipt` v1 (investor-mvp #46): a machine-readable pass or
//! fail for one combination, keyed by runtime commit × immutable model revision ×
//! quantization × hardware-fingerprint digest × topology, with the checks that decided it.
//!
//! [`assemble`] is the only constructor the runtime uses. It derives the verdict from the
//! checks and refuses a receipt that would claim more than its checks show: no coherence
//! check, no benchmark check, a failed check without a failure kind, or a key that cannot
//! name a real combination. Memory fit is not a check kind, so a receipt never passes on it.
//!
//! Owner: qci-dump (contract binding).
//! Invariants: `verdict` is `pass` exactly when every check passed.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::PLATFORM_VERSION;
use super::hardware::SoftwareVersion;
use super::ids::{GitCommit, Sha256Digest};
use super::rules::{self, Violation, require};

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRevision {
    /// 2026-10-11: `owner/name` on the model hub.
    pub repository: String,
    /// 2026-10-11: The full commit, never a branch or tag.
    pub revision: GitCommit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    Bf16,
    Fp16,
    Fp8,
    Nvfp4,
    Mxfp4,
    Int8,
    Int4,
    /// 2026-10-11: The runtime picks per projection; activations only.
    Adaptive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Quantization {
    pub weights: Precision,
    pub activations: Precision,
    pub kv_cache: Precision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Topology {
    pub nodes: u32,
    pub accelerators_per_node: u32,
    pub tensor_parallel: u32,
    pub pipeline_parallel: u32,
    pub expert_parallel: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptKey {
    pub runtime_commit: GitCommit,
    pub model: ModelRevision,
    pub quantization: Quantization,
    pub hardware_fingerprint: Sha256Digest,
    pub topology: Topology,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckKind {
    Coherence,
    Benchmark,
    Accuracy,
    ToolCall,
    StructuredOutput,
    Conformance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricUnit {
    Count,
    Permille,
    Milliseconds,
    Microseconds,
    /// 2026-10-11: Thousandths of a token per second.
    MilliTokensPerSecond,
    MillijoulesPerToken,
    Bytes,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metric {
    pub name: String,
    pub unit: MetricUnit,
    pub value: i64,
}

/// 2026-10-11: A file by content: where it was read and the SHA-256 of its bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRef {
    pub uri: String,
    pub sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Check {
    pub name: String,
    pub kind: CheckKind,
    pub passed: bool,
    pub parameters: BTreeMap<String, String>,
    pub artifact: ArtifactRef,
    pub metrics: Vec<Metric>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    IncoherentOutput,
    BelowThreshold,
    Crashed,
    Timeout,
    InvalidToolCall,
    InvalidStructuredOutput,
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Failure {
    pub check: String,
    pub kind: FailureKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Verdict {
    Pass,
    Fail { failures: Vec<Failure> },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompatibilityReceipt {
    pub version: u32,
    pub key: ReceiptKey,
    /// 2026-10-11: The recipe the run served (canonical-JSON digest).
    pub recipe: Sha256Digest,
    pub verdict: Verdict,
    pub checks: Vec<Check>,
    pub started_at_unix_ms: i64,
    pub finished_at_unix_ms: i64,
    pub runner: SoftwareVersion,
    /// 2026-10-11: A stand-in engine ran the checks; such a receipt never approves.
    pub synthetic: bool,
}

/// 2026-10-11: One check as the runner reports it, before it joins a receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckResult {
    pub check: Check,
    /// 2026-10-11: Required exactly when `check.passed` is false.
    pub failure: Option<FailureKind>,
}

/// 2026-10-11: Everything a receipt is made of; the clock readings are inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptInput {
    pub key: ReceiptKey,
    pub recipe: Sha256Digest,
    pub checks: Vec<CheckResult>,
    pub started_at_unix_ms: i64,
    pub finished_at_unix_ms: i64,
    pub runner: SoftwareVersion,
    pub synthetic: bool,
}

/// 2026-10-11: Why no receipt was written. Each names what is missing or contradictory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    NoCheckOfKind(CheckKind),
    DuplicateCheck(String),
    FailedWithoutKind(String),
    PassedWithFailureKind(String),
    InvalidField(String),
    TimeOrder,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoCheckOfKind(k) => write!(
                f,
                "no {k:?} check: a receipt needs coherence and benchmark evidence"
            ),
            Self::DuplicateCheck(n) => write!(f, "check {n} appears twice"),
            Self::FailedWithoutKind(n) => write!(f, "check {n} failed without a failure kind"),
            Self::PassedWithFailureKind(n) => {
                write!(f, "check {n} passed but names a failure kind")
            }
            Self::InvalidField(at) => write!(f, "invalid field {at}"),
            Self::TimeOrder => f.write_str("finished_at_unix_ms is before started_at_unix_ms"),
        }
    }
}

impl CompatibilityReceipt {
    /// 2026-10-11: The schema rules serde cannot express, including the one that ties the
    /// verdict to the checks.
    pub fn validate(&self) -> Result<(), Violation> {
        rules::version(self.version)?;
        let key = &self.key;
        require(
            rules::repository(&key.model.repository),
            "/key/model/repository",
        )?;
        require(
            key.quantization.weights != Precision::Adaptive,
            "/key/quantization/weights",
        )?;
        require(
            key.quantization.kv_cache != Precision::Adaptive,
            "/key/quantization/kv_cache",
        )?;
        let t = key.topology;
        let degrees = [
            t.nodes,
            t.accelerators_per_node,
            t.tensor_parallel,
            t.pipeline_parallel,
            t.expert_parallel,
        ];
        require(degrees.iter().all(|d| *d >= 1), "/key/topology")?;
        require(!self.checks.is_empty(), "/checks")?;
        for (i, c) in self.checks.iter().enumerate() {
            require(rules::text(&c.name), &format!("/checks/{i}/name"))?;
            let params_ok = c
                .parameters
                .iter()
                .all(|(k, v)| rules::flag_key(k) && rules::value(v));
            require(params_ok, &format!("/checks/{i}/parameters"))?;
            require(
                rules::text(&c.artifact.uri),
                &format!("/checks/{i}/artifact/uri"),
            )?;
            let metrics_ok = c.metrics.iter().all(|m| rules::text(&m.name));
            require(metrics_ok, &format!("/checks/{i}/metrics"))?;
        }
        let all_passed = self.checks.iter().all(|c| c.passed);
        match &self.verdict {
            Verdict::Pass => require(all_passed, "/verdict")?,
            Verdict::Fail { failures } => {
                require(!all_passed && !failures.is_empty(), "/verdict")?;
                let names_ok = failures.iter().all(|f| rules::text(&f.check));
                require(names_ok, "/verdict/fail/failures")?;
            }
        }
        require(
            rules::timestamp(self.started_at_unix_ms),
            "/started_at_unix_ms",
        )?;
        require(
            rules::timestamp(self.finished_at_unix_ms),
            "/finished_at_unix_ms",
        )?;
        self.runner.validate("/runner")
    }
}

/// 2026-10-11: The receipt for `input`, or the reason none can be written. Beyond the
/// schema, the emitter requires coherence and benchmark evidence, unique check names, a
/// failure kind on exactly the failed checks, and a run that ends after it starts.
pub fn assemble(input: ReceiptInput) -> Result<CompatibilityReceipt, Refusal> {
    if input.finished_at_unix_ms < input.started_at_unix_ms {
        return Err(Refusal::TimeOrder);
    }
    for kind in [CheckKind::Coherence, CheckKind::Benchmark] {
        if !input.checks.iter().any(|c| c.check.kind == kind) {
            return Err(Refusal::NoCheckOfKind(kind));
        }
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut failures = Vec::new();
    for r in &input.checks {
        let c = &r.check;
        if !seen.insert(c.name.as_str()) {
            return Err(Refusal::DuplicateCheck(c.name.clone()));
        }
        match (c.passed, r.failure) {
            (true, None) => {}
            (true, Some(_)) => return Err(Refusal::PassedWithFailureKind(c.name.clone())),
            (false, None) => return Err(Refusal::FailedWithoutKind(c.name.clone())),
            (false, Some(kind)) => failures.push(Failure {
                check: c.name.clone(),
                kind,
            }),
        }
    }
    let verdict = if failures.is_empty() {
        Verdict::Pass
    } else {
        Verdict::Fail { failures }
    };
    let receipt = CompatibilityReceipt {
        version: PLATFORM_VERSION,
        key: input.key,
        recipe: input.recipe,
        verdict,
        checks: input.checks.into_iter().map(|r| r.check).collect(),
        started_at_unix_ms: input.started_at_unix_ms,
        finished_at_unix_ms: input.finished_at_unix_ms,
        runner: input.runner,
        synthetic: input.synthetic,
    };
    receipt.validate().map_err(|v| Refusal::InvalidField(v.0))?;
    Ok(receipt)
}

#[cfg(test)]
#[path = "receipt_tests.rs"]
mod tests;
