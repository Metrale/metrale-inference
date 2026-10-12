// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The run spec a qualification run hands to `qci-dump --receipt`, and the
//! receipt built from it. The model half of the key comes from the campaign recipe's pins
//! (the same pins the text dump prints), never from the spec. File paths in the spec are
//! read by the binary; this module only sees their bytes' digests.
//!
//! Owner: qci-dump.
//! Invariants: [`build`] reads no file and no clock.

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer};

use crate::contract::canonical;
use crate::contract::hardware::{HardwareFingerprint, SoftwareVersion};
use crate::contract::ids::{GitCommit, Sha256Digest};
use crate::contract::receipt::{
    ArtifactRef, Check, CheckKind, CheckResult, CompatibilityReceipt, FailureKind, Metric,
    ModelRevision, Quantization, ReceiptInput, ReceiptKey, Topology, assemble,
};
use crate::recipe;

/// 2026-10-11: A field that must be written, as a value or `null`.
fn present<'de, D: Deserializer<'de>, T: Deserialize<'de>>(d: D) -> Result<Option<T>, D::Error> {
    Option::deserialize(d)
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckSpec {
    pub name: String,
    pub kind: CheckKind,
    pub passed: bool,
    #[serde(deserialize_with = "present")]
    pub failure: Option<FailureKind>,
    pub parameters: BTreeMap<String, String>,
    /// 2026-10-11: Path of the check's raw output; the binary digests its bytes.
    pub artifact: String,
    pub metrics: Vec<Metric>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunSpec {
    pub runtime_commit: GitCommit,
    pub recipe_digest: Sha256Digest,
    pub quantization: Quantization,
    pub topology: Topology,
    /// 2026-10-11: Path of a `HardwareFingerprint` v1 document.
    pub hardware_fingerprint: String,
    pub started_at_unix_ms: i64,
    pub finished_at_unix_ms: i64,
    pub synthetic: bool,
    pub checks: Vec<CheckSpec>,
}

/// 2026-10-11: Parse a run spec.
pub fn parse(bytes: &[u8]) -> Result<RunSpec, String> {
    serde_json::from_slice(bytes).map_err(|e| format!("run spec: {e}"))
}

/// 2026-10-11: The receipt of `spec` for campaign `case`. `fingerprint` is the bytes of
/// the fingerprint file; `artifacts` maps each check's artifact path to its digest.
pub fn build(
    case: &str,
    spec: RunSpec,
    fingerprint: &[u8],
    artifacts: &BTreeMap<String, Sha256Digest>,
    runner: SoftwareVersion,
) -> Result<CompatibilityReceipt, String> {
    let pins = recipe::for_case(case).ok_or_else(|| format!("unknown case {case}"))?;
    let revision = GitCommit::from_hex(&pins.revision)
        .map_err(|e| format!("recipe {} revision is not a full commit: {e}", pins.id))?;
    let fp: HardwareFingerprint =
        serde_json::from_slice(fingerprint).map_err(|e| format!("hardware fingerprint: {e}"))?;
    fp.validate()
        .map_err(|v| format!("hardware fingerprint: {v}"))?;
    let fp_digest =
        canonical::digest(&fp).map_err(|e| format!("hardware fingerprint digest: {e:?}"))?;
    let mut checks = Vec::with_capacity(spec.checks.len());
    for c in spec.checks {
        let sha256 = *artifacts
            .get(&c.artifact)
            .ok_or_else(|| format!("check {}: artifact {} was not read", c.name, c.artifact))?;
        checks.push(CheckResult {
            failure: c.failure,
            check: Check {
                name: c.name,
                kind: c.kind,
                passed: c.passed,
                parameters: c.parameters,
                artifact: ArtifactRef {
                    uri: c.artifact,
                    sha256,
                },
                metrics: c.metrics,
            },
        });
    }
    assemble(ReceiptInput {
        key: ReceiptKey {
            runtime_commit: spec.runtime_commit,
            model: ModelRevision {
                repository: pins.repository,
                revision,
            },
            quantization: spec.quantization,
            hardware_fingerprint: fp_digest,
            topology: spec.topology,
        },
        recipe: spec.recipe_digest,
        checks,
        started_at_unix_ms: spec.started_at_unix_ms,
        finished_at_unix_ms: spec.finished_at_unix_ms,
        runner,
        synthetic: spec.synthetic,
    })
    .map_err(|r| format!("receipt refused: {r}"))
}

#[cfg(test)]
#[path = "receipt_spec_tests.rs"]
mod tests;
