// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The serve settings a gate record discloses about the server it
//! measured (`GateRecord::serve_resolved`).
//!
//! `served_by` names the in-tree recipe (`gate::recipe_closure`), and
//! `serve_overrides` only the keys changed for the run; neither states what
//! the server resolved those settings to (2026-10-02). The keys are defined here, in the crate that owns the
//! record, and the server CLI fills them from its rendered serve flags.
//!
//! Owner: bench gate (records).
//! Invariants:
//! - A [`disclosure`] always carries [`SPECULATIVE`], [`WEIGHT_QUANTIZATION`] and
//!   [`ACTIVATION_QUANTIZATION`]; every other key is present only when resolved or on.

use std::collections::BTreeMap;

use super::record::GateRecord;

/// 2026-09-26: Key for the `--mtp-gate` regime in force: `force` or `auto`.
pub const MTP_GATE: &str = "mtp_gate";
/// 2026-09-26: Key for whether `--speculative` was on: `true` or `false`.
pub const SPECULATIVE: &str = "speculative";
/// 2026-09-26: Key for `--prefill-codispatch`, present (`true`) only when the
/// rendered serve gave the flag.
pub const PREFILL_CODISPATCH: &str = "prefill_codispatch";
/// 2026-09-26: Key for `--w4a4-downcast`, present (`true`) only when on. The
/// flag defaults to false and has no environment fallback, so an absent key
/// means off.
pub const W4A4_DOWNCAST: &str = "w4a4_downcast";
/// 2026-09-28: Key for `--weight-quantization`, always present on a record written since the
/// flag exists, with the tier's name (`declared`, `nvfp4`). A record without it predates the
/// flag, and its server ran what `nvfp4` names.
pub const WEIGHT_QUANTIZATION: &str = "weight_quantization";
/// 2026-09-27: Key for `--expert-quantization`, present only for a tier other than the default
/// `fp8`, with the tier's name as the value (`nvfp4-gate-up`, `nvfp4`). The flag has no
/// environment fallback, so an absent key means `fp8`.
pub const EXPERT_QUANTIZATION: &str = "expert_quantization";
/// 2026-09-30: Key for `--activation-quantization`, always present on a record written since the
/// flag exists, with the value's canonical form (`adaptive`, `declared`, a ladder). A record
/// without it predates the flag, and its server ran what `adaptive` names.
pub const ACTIVATION_QUANTIZATION: &str = "activation_quantization";

/// 2026-09-26: The disclosure for a server whose rendered flags resolved to
/// these values.
///
/// `mtp_gate_force` is the server's resolution of `--mtp-gate` and
/// `--hermetic`: `Some(true)` is written `force`, `Some(false)` `auto`.
/// `None` (no flag, so the server's `METRALE_MTP_GATE_FORCE` decides) writes
/// no key rather than a guessed default.
///
/// `prefill_codispatch` writes `true` when the flag was given and no key
/// otherwise; the server's `METRALE_PREFILL_CODISPATCH` then decides, and
/// `serve_env` discloses it when the recipe declares it.
///
/// `expert_quantization` is the tier's name when it is not the default `fp8`, else `None`.
/// `weight_quantization` is the `--weight-quantization` tier's name, always written, and so is
/// `activation_quantization`, the `--activation-quantization` value's canonical form.
pub fn disclosure(
    mtp_gate_force: Option<bool>,
    speculative: bool,
    prefill_codispatch: bool,
    w4a4_downcast: bool,
    expert_quantization: Option<&str>,
    weight_quantization: &str,
    activation_quantization: &str,
) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    m.insert(
        ACTIVATION_QUANTIZATION.to_string(),
        activation_quantization.to_string(),
    );
    m.insert(SPECULATIVE.to_string(), speculative.to_string());
    m.insert(
        WEIGHT_QUANTIZATION.to_string(),
        weight_quantization.to_string(),
    );
    if let Some(force) = mtp_gate_force {
        m.insert(
            MTP_GATE.to_string(),
            if force { "force" } else { "auto" }.to_string(),
        );
    }
    if prefill_codispatch {
        m.insert(PREFILL_CODISPATCH.to_string(), "true".to_string());
    }
    if w4a4_downcast {
        m.insert(W4A4_DOWNCAST.to_string(), "true".to_string());
    }
    if let Some(tier) = expert_quantization {
        m.insert(EXPERT_QUANTIZATION.to_string(), tier.to_string());
    }
    m
}

/// 2026-09-28: Key for `--forward`, present only for a forward other than `legacy`.
pub const FORWARD: &str = "forward";
/// 2026-09-28: Key for the live decode plan's digest (`metrale_circuit::digest::plan_digest`),
/// present only when the server runs a circuit forward.
pub const PLAN_DIGEST: &str = "plan_digest";

/// 2026-09-28: What a server reports about its forward (`GET /forward`): the server fills it
/// from its model, the harness reads it into the record.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LiveForward {
    /// 2026-09-28: `legacy`, `circuit` or `circuit-reference`.
    pub forward: String,
    /// 2026-09-28: The decode plan's digest; `None` under `legacy`.
    #[serde(default)]
    pub plan_digest: Option<String>,
    /// 2026-10-01: The slot count `--max-batch-size auto` resolved to; `None` for an explicit
    /// count (which the rendered serve already states).
    #[serde(default)]
    pub auto_max_batch_size: Option<usize>,
    /// 2026-10-02: The MoE expert-table decision the serve's memory plan made before load
    /// (`build` or `skip`); `None` when the loader reads none.
    #[serde(default)]
    pub moe_expert_tables: Option<String>,
    /// 2026-10-03: SHA-256 of the kernel tree the serve's memory plan was computed from
    /// (`metrale_kernel_tree::SHA256`); `None` when the serve made no plan.
    #[serde(default)]
    pub kernel_tree: Option<String>,
    /// 2026-10-03: The digest of the mock (rehearsal) checkpoint the server runs (`met serve
    /// --mock`, or a mock checkpoint's directory); `None` for a real checkpoint.
    #[serde(default)]
    pub mock: Option<String>,
}

/// 2026-10-03: Key for the digest of the mock (rehearsal) checkpoint the server runs; present only
/// for a mock. A record carrying it fails every gate ([`mock_problem`]).
pub const MOCK: &str = "mock";

/// 2026-10-03: Why a record of a mock server cannot pass a gate; `None` for a real checkpoint.
pub fn mock_problem(record: &GateRecord) -> Option<String> {
    record.mock.as_ref().map(|d| {
        format!(
            "measured on a mock (rehearsal) checkpoint {d}: synthetic weights certify no number"
        )
    })
}

/// 2026-10-03: Whether a run may measure the server `live` describes (`None`: not a Metrale
/// endpoint, or one too old to say): a gate run never measures a mock, and an accuracy
/// (`Correctness`) benchmark on synthetic weights measures nothing. Speed runs on a mock are the
/// point of a mock and are allowed; their records are not gate records.
pub fn mock_run_allowed(
    live: Option<&LiveForward>,
    correctness: bool,
    gate: bool,
) -> Result<(), String> {
    match live.and_then(|l| l.mock.as_deref()) {
        Some(d) if gate => Err(format!(
            "the server runs mock (rehearsal) checkpoint {d}: a gate run never measures a mock"
        )),
        Some(d) if correctness => Err(format!(
            "the server runs mock (rehearsal) checkpoint {d}: an accuracy benchmark on \
             synthetic weights measures nothing (measure accuracy on the full model)"
        )),
        _ => Ok(()),
    }
}

/// 2026-10-01: Key for the slot count `--max-batch-size auto` resolved to, written `auto:<n>`;
/// present only for `auto`.
pub const MAX_BATCH_SIZE: &str = "max_batch_size";
/// 2026-10-02: Key for the MoE expert-table decision, present (`skip`) only when the serve's
/// memory plan dropped the transposed MoE prefill tables. Absent means built, as every serve
/// before the decision existed did.
pub const MOE_EXPERT_TABLES: &str = "moe_expert_tables";
/// 2026-10-03: Key for the SHA-256 of the kernel tree the serve's memory plan read, present
/// whenever the serve planned (every serve whose loader reads the MoE expert-table decision).
pub const KERNEL_TREE: &str = "kernel_tree";

/// 2026-09-28: Add the live forward to `resolved`: [`FORWARD`] when it is not `legacy`, and
/// [`PLAN_DIGEST`]. `requested` is the forward the rendered serve asked for; a server running
/// another one, or a circuit without a digest, or legacy with one, is refused: the record would
/// state a configuration the measurement did not run.
pub fn merge_live_forward(
    resolved: &mut BTreeMap<String, String>,
    requested: &str,
    live: &LiveForward,
) -> Result<(), String> {
    let skipped = match live.moe_expert_tables.as_deref() {
        None | Some("build") => false,
        Some("skip") => true,
        Some(other) => return Err(format!("the server reports MoE expert tables `{other}`")),
    };
    if let Some(d) = &live.kernel_tree
        && !(d.len() == 64 && d.bytes().all(|b| b.is_ascii_hexdigit()))
    {
        return Err(format!(
            "the server reports kernel tree `{d}`, not a SHA-256"
        ));
    }
    if live.forward != requested {
        return Err(format!(
            "the server runs forward `{}`, the rendered serve asked for `{requested}`",
            live.forward
        ));
    }
    match (live.forward.as_str(), &live.plan_digest) {
        ("legacy", None) => {}
        ("legacy", Some(d)) => return Err(format!("a legacy forward reports plan digest {d}")),
        (other, Some(d)) => {
            resolved.insert(FORWARD.to_string(), other.to_string());
            resolved.insert(PLAN_DIGEST.to_string(), d.clone());
        }
        (other, None) => return Err(format!("forward `{other}` reports no plan digest")),
    }
    if let Some(n) = live.auto_max_batch_size {
        resolved.insert(MAX_BATCH_SIZE.to_string(), format!("auto:{n}"));
    }
    if skipped {
        resolved.insert(MOE_EXPERT_TABLES.to_string(), "skip".to_string());
    }
    if let Some(d) = &live.kernel_tree {
        resolved.insert(KERNEL_TREE.to_string(), d.clone());
    }
    if let Some(d) = &live.mock {
        resolved.insert(MOCK.to_string(), d.clone());
    }
    Ok(())
}

impl GateRecord {
    /// 2026-09-26: Attach what the gate's serve resolved; see [`disclosure`].
    #[must_use]
    pub fn with_serve_resolved(mut self, resolved: BTreeMap<String, String>) -> Self {
        // 2026-10-03: The mock digest is also a field of its own: `check_record` never reads
        // `serve_resolved`, and the checks that refuse a mock read the field.
        self.mock = resolved.get(MOCK).cloned();
        self.serve_resolved = resolved;
        self
    }
}

#[cfg(test)]
#[path = "record_serve_tests.rs"]
mod record_serve_tests;
