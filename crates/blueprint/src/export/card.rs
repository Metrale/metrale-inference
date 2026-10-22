// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: `kbp-card/v1`: the store's index view of one verified blueprint, no kernel
//! payloads (DESIGN.md §10).
//!
//! Owner: blueprint.
//! Invariants:
//! - A pure function of a verified, opened file ([`crate::frame::Opened`]); the exporter never
//!   sees an unverified body, so `signature.status` is `verified` whenever a card exists.
//! - `performance` is filled only from gate records the blueprint's `evidence` names, each with
//!   its fingerprint; Phase 1 resolves no records, so it is empty.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use super::schema::{json_enum, json_struct};
use super::{ExportContext, linear_formats};
use crate::frame::{FLAG_FULL, Opened};
use crate::types::FlagValue;

/// 2026-10-22: The card schema id.
pub const CARD_SCHEMA: &str = "kbp-card/v1";

json_enum! {
    /// 2026-10-22: What the exporter found when it verified the file.
    pub enum SignatureStatus {
        /// 2026-10-22: Verified against an anchor.
        Verified => "verified",
        /// 2026-10-22: Not verified.
        Unverified => "unverified",
        /// 2026-10-22: Signed by a revoked key.
        Revoked => "revoked",
        /// 2026-10-22: Signed under an expired certificate.
        Expired => "expired",
    }
}

json_struct! {
    /// 2026-10-22: The checkpoint and shape.
    pub struct CardModel {
        /// 2026-10-22: HF repo.
        pub repo: String,
        /// 2026-10-22: Git revision.
        pub revision: String,
        /// 2026-10-22: Circuit arch.
        pub arch: String,
        /// 2026-10-22: Layer count per kind.
        pub layer_kinds: BTreeMap<String, u64>,
        /// 2026-10-22: `max_model_len`, when the policy states it as an integer.
        pub max_seq_len: Option<u64>,
    }
}

json_struct! {
    /// 2026-10-22: The hardware pin.
    pub struct CardHardware {
        /// 2026-10-22: Class.
        pub class: String,
        /// 2026-10-22: PTX arch.
        pub ptx_arch: String,
        /// 2026-10-22: Ranks.
        pub world_size: u32,
        /// 2026-10-22: Least visible device memory, MiB.
        pub min_visible_mib: u64,
        /// 2026-10-22: Allowed devices.
        pub devices: Vec<String>,
    }
}

json_struct! {
    /// 2026-10-22: Precision summary.
    pub struct CardPrecision {
        /// 2026-10-22: The precision table name.
        pub tier: String,
        /// 2026-10-22: Its text's SHA-256 (hex).
        pub table_sha256: String,
        /// 2026-10-22: Weight-reading nodes by `weight/activation` format.
        pub linear_by_format: BTreeMap<String, u64>,
        /// 2026-10-22: `kv_cache_dtype`, when stated.
        pub kv_cache: Option<String>,
    }
}

json_struct! {
    /// 2026-10-22: Kernel set summary.
    pub struct CardKernels {
        /// 2026-10-22: Modules.
        pub count: u64,
        /// 2026-10-22: Entry points across the modules.
        pub entries: u64,
        /// 2026-10-22: `module::func` entries the plans name.
        pub planned: u64,
        /// 2026-10-22: Distinct families among the planned entries.
        pub families: u64,
        /// 2026-10-22: Planned entries no family declares.
        pub residual: u64,
        /// 2026-10-22: Modules by artifact kind.
        pub kinds: BTreeMap<String, u64>,
        /// 2026-10-22: Raw module bytes.
        pub bytes_raw: u64,
        /// 2026-10-22: Inline payload bytes, for a full file.
        pub bytes_deflated: Option<u64>,
        /// 2026-10-22: Modules by source.
        pub sources: BTreeMap<String, u64>,
    }
}

json_struct! {
    /// 2026-10-22: One plan arm.
    pub struct CardPlan {
        /// 2026-10-22: Mode.
        pub mode: String,
        /// 2026-10-22: Rows.
        pub rows: u64,
        /// 2026-10-22: Plan digest (hex).
        pub digest: String,
        /// 2026-10-22: Groups.
        pub groups: u64,
    }
}

json_struct! {
    /// 2026-10-22: A listed arm the rules do not cover.
    pub struct CardPlanGap {
        /// 2026-10-22: Mode.
        pub mode: String,
        /// 2026-10-22: Rows.
        pub rows: u64,
        /// 2026-10-22: Why.
        pub reason: String,
    }
}

json_struct! {
    /// 2026-10-22: Sizes in bytes; a file reports its own form, the other is null.
    pub struct CardSizes {
        /// 2026-10-22: The body section.
        pub body: u64,
        /// 2026-10-22: The lite file.
        pub lite_file: Option<u64>,
        /// 2026-10-22: The full file.
        pub full_file: Option<u64>,
    }
}

json_struct! {
    /// 2026-10-22: The verification the exporter ran.
    pub struct CardSignature {
        /// 2026-10-22: Status.
        pub status: SignatureStatus,
        /// 2026-10-22: Suite of the signing key.
        pub suite: String,
        /// 2026-10-22: Signing kid.
        pub kid: String,
        /// 2026-10-22: Anchor kid.
        pub anchor: String,
        /// 2026-10-22: Unix ms of the verification.
        pub verified_at: i64,
    }
}

json_struct! {
    /// 2026-10-22: A benchmark number's fingerprint.
    pub struct CardFingerprint {
        /// 2026-10-22: The gate record's SHA-256.
        pub record_sha256: String,
        /// 2026-10-22: Closure digest.
        pub closure: String,
        /// 2026-10-22: Served argv SHA-256.
        pub argv_sha256: String,
        /// 2026-10-22: Serve env SHA-256.
        pub env_sha256: String,
        /// 2026-10-22: Sample draw SHA-256, for drawn benchmarks.
        pub draw_sha256: Option<String>,
        /// 2026-10-22: Engine git sha.
        pub git_sha: String,
    }
}

json_struct! {
    /// 2026-10-22: One measured number with its fingerprint.
    pub struct CardPerformance {
        /// 2026-10-22: Benchmark id.
        pub benchmark: String,
        /// 2026-10-22: Metric.
        pub metric: String,
        /// 2026-10-22: Value (presentation only; JSON is never signed).
        pub value: f64,
        /// 2026-10-22: Unit.
        pub unit: String,
        /// 2026-10-22: Box class.
        pub box_class: String,
        /// 2026-10-22: Fingerprint.
        pub fingerprint: CardFingerprint,
    }
}

json_struct! {
    /// 2026-10-22: An evidence reference.
    pub struct CardEvidence {
        /// 2026-10-22: Kind.
        pub kind: String,
        /// 2026-10-22: Id.
        pub id: String,
        /// 2026-10-22: Closure (hex), if any.
        pub closure: Option<String>,
    }
}

json_struct! {
    /// 2026-10-22: `kbp-card/v1`.
    pub struct Card {
        /// 2026-10-22: `kbp-card/v1`.
        pub schema: String,
        /// 2026-10-22: `sha256:<hex>`.
        pub blueprint_id: String,
        /// 2026-10-22: Lineage (hex).
        pub lineage: String,
        /// 2026-10-22: Revision.
        pub revision: u64,
        /// 2026-10-22: Name.
        pub name: String,
        /// 2026-10-22: Exporting engine version.
        pub exported_by: String,
        /// 2026-10-22: Model.
        pub model: CardModel,
        /// 2026-10-22: Hardware.
        pub hardware: CardHardware,
        /// 2026-10-22: Precision.
        pub precision: CardPrecision,
        /// 2026-10-22: `executor` or `legacy:<family>`.
        pub execution: String,
        /// 2026-10-22: Kernels.
        pub kernels: CardKernels,
        /// 2026-10-22: Plans.
        pub plans: Vec<CardPlan>,
        /// 2026-10-22: Listed arms the rules do not cover (legacy engines only).
        pub plan_gaps: Vec<CardPlanGap>,
        /// 2026-10-22: Sizes.
        pub sizes: CardSizes,
        /// 2026-10-22: Signature.
        pub signature: CardSignature,
        /// 2026-10-22: Performance.
        pub performance: Vec<CardPerformance>,
        /// 2026-10-22: Evidence.
        pub evidence: Vec<CardEvidence>,
    }
}

fn bump(m: &mut BTreeMap<String, u64>, k: &str) {
    *m.entry(String::from(k)).or_default() += 1;
}

/// 2026-10-22: The card of a verified file.
pub fn card(o: &Opened, ctx: &ExportContext) -> Card {
    let bp = &o.blueprint;
    let mut layer_kinds = BTreeMap::new();
    for k in &bp.circuit.layer_kinds {
        bump(&mut layer_kinds, k.name());
    }
    let mut linear_by_format = BTreeMap::new();
    for (w, a) in linear_formats(bp) {
        bump(&mut linear_by_format, &alloc::format!("{w}/{a}"));
    }
    let k = &bp.kernels;
    let mut kinds = BTreeMap::new();
    for m in &k.modules {
        bump(&mut kinds, m.kind.name());
    }
    let mut sources = BTreeMap::new();
    for s in &o.unsigned.sources {
        bump(&mut sources, s.name());
    }
    let families: alloc::collections::BTreeSet<&str> =
        k.specs.iter().filter_map(|s| s.family.as_deref()).collect();
    let full = o.header.flags & FLAG_FULL != 0;
    let n = |x: usize| u64::try_from(x).unwrap_or(u64::MAX);
    Card {
        schema: String::from(CARD_SCHEMA),
        blueprint_id: alloc::format!("{}", o.id),
        lineage: bp.lineage.0.hex(),
        revision: bp.revision,
        name: bp.name.clone(),
        exported_by: ctx.exported_by.clone(),
        model: CardModel {
            repo: bp.checkpoint.repo.clone(),
            revision: bp.checkpoint.revision.clone(),
            arch: bp.circuit.arch.clone(),
            layer_kinds,
            max_seq_len: match bp.policy.flags.get("max_model_len") {
                Some(FlagValue::Int(v)) => u64::try_from(*v).ok(),
                _ => None,
            },
        },
        hardware: CardHardware {
            class: bp.hardware.class.clone(),
            ptx_arch: bp.hardware.ptx_arch.clone(),
            world_size: bp.hardware.world_size,
            min_visible_mib: bp.hardware.min_visible_mib,
            devices: bp.hardware.devices.clone(),
        },
        precision: CardPrecision {
            tier: bp.precision.tier.clone(),
            table_sha256: bp.precision.table_sha256.hex(),
            linear_by_format,
            kv_cache: bp.policy.flags.get("kv_cache_dtype").map(FlagValue::render),
        },
        execution: bp.execution.label(),
        kernels: CardKernels {
            count: n(k.modules.len()),
            entries: n(k.modules.iter().map(|m| m.entries.len()).sum()),
            planned: n(k.specs.len()),
            families: n(families.len()),
            residual: n(k.specs.iter().filter(|s| s.family.is_none()).count()),
            kinds,
            bytes_raw: k.modules.iter().map(|m| m.len).sum(),
            bytes_deflated: full.then(|| o.payload.iter().map(|p| n(p.deflated.len())).sum()),
            sources,
        },
        plans: bp
            .plans
            .iter()
            .map(|p| CardPlan {
                mode: String::from(p.mode.name()),
                rows: p.rows,
                digest: p.digest.hex(),
                groups: n(p.groups.len()),
            })
            .collect(),
        plan_gaps: bp
            .plan_gaps
            .iter()
            .map(|g| CardPlanGap {
                mode: String::from(g.mode.name()),
                rows: g.rows,
                reason: g.reason.clone(),
            })
            .collect(),
        sizes: CardSizes {
            body: o.header.body_len,
            lite_file: (!full).then_some(ctx.file_len),
            full_file: full.then_some(ctx.file_len),
        },
        signature: CardSignature {
            status: SignatureStatus::Verified,
            suite: String::from(o.signer.cert.keys.suite.name()),
            kid: String::from(o.signer.kid.as_str()),
            anchor: String::from(o.signer.anchor.as_str()),
            verified_at: ctx.verified_at_ms,
        },
        performance: Vec::new(),
        evidence: bp
            .evidence
            .iter()
            .map(|e| CardEvidence {
                kind: e.kind.clone(),
                id: e.id.clone(),
                closure: e.closure.map(|c| c.hex()),
            })
            .collect(),
    }
}
