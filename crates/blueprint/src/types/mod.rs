// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The types of a blueprint body: one serve, fully stated.
//!
//! Owner: blueprint.
//! Invariants (the `wire-types` gate, `tests/wire_gate.rs`):
//! - The circuit instance, its plans and their kernel ids are `metrale_circuit`'s own types
//!   (SSOT): this crate defines no type that duplicates one of them, and no conversion.
//! - The types defined here hold only `BTreeMap`, `Vec`, `Option`, integers of fixed width,
//!   `bool`, `String` and each other: no `HashMap`, `usize`, `f32` or `f64`. The circuit's
//!   types are ordered (`BTreeMap`, `Vec`) and float-free, and postcard writes their `usize`
//!   indices as the same varint as a `u64`, so a value re-encodes to the same bytes on every
//!   host (the body is re-encoded and compared on every open).

mod checkpoint;
mod kernel;
mod plan;
mod settings;

use alloc::string::String;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

pub use checkpoint::{CheckpointPin, HardwareSpec, ShardDigest, Vendor, WeightsDigest};
pub use kernel::{ArtifactKind, KernelSet, KernelSpec, KernelTarget, ModuleSpec};
pub use metrale_circuit::{Circuit, FusionPlan, KernelId, Mode};
pub use plan::PlanGap;
pub use settings::{
    Fixed, FlagValue, LeverValue, PrecisionPin, Resources, ServePolicy, ServeSettings, TemplatePin,
};

use crate::ids::{LineageId, Sha256};

/// 2026-10-22: The body schema. Bumped with every change of the body's shape; an additive
/// field is `serde(default)`. 2: the circuit and plans became `metrale_circuit`'s own types
/// (was: mirrored wire records), a change of shape, so a schema-1 body is refused rather than
/// misread ([`OLDEST_BODY_SCHEMA`]); none was ever published.
pub const BODY_SCHEMA: u32 = 2;

/// 2026-10-22: The oldest body schema this reader decodes.
pub const OLDEST_BODY_SCHEMA: u32 = 2;

/// 2026-10-22: One serve, fully stated and signed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Blueprint {
    /// 2026-10-22: [`BODY_SCHEMA`] when written; the first field, so a reader can refuse a
    /// newer body before decoding the rest.
    pub schema: u32,
    /// 2026-10-22: The line this blueprint belongs to.
    pub lineage: LineageId,
    /// 2026-10-22: Monotonic within the lineage.
    pub revision: u64,
    /// 2026-10-22: The recipe stem, e.g. `glm-5.3-flash-nvfp4-tp3-512k`.
    pub name: String,
    /// 2026-10-22: The engine it was built against.
    pub engine: EnginePin,
    /// 2026-10-22: The checkpoint.
    pub checkpoint: CheckpointPin,
    /// 2026-10-22: The hardware.
    pub hardware: HardwareSpec,
    /// 2026-10-22: The circuit instance.
    pub circuit: Circuit,
    /// 2026-10-22: Where the per-node formats came from.
    pub precision: PrecisionPin,
    /// 2026-10-22: Every `(mode, rows)` arm the instance lists that the rules fuse.
    pub plans: Vec<FusionPlan>,
    /// 2026-10-22: The listed arms the rules do not cover. Empty for `Execution::Executor`
    /// (the builder refuses a gap there); a legacy engine runs those arms outside the plan.
    pub plan_gaps: Vec<PlanGap>,
    /// 2026-10-22: Every module the serve may load, by hash.
    pub kernels: KernelSet,
    /// 2026-10-22: The `METRALE_*` levers.
    pub settings: ServeSettings,
    /// 2026-10-22: The serve flags.
    pub policy: ServePolicy,
    /// 2026-10-22: References to evidence (gate records); never numbers.
    pub evidence: Vec<EvidenceRef>,
    /// 2026-10-22: Who executes the plan.
    pub execution: Execution,
}

/// 2026-10-22: The engine a blueprint was built against.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnginePin {
    /// 2026-10-22: The engine version that built it (`met --version`).
    pub version: String,
    /// 2026-10-22: The plan digest schema (`metrale_circuit::digest::DIGEST_SCHEMA`).
    pub digest_schema: u32,
    /// 2026-10-22: SHA-256 of the FUSIONS.toml the plans were fused from.
    pub rules_sha256: Sha256,
}

/// 2026-10-22: How the engine runs the plan.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Execution {
    /// 2026-10-22: The circuit executor builds from the plans (golden instances).
    Executor,
    /// 2026-10-22: A model-family engine, held to the kernel set; `family` is the circuit arch.
    Legacy {
        /// 2026-10-22: The arch, e.g. `glm5_next`.
        family: String,
    },
}

impl Execution {
    /// 2026-10-22: `executor` or `legacy:<family>`.
    pub fn label(&self) -> String {
        match self {
            Execution::Executor => String::from("executor"),
            Execution::Legacy { family } => alloc::format!("legacy:{family}"),
        }
    }
}

/// 2026-10-22: A reference to evidence: a gate record id and its closure digest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRef {
    /// 2026-10-22: `gate_record`, `golden_plan`.
    pub kind: String,
    /// 2026-10-22: The record id or path.
    pub id: String,
    /// 2026-10-22: The closure digest it carries.
    pub closure: Option<Sha256>,
}
