// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `kernels/circuits/INSTANCES.toml`: which recipes have a circuit, with the arch
//! shape, precision table, policy and plan matrix each one is shown and checked under.
//!
//! The shape is stated here, not read from the checkpoint, so the golden plans are checked
//! without the checkpoint on disk. `met circuit show` compares it against the checkpoint's
//! `config.json` when that is in the local cache.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Every policy setting is stated; nothing is filled from an engine default.
//! - `layer_kinds` spells one letter per layer: `G` a GatedDeltaNet (linear attention) layer,
//!   `A` a full-attention layer, `M` a Mamba2 layer, `E` a MoE-only layer (the Nemotron-H
//!   hybrid-pattern letters); whitespace is ignored.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

use crate::format::Format;
use crate::fuser::Policy;
use crate::ir::{ArchShape, LayerKind};
use crate::precision::LinearFormats;
use crate::rules::Mode;

/// 2026-09-28: One recipe's circuit instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instance {
    /// 2026-09-28: Recipe id (`qwen3.8/qwen3.8-27b-nvfp4-unsloth`).
    pub recipe: String,
    /// 2026-09-28: Checkpoint id.
    pub checkpoint: String,
    /// 2026-09-28: Circuit arch: `kernels/circuits/<arch>.toml`.
    pub arch: String,
    /// 2026-09-30: What tells this instance's golden plans apart from another golden instance
    /// of the same arch (the recipe served under another `--weight-quantization` tier); `None`
    /// for the arch's only golden instance.
    pub variant: Option<String>,
    /// 2026-09-28: Where each linear module's formats come from.
    pub precision: PrecisionSpec,
    /// 2026-09-28: Kernel target, `hw/model/quant`.
    pub target: String,
    /// 2026-09-28: Whether its plans are checked in under `kernels/circuits/plans/`.
    pub golden: bool,
    /// 2026-09-28: The arch shape.
    pub shape: ArchShape,
    /// 2026-09-28: The policy.
    pub policy: Policy,
    /// 2026-09-28: Row counts per mode that have a plan.
    pub plans: BTreeMap<Mode, Vec<u64>>,
}

impl Instance {
    /// 2026-09-28: The golden file name of one plan: `<stem>-<mode>-n<rows>.txt`.
    pub fn plan_file(&self, mode: Mode, rows: u64) -> String {
        format!("{}-{}-n{rows}.txt", self.plan_stem(), mode.name())
    }

    /// 2026-09-30: `<arch>`, or `<arch>.<variant>` for a variant: the prefix of every golden
    /// file of this instance.
    pub fn plan_stem(&self) -> String {
        match &self.variant {
            Some(v) => format!("{}.{v}", self.arch),
            None => self.arch.clone(),
        }
    }
}

/// 2026-09-28: An instance's source of linear formats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrecisionSpec {
    /// 2026-09-28: A stated table, `kernels/circuits/precision/<name>.toml`.
    Table(String),
    /// 2026-09-28: The engine's policy over the checkpoint's declared plan
    /// ([`crate::precision_policy::PolicyPrecision`]).
    Policy {
        /// 2026-09-28: The plan fixture, `kernels/circuits/checkpoints/<name>.toml`.
        checkpoint_plan: String,
        /// 2026-09-28: The `--weight-quantization` tier served.
        tier: String,
        /// 2026-09-28: The kernel capabilities present (`KernelCaps` field names).
        caps: Vec<String>,
        /// 2026-09-28: Formats the engine chooses itself, first match wins.
        engine: Vec<(String, LinearFormats)>,
    },
}

impl PrecisionSpec {
    /// 2026-09-28: The name of the file the spec reads: the table or the plan fixture.
    pub fn file_name(&self) -> &str {
        match self {
            PrecisionSpec::Table(n) => n,
            PrecisionSpec::Policy {
                checkpoint_plan, ..
            } => checkpoint_plan,
        }
    }
}

/// 2026-09-28: Why INSTANCES.toml did not load.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InstanceError {
    /// 2026-09-28: Not valid TOML or not the shape.
    #[error("INSTANCES.toml: {0}")]
    Parse(String),
    /// 2026-09-28: A bad field in one instance.
    #[error("instance `{recipe}`: {detail}")]
    Field {
        /// 2026-09-28: Recipe id.
        recipe: String,
        /// 2026-09-28: What was wrong.
        detail: String,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    schema: u32,
    instance: Vec<InstanceFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InstanceFile {
    recipe: String,
    checkpoint: String,
    arch: String,
    variant: Option<String>,
    precision: PrecisionFile,
    target: String,
    golden: bool,
    layer_kinds: String,
    dims: BTreeMap<String, u64>,
    policy: PolicyFile,
    plans: BTreeMap<String, Vec<u64>>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum PrecisionFile {
    Table(String),
    Policy(PolicyPrecisionFile),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyPrecisionFile {
    checkpoint_plan: String,
    tier: String,
    caps: Vec<String>,
    engine: Vec<EngineFormatFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EngineFormatFile {
    #[serde(rename = "match")]
    pattern: String,
    weight: String,
    activation: String,
    /// 2026-09-28: Why the engine, not the checkpoint, decides this module's format.
    why: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyFile {
    opt_in_levers: Vec<String>,
    settings: BTreeMap<String, String>,
}

/// 2026-09-28: Parse INSTANCES.toml text.
pub fn parse_instances(text: &str) -> Result<Vec<Instance>, InstanceError> {
    let file: File = toml::from_str(text).map_err(|e| InstanceError::Parse(e.to_string()))?;
    if file.schema != 1 {
        return Err(InstanceError::Parse(format!(
            "schema {} (this build reads 1)",
            file.schema
        )));
    }
    let mut seen = BTreeSet::new();
    let mut stems = BTreeSet::new();
    let mut out = Vec::with_capacity(file.instance.len());
    for f in file.instance {
        let field = |detail: String| InstanceError::Field {
            recipe: f.recipe.clone(),
            detail,
        };
        if !seen.insert(f.recipe.clone()) {
            return Err(field("listed twice".into()));
        }
        if f.variant.as_deref().is_some_and(|v| v.is_empty()) {
            return Err(field("`variant` is empty".into()));
        }
        let stem = (f.arch.clone(), f.variant.clone());
        if f.golden && !stems.insert(stem) {
            return Err(field(format!(
                "a second golden `{}` instance without its own `variant`: the golden plan files \
                 would collide",
                f.arch
            )));
        }
        let mut layer_kinds = Vec::new();
        for c in f.layer_kinds.chars().filter(|c| !c.is_whitespace()) {
            layer_kinds.push(match c {
                'G' => LayerKind::LinearAttention,
                'A' => LayerKind::FullAttention,
                'M' => LayerKind::Mamba,
                'E' => LayerKind::Moe,
                other => return Err(field(format!("layer kind `{other}` is not G, A, M or E"))),
            });
        }
        let mut plans = BTreeMap::new();
        for (m, rows) in &f.plans {
            let mode = Mode::parse(m).ok_or_else(|| field(format!("unknown mode `{m}`")))?;
            if rows.is_empty() || rows.contains(&0) {
                return Err(field(format!("mode `{m}` needs row counts of at least 1")));
            }
            plans.insert(mode, rows.clone());
        }
        let precision = match f.precision {
            PrecisionFile::Table(name) => PrecisionSpec::Table(name),
            PrecisionFile::Policy(p) => {
                let mut engine = Vec::with_capacity(p.engine.len());
                for e in p.engine {
                    if e.why.trim().is_empty() {
                        return Err(field(format!(
                            "engine format `{}` states no `why`",
                            e.pattern
                        )));
                    }
                    let fmt = |s: &str| {
                        Format::parse(s)
                            .map_err(|err| field(format!("engine format `{}`: {err}", e.pattern)))
                    };
                    let formats = LinearFormats {
                        weight: fmt(&e.weight)?,
                        activation: fmt(&e.activation)?,
                    };
                    engine.push((e.pattern, formats));
                }
                PrecisionSpec::Policy {
                    checkpoint_plan: p.checkpoint_plan,
                    tier: p.tier,
                    caps: p.caps,
                    engine,
                }
            }
        };
        out.push(Instance {
            recipe: f.recipe.clone(),
            checkpoint: f.checkpoint,
            arch: f.arch,
            variant: f.variant,
            precision,
            target: f.target,
            golden: f.golden,
            shape: ArchShape {
                layer_kinds,
                dims: f.dims,
            },
            policy: Policy {
                opt_in_levers: f.policy.opt_in_levers.into_iter().collect(),
                settings: f.policy.settings,
            },
            plans,
        });
    }
    Ok(out)
}
