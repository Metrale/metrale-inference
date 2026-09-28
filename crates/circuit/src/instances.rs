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
//!   `A` a full-attention layer; whitespace is ignored.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

use crate::fuser::Policy;
use crate::ir::{ArchShape, LayerKind};
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
    /// 2026-09-28: Precision table: `kernels/circuits/precision/<name>.toml`.
    pub precision: String,
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
    /// 2026-09-28: The golden file name of one plan: `<arch>-<mode>-n<rows>.txt`.
    pub fn plan_file(&self, mode: Mode, rows: u64) -> String {
        format!("{}-{}-n{rows}.txt", self.arch, mode.name())
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
    precision: String,
    target: String,
    golden: bool,
    layer_kinds: String,
    dims: BTreeMap<String, u64>,
    policy: PolicyFile,
    plans: BTreeMap<String, Vec<u64>>,
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
    let mut out = Vec::with_capacity(file.instance.len());
    for f in file.instance {
        let field = |detail: String| InstanceError::Field {
            recipe: f.recipe.clone(),
            detail,
        };
        if !seen.insert(f.recipe.clone()) {
            return Err(field("listed twice".into()));
        }
        let mut layer_kinds = Vec::new();
        for c in f.layer_kinds.chars().filter(|c| !c.is_whitespace()) {
            layer_kinds.push(match c {
                'G' => LayerKind::LinearAttention,
                'A' => LayerKind::FullAttention,
                other => return Err(field(format!("layer kind `{other}` is not G or A"))),
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
        out.push(Instance {
            recipe: f.recipe.clone(),
            checkpoint: f.checkpoint,
            arch: f.arch,
            precision: f.precision,
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
