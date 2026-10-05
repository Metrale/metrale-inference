// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: [`PolicyPrecision`], the [`EdgePrecision`] the engine's own precision answers
//! give: the checkpoint's declared plan (`DeclaredPrecisionPlan`) under the served tier
//! (`WeightQuantPolicy`), plus the formats the engine chooses itself (a recipe's lm_head, the
//! MTP head), which the checkpoint does not declare.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Engine formats are stated, never inferred: a module they name is answered from them
//!   first, in order; every other module from the policy.
//! - Under the `declared` tier a module declared FP8 runs its FP8 weights, with FP8
//!   activations exactly when `WeightQuantPolicy::fp8_decode_act` says so; one declared FP4
//!   runs NVFP4 weights, with FP4 activations when the checkpoint declares them. Under
//!   `nvfp4` every quantized module runs NVFP4 weights and 16-bit activations.
//! - An unquantized module is BF16 weights and activations.

use metrale_config::precision_plan::{Granularity, Operand};
use metrale_config::weight_quantization::{ActFormat, KernelCaps};
use metrale_config::{
    DeclaredPrecisionPlan, W4a4Downcast, WeightQuantPolicy, WeightQuantTier, WeightQuantization,
};
use serde::Deserialize;

use crate::format::{Format, Scale};
use crate::precision::{EdgePrecision, LinearFormats, PrecisionError, glob};

/// 2026-09-28: The NVFP4 group size the engine's kernels read.
const NVFP4_GROUP: u32 = 16;

/// 2026-09-28: A checkpoint's `quantization_config`, as the circuit fixtures store it
/// (`kernels/circuits/checkpoints/<name>.toml`).
#[derive(Debug, Clone)]
pub struct CheckpointPlan {
    /// 2026-09-28: The checkpoint id.
    pub checkpoint: String,
    /// 2026-09-28: Its declared plan.
    pub plan: DeclaredPrecisionPlan,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureFile {
    schema: u32,
    checkpoint: String,
    quantization_config: String,
}

impl CheckpointPlan {
    /// 2026-09-28: Parse a fixture: `checkpoint` and the `quantization_config` object as JSON.
    pub fn parse(text: &str) -> Result<Self, PrecisionError> {
        let f: FixtureFile =
            toml::from_str(text).map_err(|e| PrecisionError::Parse(e.to_string()))?;
        if f.schema != 1 {
            return Err(PrecisionError::Parse(format!(
                "checkpoint fixture schema {} (this build reads 1)",
                f.schema
            )));
        }
        let qc: serde_json::Value = serde_json::from_str(&f.quantization_config)
            .map_err(|e| PrecisionError::Parse(format!("quantization_config: {e}")))?;
        let plan = DeclaredPrecisionPlan::from_quantization_config(&qc)
            .map_err(|e| PrecisionError::Parse(format!("quantization_config: {e:#}")))?;
        Ok(CheckpointPlan {
            checkpoint: f.checkpoint,
            plan,
        })
    }
}

/// 2026-09-28: The tier named `name` (`declared`, `nvfp4`).
pub fn tier_named(name: &str) -> Result<WeightQuantTier, PrecisionError> {
    let tier = WeightQuantization::ALL
        .into_iter()
        .find(|t| t.name() == name)
        .ok_or_else(|| {
            PrecisionError::Parse(format!("unknown weight-quantization tier `{name}`"))
        })?;
    WeightQuantTier::new(tier, W4a4Downcast::Off).map_err(|e| PrecisionError::Parse(e.to_string()))
}

/// 2026-09-28: The kernel capabilities named `names` (field names of `KernelCaps`).
pub fn caps_named(names: &[String]) -> Result<KernelCaps, PrecisionError> {
    let mut caps = KernelCaps::default();
    for n in names {
        let bit = match n.as_str() {
            "w8a8_decode" => &mut caps.w8a8_decode,
            "w8a8_moe_decode" => &mut caps.w8a8_moe_decode,
            "w8a8_block_scaled_decode" => &mut caps.w8a8_block_scaled_decode,
            "fp8_lm_head_batched" => &mut caps.fp8_lm_head_batched,
            other => {
                return Err(PrecisionError::Parse(format!(
                    "unknown kernel cap `{other}`"
                )));
            }
        };
        *bit = true;
    }
    Ok(caps)
}

/// 2026-09-28: The engine's precision answers as an [`EdgePrecision`].
pub struct PolicyPrecision<'a> {
    policy: WeightQuantPolicy<'a>,
    engine: &'a [(String, LinearFormats)],
}

impl<'a> PolicyPrecision<'a> {
    /// 2026-09-28: `engine` lists the engine-chosen formats, first match wins.
    pub fn new(policy: WeightQuantPolicy<'a>, engine: &'a [(String, LinearFormats)]) -> Self {
        PolicyPrecision { policy, engine }
    }
}

const BF16: LinearFormats = LinearFormats {
    weight: Format::Bf16,
    activation: Format::Bf16,
};

fn fp8_weight(w: Operand) -> Format {
    Format::Fp8E4m3 {
        scale: match w.granularity {
            Granularity::Block(r, c) => Scale::Block(r, c),
            Granularity::Group(g) => Scale::Group(g),
            Granularity::Tensor => Scale::PerTensor,
            _ => Scale::PerChannel,
        },
    }
}

fn fp8_act(a: Option<Operand>) -> Format {
    Format::Fp8E4m3 {
        scale: match a.map(|a| a.granularity) {
            Some(Granularity::Group(g)) => Scale::Group(g),
            Some(Granularity::Tensor) => Scale::PerTensor,
            _ => Scale::PerToken,
        },
    }
}

impl EdgePrecision for PolicyPrecision<'_> {
    fn linear(&self, module: &str) -> LinearFormats {
        if let Some((_, f)) = self.engine.iter().find(|(p, _)| glob(p, module)) {
            return *f;
        }
        // 2026-10-02: As `DeclaredPrecision`: an expert projection the plan does not name is
        // declared by its experts module (`precision::expert_container`).
        let mut declared = self.policy.declared(module);
        if declared.weight.is_none()
            && let Some(container) = crate::precision::expert_container(module)
        {
            declared = self.policy.declared(container);
        }
        let Some(w) = declared.weight else {
            return BF16;
        };
        let nvfp4 = Format::Nvfp4 { group: NVFP4_GROUP };
        if !self.policy.follows_plan() {
            return LinearFormats {
                weight: nvfp4,
                activation: Format::Bf16,
            };
        }
        if w.is_fp8() {
            // 2026-10-03: A 128x128 block-scaled FP8 weight outside the experts decodes W8A8 only
            // under the block-scaled kernel cap, as the loader adopts it (`fp8_block_scaled_decode_act`).
            let act = if matches!(w.granularity, Granularity::Block(..)) {
                self.policy.fp8_block_scaled_decode_act(module)
            } else {
                self.policy.fp8_decode_act(module)
            };
            let activation = match act {
                Some(ActFormat::Fp8) => fp8_act(declared.activation),
                _ => Format::Bf16,
            };
            return LinearFormats {
                weight: fp8_weight(w),
                activation,
            };
        }
        LinearFormats {
            weight: nvfp4,
            activation: if declared.activation_is_fp4() {
                nvfp4
            } else {
                Format::Bf16
            },
        }
    }
}

#[cfg(test)]
#[path = "precision_policy_tests.rs"]
mod precision_policy_tests;
