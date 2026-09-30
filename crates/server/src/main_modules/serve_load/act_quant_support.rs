// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: Which `--activation-quantization` families a model honours, checked before the
//! model is built. A family a model has no row-invariant path for runs `adaptive`, and the load
//! log names it; a format a family cannot run at all on this checkpoint is refused.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - Pure given the model config and the published value (`support`); `check` only logs.

use anyhow::{Result, bail};
use metrale_config::{ActQuantFormat, ActivationQuantization, ModelConfig, ProjFamily};

/// 2026-09-30: What the model is, as far as the fixed formats care.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModelKind {
    /// 2026-09-30: A Qwen3.5-family hybrid (GDN + attention), whose decode sites read the flag.
    pub qwen_hybrid: bool,
    /// 2026-09-30: MoE with FP8 experts and FP8 (W8A16-decoded) attention/GDN projections.
    pub fp8_moe: bool,
    /// 2026-09-30: `--lm-head-dtype fp8`.
    pub fp8_head: bool,
    /// 2026-09-30: `--weight-quantization declared`: a dense checkpoint's FP8-declared
    /// attention/GDN projections and MLPs decode W8A8 at every row count, and its A4 MLPs keep
    /// NVFP4 weights.
    pub declared_tier: bool,
}

impl ModelKind {
    pub(crate) fn of(config: &ModelConfig, lm_head_dtype: &str) -> Self {
        use metrale_config::WeightQuantization;
        let qwen_hybrid = config.num_ssm_layers() > 0 && config.model_type.starts_with("qwen3");
        Self {
            qwen_hybrid,
            fp8_moe: qwen_hybrid
                && config.num_experts > 0
                && super::super::serve::canonicalize_model_quant(config) == "fp8",
            fp8_head: lm_head_dtype == "fp8",
            declared_tier: metrale_model_layers::layers::weight_quantization().tier()
                == WeightQuantization::Declared,
        }
    }

    /// 2026-09-30: What happens to `family` at a fixed `format` on this model.
    fn classify(self, family: ProjFamily, format: ActQuantFormat) -> Support {
        use ActQuantFormat::*;
        use Support::*;
        let dense = self.qwen_hybrid && !self.fp8_moe;
        match family {
            ProjFamily::LmHead => match format {
                Fp8 if !self.fp8_head => {
                    Refused("the LM head runs 16-bit activations (fp8 needs --lm-head-dtype fp8)")
                }
                Nvfp4 => Refused("the LM head has no NVFP4-activation kernel"),
                _ => Honoured,
            },
            ProjFamily::Moe if self.fp8_moe => match format {
                Nvfp4 => Refused("nvfp4 activations need NVFP4 expert weights"),
                _ => Honoured,
            },
            ProjFamily::Gdn | ProjFamily::Attn if self.fp8_moe => match format {
                Fp8 | Nvfp4 => Refused(
                    "this checkpoint's attention/GDN projections decode W8A16; their \
                     block-scaled W8A8 path is off until it is re-validated",
                ),
                _ => Honoured,
            },
            ProjFamily::Gdn | ProjFamily::Attn if dense && self.declared_tier => match format {
                Declared | Fp8 => Honoured,
                _ => Refused("the declared W8A8 projections run FP8 activations only"),
            },
            ProjFamily::Ffn if dense => match format {
                Nvfp4 => Honoured,
                Declared if self.declared_tier => Honoured,
                Declared => Unhonoured,
                _ => Refused("the dense MLP runs a fixed format as NVFP4 or its declared one"),
            },
            // 2026-09-30: A family the model does not have: nothing to route.
            ProjFamily::Moe if dense => Honoured,
            ProjFamily::Ffn if self.fp8_moe => Honoured,
            _ => Unhonoured,
        }
    }
}

/// 2026-09-30: A family at a fixed format, on one model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Support {
    /// 2026-09-30: Its decode sites run the format at every row count.
    Honoured,
    /// 2026-09-30: The model has no invariant path for it here: it runs adaptive.
    Unhonoured,
    /// 2026-09-30: The checkpoint cannot run the format.
    Refused(&'static str),
}

/// 2026-09-30: The families `value` fixes that this model does not honour (they run
/// `adaptive`), or the first refusal.
pub(crate) fn support(value: &ActivationQuantization, kind: ModelKind) -> Result<Vec<ProjFamily>> {
    let mut unhonoured = Vec::new();
    for family in ProjFamily::ALL {
        let rungs = value.ladder(family).rungs();
        let fixed: Vec<ActQuantFormat> = rungs
            .iter()
            .map(|r| r.format)
            .filter(|&f| f != ActQuantFormat::Adaptive)
            .collect();
        if fixed.is_empty() {
            continue;
        }
        // 2026-09-30: The MoE expert activation format is one process-wide cell.
        if family == ProjFamily::Moe && kind.fp8_moe && rungs.len() > 1 {
            bail!(
                "--activation-quantization {value}: moe takes one format for every row count \
                 (the expert decode's activation format is chosen once per process)"
            );
        }
        for format in fixed {
            match kind.classify(family, format) {
                Support::Honoured => {}
                Support::Unhonoured => {
                    if !unhonoured.contains(&family) {
                        unhonoured.push(family);
                    }
                }
                Support::Refused(why) => bail!(
                    "--activation-quantization {value}: {} cannot run {} on this model: {why}",
                    family.name(),
                    format.name()
                ),
            }
        }
    }
    Ok(unhonoured)
}

/// 2026-09-30: `support` for the published value, logging what runs.
pub(crate) fn check(config: &ModelConfig, lm_head_dtype: &str) -> Result<()> {
    let value = metrale_model_layers::layers::activation_quantization();
    let unhonoured = support(value, ModelKind::of(config, lm_head_dtype))?;
    if !unhonoured.is_empty() {
        tracing::warn!(
            "--activation-quantization {value}: this model has no row-invariant decode path for \
             {}; those run adaptive (their activation precision still depends on the row count)",
            unhonoured
                .iter()
                .map(|f| f.name())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "act_quant_support_tests.rs"]
mod tests;
