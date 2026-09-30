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
}

impl ModelKind {
    pub(crate) fn of(config: &ModelConfig, lm_head_dtype: &str) -> Self {
        let qwen_hybrid = config.num_ssm_layers() > 0 && config.model_type.starts_with("qwen3");
        Self {
            qwen_hybrid,
            fp8_moe: qwen_hybrid
                && config.num_experts > 0
                && super::super::serve::canonicalize_model_quant(config) == "fp8",
            fp8_head: lm_head_dtype == "fp8",
        }
    }

    /// 2026-09-30: The families whose decode sites run the fixed formats on this model.
    pub(crate) fn honoured(self) -> Vec<ProjFamily> {
        let mut v = vec![ProjFamily::LmHead];
        if self.fp8_moe {
            v.extend([ProjFamily::Gdn, ProjFamily::Attn, ProjFamily::Moe]);
        }
        v
    }

    /// 2026-09-30: Why `family` cannot run `format` on this model, if it cannot.
    fn refuses(self, family: ProjFamily, format: ActQuantFormat) -> Option<&'static str> {
        use ActQuantFormat::*;
        match (family, format) {
            (_, Adaptive | Declared | Bf16) => None,
            (ProjFamily::LmHead, Fp8) if self.fp8_head => None,
            (ProjFamily::LmHead, _) => Some(
                "the LM head runs 16-bit activations (fp8 needs --lm-head-dtype fp8; nvfp4 has no \
                 head kernel)",
            ),
            (ProjFamily::Moe, Fp8) if self.fp8_moe => None,
            (ProjFamily::Moe, _) => Some("nvfp4 activations need NVFP4 expert weights"),
            (ProjFamily::Gdn | ProjFamily::Attn, _) if self.fp8_moe => Some(
                "this checkpoint's attention/GDN projections decode W8A16; their block-scaled \
                 W8A8 path is off until it is re-validated",
            ),
            _ => None,
        }
    }
}

/// 2026-09-30: The families `value` fixes that this model does not honour (they run
/// `adaptive`), or the first refusal.
pub(crate) fn support(value: &ActivationQuantization, kind: ModelKind) -> Result<Vec<ProjFamily>> {
    let honoured = kind.honoured();
    let mut unhonoured = Vec::new();
    for family in ProjFamily::ALL {
        let rungs = value.ladder(family).rungs();
        if rungs.iter().all(|r| r.format == ActQuantFormat::Adaptive) {
            continue;
        }
        if !honoured.contains(&family) {
            unhonoured.push(family);
            continue;
        }
        for r in rungs {
            if let Some(why) = kind.refuses(family, r.format) {
                bail!(
                    "--activation-quantization {value}: {} cannot run {} on this model: {why}",
                    family.name(),
                    r.format.name()
                );
            }
        }
        // 2026-09-30: The MoE expert activation format is one process-wide cell.
        if family == ProjFamily::Moe && rungs.len() > 1 {
            bail!(
                "--activation-quantization {value}: moe takes one format for every row count \
                 (the expert decode's activation format is chosen once per process)"
            );
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
