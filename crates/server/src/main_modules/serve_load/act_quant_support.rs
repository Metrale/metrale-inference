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
    /// 2026-10-08: GLM-5.3 (`glm5_next`), whose routed experts and dense MLP read the `moe` and
    /// `ffn` ladders (`glm5next_mlp::precision`).
    pub glm5_next: bool,
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
            glm5_next: config.model_type == "glm5_next",
        }
    }

    /// 2026-09-30: What happens to `family` at a fixed `format` on this model.
    fn classify(self, family: ProjFamily, format: ActQuantFormat) -> Support {
        use ActQuantFormat::*;
        use Support::*;
        let dense = self.qwen_hybrid && !self.fp8_moe;
        match family {
            // 2026-10-08: GLM-5.3's routed experts and dense MLP run their declared W4A4 (or a
            // named nvfp4) on row-invariant kernels: the dense MLP at every width, the experts
            // up to the slot GEMV's row cap, past which the grouped W4A16 GEMM runs and the
            // model's load log names those widths as above declared. Their 16-bit paths are not
            // row-invariant, so a fixed bf16 runs adaptive.
            ProjFamily::Ffn | ProjFamily::Moe if self.glm5_next => match format {
                Fp8 => Refused("GLM-5.3 has no FP8-activation MLP kernels"),
                Declared | Nvfp4 => Honoured,
                Bf16 | Adaptive => Unhonoured,
            },
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
            // 2026-10-01: On the nvfp4 tier the dense GDN and attention projections hold NVFP4
            // weights, which a fixed `nvfp4` format runs on the row-invariant W4A4 mx path.
            ProjFamily::Gdn | ProjFamily::Attn if dense => match format {
                Nvfp4 => Honoured,
                _ => Unhonoured,
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
            // 2026-10-01: The fixed `nvfp4` arms of the attention and GDN layers serve the Qwen3.5
            // family only; elsewhere they would cover some decode sites and not others.
            ProjFamily::Gdn | ProjFamily::Attn if format == Nvfp4 => Refused(
                "the row-invariant nvfp4 attention/GDN path serves the Qwen3.5-family hybrids only",
            ),
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

/// 2026-10-01: The refusal of the cross-sequence prefill levers beside a fixed format, given their
/// resolved values (flag or environment fallback): `--prefill-varlen-batch`,
/// `--prefill-codispatch` and the batched first chunk prefill prompts that arrive together in one
/// forward whose GEMMs are sized by the wave's total tokens, so a prompt's prefill, and from it
/// its whole output, depends on its wave-mates. `None` under `adaptive`, or with all three off.
pub(crate) fn prefill_lever_refusal(
    value: &ActivationQuantization,
    varlen: bool,
    codispatch: bool,
    first_chunk: bool,
) -> Option<String> {
    if value.is_adaptive() {
        return None;
    }
    let on: Vec<&str> = [
        (varlen, "--prefill-varlen-batch (or METRALE_PREFILL_VARLEN)"),
        (
            codispatch,
            "--prefill-codispatch (or METRALE_PREFILL_CODISPATCH)",
        ),
        (
            first_chunk && !codispatch,
            "METRALE_Q12_BATCHED_FIRST_CHUNK",
        ),
    ]
    .into_iter()
    .filter_map(|(set, name)| set.then_some(name))
    .collect();
    (!on.is_empty()).then(|| {
        format!(
            "--activation-quantization {value} with {}: these prefill concurrently arriving \
             prompts in one forward sized by the wave's total tokens, so a prompt's output \
             depends on its wave-mates; drop them, or add --activation-quantization adaptive",
            on.join(" and ")
        )
    })
}

/// 2026-09-30: `support` for the published value, logging what runs. 2026-10-01: Also refuses the
/// cross-sequence prefill levers beside a fixed format ([`prefill_lever_refusal`]).
pub(crate) fn check(config: &ModelConfig, lm_head_dtype: &str) -> Result<()> {
    let value = metrale_model_layers::layers::activation_quantization();
    {
        use metrale_model_layers::layers::ops;
        if let Some(why) = prefill_lever_refusal(
            value,
            ops::prefill_varlen_enabled(),
            ops::prefill_codispatch_enabled(),
            ops::prefill_batched_first_chunk_enabled(),
        ) {
            bail!("{why}");
        }
    }
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
