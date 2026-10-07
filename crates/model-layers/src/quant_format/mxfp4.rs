// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Native GPT-OSS packed E2M1/E8M0 identity, with no NVFP4 conversion or fallback.
use super::{IgnoreList, QuantFormat};
use crate::weight_map::{Nvfp4Variant, PackedMxfp4Experts};
use anyhow::{Context, Result, ensure};
use metrale_config::{
    ModelConfig,
    precision_plan::{IgnoreDialect, PlanSource},
};
use metrale_model_weights::weights::WeightStore;

#[derive(Debug)]
pub struct Mxfp4Format {
    ignore: IgnoreList,
}
impl Mxfp4Format {
    pub fn from_checkpoint(config: &ModelConfig, store: &WeightStore) -> Result<Self> {
        ensure!(
            config.model_type == "gpt_oss" && config.gpt_oss.is_some(),
            "MXFP4 layout requires parsed GPT-OSS policy"
        );
        let qc = config
            .quantization_config
            .as_ref()
            .context("MXFP4 declaration missing")?;
        ensure!(
            qc.quant_method == "mxfp4" && qc.precision.source == PlanSource::Mxfp4,
            "MXFP4 precision plan missing or incompatible"
        );
        ensure!(
            config.weight_prefix == "model" && config.num_hidden_layers > 0,
            "MXFP4 GPT layout requires explicit model layers"
        );
        let gate = config
            .moe_intermediate_size
            .checked_mul(2)
            .context("MXFP4 gate width overflow")?;
        for layer in 0..config.num_hidden_layers {
            for (projection, rows, cols) in [
                ("gate_up_proj", gate, config.hidden_size),
                (
                    "down_proj",
                    config.hidden_size,
                    config.moe_intermediate_size,
                ),
            ] {
                let prefix = format!("model.layers.{layer}.mlp.experts.{projection}");
                PackedMxfp4Experts::bind(
                    store,
                    &format!("{prefix}_blocks"),
                    &format!("{prefix}_scales"),
                    config.num_experts,
                    rows,
                    cols,
                )?;
            }
        }
        Ok(Self {
            ignore: IgnoreList::new(IgnoreDialect::ModelOpt, &qc.ignore_modules)?,
        })
    }
}
impl QuantFormat for Mxfp4Format {
    fn name(&self) -> &'static str {
        "mxfp4-e2m1-e8m0-native"
    }
    fn base_variant(&self) -> Option<Nvfp4Variant> {
        None
    }
    fn is_ignored(&self, module_path: &str) -> bool {
        self.ignore.matches(module_path)
    }
}
