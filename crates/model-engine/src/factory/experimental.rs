// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Explicit experimental admission leaves the supported factory dispatch unchanged.
use anyhow::{Result, ensure};
use metrale_config::ModelConfig;
use metrale_model_arch::weight_loader::{ModelWeightLoader, gpt_oss::loader::GptOssWeightLoader};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExperimentalModelPolicy {
    #[default]
    Disabled,
    GptOssC1,
    GptOssC1Chunked,
}

impl ExperimentalModelPolicy {
    // 2026-10-07: Lazy C1 scratch is charged before KV sizing, from allocator geometry.
    pub(super) fn layer_runtime_reserve(
        self,
        config: &ModelConfig,
        block_size: usize,
    ) -> Result<usize> {
        if self != Self::GptOssC1Chunked {
            return Ok(0);
        }
        ensure!(
            block_size > 0,
            "GPT chunk reserve requires nonzero block size"
        );
        metrale_model_arch::weight_loader::gpt_oss::runtime::PrefillScratch::required_bytes(
            16,
            config.max_position_embeddings.div_ceil(block_size),
        )?
        .checked_mul(config.num_hidden_layers)
        .ok_or_else(|| anyhow::anyhow!("GPT chunk scratch reserve overflow"))
    }
}

pub fn loader_for_config_with_policy(
    config: &ModelConfig,
    policy: ExperimentalModelPolicy,
) -> Result<Box<dyn ModelWeightLoader>> {
    if policy == ExperimentalModelPolicy::GptOssC1Chunked {
        ensure!(
            config.model_type == "gpt_oss",
            "chunk prefill requires GPT-OSS"
        );
    }
    if config.model_type == "gpt_oss" && policy != ExperimentalModelPolicy::Disabled {
        ensure!(
            config.tp_world_size <= 1 && config.ep_world_size <= 1,
            "experimental GPT-OSS requires single-device execution"
        );
        return Ok(Box::new(GptOssWeightLoader {
            chunk_prefill: policy == ExperimentalModelPolicy::GptOssC1Chunked,
        }));
    }
    super::loader_for_config(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chunk_policy_refuses_other_models_and_parallelism() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        let policy = ExperimentalModelPolicy::GptOssC1Chunked;
        assert!(loader_for_config_with_policy(&config, policy).is_err());
        config.model_type = "gpt_oss".into();
        assert!(loader_for_config_with_policy(&config, policy).is_ok());
        config.tp_world_size = 2;
        assert!(loader_for_config_with_policy(&config, policy).is_err());
        config.tp_world_size = 1;
        config.ep_world_size = 2;
        assert!(loader_for_config_with_policy(&config, policy).is_err());
    }
    #[test]
    fn explicit_policy_preserves_default_refusal_and_rejects_parallelism() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "gpt_oss".into();
        assert!(super::super::loader_for_config(&config).is_err());
        assert!(loader_for_config_with_policy(&config, ExperimentalModelPolicy::Disabled).is_err());
        assert!(loader_for_config_with_policy(&config, ExperimentalModelPolicy::GptOssC1).is_ok());
        config.tp_world_size = 2;
        assert!(loader_for_config_with_policy(&config, ExperimentalModelPolicy::GptOssC1).is_err());
        config.tp_world_size = 1;
        config.ep_world_size = 2;
        assert!(loader_for_config_with_policy(&config, ExperimentalModelPolicy::GptOssC1).is_err());
    }
}

// 2026-10-07: Reject padded/unmapped IDs before embedding lookup or any sequence mutation.
pub(crate) fn validate_gpt_tokens(config: &ModelConfig, tokens: &[u32]) -> Result<()> {
    if let Some(policy) = config.gpt_oss {
        ensure!(
            config.vocab_size > 0 && config.vocab_size <= policy.checkpoint_vocab_size,
            "GPT-OSS logical vocabulary exceeds physical storage"
        );
        ensure!(
            tokens.iter().all(|&id| (id as usize) < config.vocab_size),
            "GPT-OSS token ID is outside the logical tokenizer vocabulary"
        );
    }
    Ok(())
}

#[cfg(test)]
mod token_bounds_tests {
    use super::*;
    #[test]
    fn logical_boundary_excludes_physical_padding_and_overflow_ids() {
        let mut config = metrale_config::parse_config(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../circuit/tests/fixtures/checkpoints/openai--gpt-oss-20b/config.json"
        )))
        .unwrap();
        config.vocab_size = 200019;
        assert!(validate_gpt_tokens(&config, &[0, 200018]).is_ok());
        for id in [200019, 201087, 201088, u32::MAX] {
            assert!(validate_gpt_tokens(&config, &[id]).is_err());
        }
    }
}
