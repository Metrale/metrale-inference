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
}

pub fn loader_for_config_with_policy(
    config: &ModelConfig,
    policy: ExperimentalModelPolicy,
) -> Result<Box<dyn ModelWeightLoader>> {
    if config.model_type == "gpt_oss" && policy == ExperimentalModelPolicy::GptOssC1 {
        ensure!(
            config.tp_world_size <= 1 && config.ep_world_size <= 1,
            "experimental GPT-OSS requires single-device execution"
        );
        return Ok(Box::new(GptOssWeightLoader));
    }
    super::loader_for_config(config)
}

#[cfg(test)]
mod tests {
    use super::*;
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
