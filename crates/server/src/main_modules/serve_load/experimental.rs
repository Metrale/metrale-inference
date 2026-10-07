// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Fail before uploading weights when an experimental GPT request exceeds proven runtime mechanics.
use crate::cli::ServeArgs;
use anyhow::{Result, ensure};
use metrale_config::ModelConfig;
use metrale_model_engine::factory::SlotRequest;

pub(super) fn validate(args: &ServeArgs, config: &ModelConfig) -> Result<()> {
    if config.model_type != "gpt_oss" {
        ensure!(
            !args.experimental_gpt_oss,
            "--experimental-gpt-oss requires a GPT-OSS checkpoint"
        );
        return Ok(());
    }
    ensure!(
        args.experimental_gpt_oss,
        "GPT-OSS is uncertified; explicit --experimental-gpt-oss is required"
    );
    ensure!(
        super::super::serve::canonicalize_model_quant(config) == "mxfp4",
        "experimental GPT-OSS requires original MXFP4 weights"
    );
    ensure!(
        matches!(args.max_batch_size, SlotRequest::Count(1)),
        "experimental GPT-OSS requires --max-batch-size 1"
    );
    ensure!(
        args.kv_cache_dtype.as_deref().is_none_or(|s| s == "bf16"),
        "experimental GPT-OSS requires BF16 KV"
    );
    ensure!(
        !args.enable_prefix_caching,
        "experimental GPT-OSS cannot reuse cached prefixes"
    );
    ensure!(
        args.swap_space_gb == 0 && !args.high_speed_swap,
        "experimental GPT-OSS requires --swap-space-gb 0 and no high-speed swap"
    );
    ensure!(
        args.world_size == 1 && args.tp_size == 1 && args.ep_size == 1,
        "experimental GPT-OSS requires one device (no TP/EP)"
    );
    ensure!(
        !args.speculative_proposer_requested() && args.draft_model.is_none(),
        "experimental GPT-OSS does not support speculation"
    );
    ensure!(
        !args.prefill_varlen_batch && !args.prefill_codispatch,
        "experimental GPT-OSS requires serial prefill"
    );
    ensure!(
        args.lora_adapter.is_empty()
            && args.lora_stageable.is_empty()
            && args.lora_stageable_disk.is_empty(),
        "experimental GPT-OSS does not support LoRA"
    );
    ensure!(
        args.gpu_memory_utilization.is_finite()
            && args.gpu_memory_utilization > 0.0
            && args.gpu_memory_utilization <= 0.85,
        "experimental GPT-OSS requires memory utilization in (0, 0.85]"
    );
    ensure!(
        args.lm_head_dtype == "bf16",
        "experimental GPT-OSS requires --lm-head-dtype bf16"
    );
    ensure!(
        args.forward == crate::cli::flag_values::ForwardArg::Legacy,
        "experimental GPT-OSS requires --forward legacy; circuit lowering remains unsupported"
    );
    use metrale_model_layers::layers::ops;
    ensure!(
        !ops::prefill_varlen_enabled()
            && !ops::prefill_codispatch_enabled()
            && !ops::prefill_batched_first_chunk_enabled(),
        "experimental GPT-OSS cannot enable batched prefill through environment or CLI"
    );
    // 2026-10-07: LayerCapabilities veto graph capture and multi-sequence routes; decode additionally rejects either at entry.
    tracing::warn!(
        "Experimental GPT-OSS C1: numerical qualification open; graph capture, batching and prefix reuse are unsupported"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    fn fixture() -> (ServeArgs, ModelConfig) {
        let crate::cli::Command::Serve(args) = crate::cli::Cli::parse_from([
            "met",
            "serve",
            "checkpoint",
            "--experimental-gpt-oss",
            "--max-batch-size",
            "1",
            "--swap-space-gb",
            "0",
            "--gpu-memory-utilization",
            "0.85",
            "--lm-head-dtype",
            "bf16",
        ])
        .command
        else {
            panic!("serve fixture")
        };
        let config = metrale_config::parse_config(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../circuit/tests/fixtures/checkpoints/openai--gpt-oss-20b/config.json"
        )))
        .unwrap();
        (args, config)
    }
    #[test]
    fn opt_in_is_required_and_does_not_expand_to_other_models() {
        let (mut args, mut config) = fixture();
        assert!(validate(&args, &config).is_ok());
        args.experimental_gpt_oss = false;
        assert!(validate(&args, &config).is_err());
        args.experimental_gpt_oss = true;
        config.model_type = "qwen3".into();
        assert!(validate(&args, &config).is_err());
    }
    #[test]
    fn incompatible_runtime_options_are_refused_individually() {
        let (args, config) = fixture();
        let mutations: &[fn(&mut ServeArgs)] = &[
            |a| a.max_batch_size = SlotRequest::Count(2),
            |a| a.max_batch_size = SlotRequest::Auto,
            |a| a.kv_cache_dtype = Some("fp8".into()),
            |a| a.enable_prefix_caching = true,
            |a| a.swap_space_gb = 1,
            |a| a.high_speed_swap = true,
            |a| a.tp_size = 2,
            |a| a.ep_size = 2,
            |a| a.world_size = 2,
            |a| a.speculative = true,
            |a| a.prefill_varlen_batch = true,
            |a| a.prefill_codispatch = true,
            |a| a.gpu_memory_utilization = 0.86,
            |a| a.gpu_memory_utilization = f64::NAN,
            |a| a.gpu_memory_utilization = 0.0,
            |a| a.draft_model = Some("draft".into()),
            |a| a.lora_adapter.push(("adapter".into(), "path".into())),
            |a| a.forward = crate::cli::flag_values::ForwardArg::Circuit,
            |a| a.lm_head_dtype = "default".into(),
        ];
        for mutate in mutations {
            let mut candidate = args.clone();
            mutate(&mut candidate);
            assert!(
                validate(&candidate, &config).is_err(),
                "accepted {candidate:?}"
            );
        }
    }
}
