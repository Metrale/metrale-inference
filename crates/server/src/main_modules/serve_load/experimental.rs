// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Fail before uploading weights when an experimental GPT request exceeds proven runtime mechanics.
use crate::cli::ServeArgs;
use anyhow::{Result, ensure};
use metrale_config::ModelConfig;
use metrale_model_engine::factory::SlotRequest;

pub(super) fn validate(args: &ServeArgs, config: &ModelConfig) -> Result<()> {
    ensure!(
        !args.experimental_gpt_oss_chunk_prefill
            || args.experimental_gpt_oss_chunk_tokens.is_some(),
        "--experimental-gpt-oss-chunk-prefill requires --experimental-gpt-oss-chunk-tokens 16, 64 or 128"
    );
    if let Some(tokens) = args.experimental_gpt_oss_chunk_tokens {
        ensure!(
            args.experimental_gpt_oss_chunk_prefill && [16, 64, 128].contains(&tokens),
            "--experimental-gpt-oss-chunk-tokens requires chunk prefill and capacity 16, 64 or 128"
        );
    }
    if config.model_type != "gpt_oss" {
        ensure!(
            !args.experimental_gpt_oss
                && !args.experimental_gpt_oss_chunk_prefill
                && args.experimental_gpt_oss_chunk_tokens.is_none(),
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
        !args.prefix_caching_enabled(),
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
#[path = "experimental_tests.rs"]
mod tests;
