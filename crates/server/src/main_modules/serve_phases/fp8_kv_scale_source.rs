// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Where an FP8 KV cache's per-layer K/V scales come from at run
//! time, and the serve log line that says so.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - The checkpoint count is `WeightStore::kv_scale_census`, which resolves keys
//!   with the resolver the per-layer loader uses (`weight_map::load_kv_scales`),
//!   so "N layers from the checkpoint" is the number of layers that load one.
//! - Online calibration replaces every FP8 layer's scale
//!   (`Qwen3AttentionLayer::effective_fp8_scales`), so with it on the checkpoint
//!   scales are loaded but not used.

use metrale_model_weights::weights::KvScaleCensus;

/// 2026-09-28: The run-time source of the FP8 KV scales across the attention
/// layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Fp8KvScaleSource {
    /// 2026-09-28: Online calibration over the first `tokens` tokens; the
    /// `shadowed` checkpoint scales are not used.
    Calibration { tokens: usize, shadowed: usize },
    /// 2026-09-28: Every attention layer uses its checkpoint scale.
    Checkpoint { layers: usize },
    /// 2026-09-28: `from_checkpoint` layers use a checkpoint scale; the other
    /// `attention_layers - from_checkpoint` run at 1.0.
    Partial {
        from_checkpoint: usize,
        attention_layers: usize,
    },
    /// 2026-09-28: No scales and no calibration: 1.0 on every layer.
    Unscaled { attention_layers: usize },
}

/// 2026-09-28: Classify the scale source. `checkpoint_layers` may exceed
/// `attention_layers` when a head outside the count (an MTP layer) ships
/// scales too; that is still every counted layer.
pub(crate) fn fp8_kv_scale_source(
    calibration_tokens: usize,
    checkpoint_layers: usize,
    attention_layers: usize,
) -> Fp8KvScaleSource {
    if calibration_tokens > 0 {
        Fp8KvScaleSource::Calibration {
            tokens: calibration_tokens,
            shadowed: checkpoint_layers,
        }
    } else if checkpoint_layers == 0 {
        Fp8KvScaleSource::Unscaled { attention_layers }
    } else if checkpoint_layers >= attention_layers {
        Fp8KvScaleSource::Checkpoint {
            layers: checkpoint_layers,
        }
    } else {
        Fp8KvScaleSource::Partial {
            from_checkpoint: checkpoint_layers,
            attention_layers,
        }
    }
}

/// 2026-09-28: Log the source. `auto_enabled` marks a calibration window taken
/// from MODEL.toml rather than the flag.
pub(crate) fn log_fp8_kv_scale_source(
    source: &Fp8KvScaleSource,
    census: &KvScaleCensus,
    auto_enabled: bool,
) {
    let spellings = census.spellings().join(", ");
    match *source {
        Fp8KvScaleSource::Calibration {
            tokens,
            shadowed: 0,
        } => {
            // 2026-09-26: Each attention layer logs "FP8 KV scales frozen after
            // N tokens (requested M)" when its window closes.
            tracing::info!(
                "FP8 KV scales: online calibration on every layer (checkpoint ships no k/v \
                 scales): accumulating per-tensor K/V amax over the first {tokens} observed \
                 tokens (across requests, readiness probe included) before freezing.{}",
                if auto_enabled {
                    " (auto-enabled from MODEL.toml)"
                } else {
                    ""
                },
            );
        }
        Fp8KvScaleSource::Calibration { tokens, shadowed } => tracing::info!(
            "FP8 KV scales: online calibration over {tokens} tokens on every layer; the \
             {shadowed} per-layer checkpoint scales ({spellings}) are loaded but NOT used. \
             Pass --fp8-kv-calibration-tokens 0 to use them.",
        ),
        Fp8KvScaleSource::Checkpoint { layers } => tracing::info!(
            "FP8 KV scales: from the checkpoint on all {layers} attention layers ({spellings}); \
             no calibration needed.",
        ),
        Fp8KvScaleSource::Partial {
            from_checkpoint,
            attention_layers,
        } => tracing::warn!(
            "FP8 KV scales: PARTIAL — the checkpoint ships scales ({spellings}) for only \
             {from_checkpoint} of {attention_layers} attention layers; the other {} run at \
             1.0, which wastes E4M3 range on small K/V and clips any |x| > 448. Enable \
             --fp8-kv-calibration-tokens 256, or use --kv-cache-dtype bf16/nvfp4.",
            attention_layers - from_checkpoint,
        ),
        Fp8KvScaleSource::Unscaled { attention_layers } => tracing::warn!(
            "FP8 KV scales: 1.0 on all {attention_layers} attention layers — the checkpoint \
             ships NO k/v scales and calibration is off. 1.0 wastes E4M3 range on small K/V \
             and clips any |x| > 448. Enable --fp8-kv-calibration-tokens 256 for online \
             calibration, or use --kv-cache-dtype nvfp4/bf16."
        ),
    }
}

#[cfg(test)]
#[path = "fp8_kv_scale_source_tests.rs"]
mod tests;
