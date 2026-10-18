// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: Parser for `deepseek_v41` (Hugging Face) configs. The model fields sit under
//! `text_config`; the outer object supplies the token ids and `quantization_config`. Every
//! field the model reads is required: an absent one is an error, never a default. The GGUF
//! route (`gguf.rs`, the `deepseek_v41` block) fills the same fields from `deepseek41.*`.
//!
//! Owner: config (model parsers).
//! Invariants:
//! - `rope_theta` is the uncompressed-layer base and `compress_rope_theta` the compressed one;
//!   neither overwrites the other (unlike `deepseek_v4`).
//! - The Engram hash multipliers, primes and offsets, and the compressed-vocab token map, are
//!   not in `config.json`; they stay empty here and the loader supplies them as data.

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;

use super::super::{LayerType, ModelConfig, finalize_config};
use super::deepseek_v4::validate_dspark_contract;

const FAMILY: &str = "DeepSeek-V4.1";

/// 2026-10-18: `text_config` keys the model reads, each required.
const REQUIRED: [&str; 41] = [
    "hidden_size",
    "num_hidden_layers",
    "vocab_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "qk_rope_head_dim",
    "q_lora_rank",
    "o_lora_rank",
    "o_groups",
    "moe_intermediate_size",
    "n_routed_experts",
    "n_shared_experts",
    "num_experts_per_tok",
    "norm_topk_prob",
    "routed_scaling_factor",
    "scoring_func",
    "topk_method",
    "swiglu_limit",
    "sliding_window",
    "compress_ratios",
    "compress_rope_theta",
    "rope_theta",
    "rope_scaling",
    "max_position_embeddings",
    "rms_norm_eps",
    "tie_word_embeddings",
    "hc_mult",
    "hc_sinkhorn_iters",
    "hc_eps",
    "index_n_heads",
    "index_head_dim",
    "index_topk",
    "kv_source_layer_ids",
    "index_source_layer_ids",
    "candidate_source_layer_id",
    "candidate_topk_blocks",
    "candidate_block_size",
    "engram_layer_ids",
    "num_nextn_predict_layers",
    "dspark_block_size",
];

/// 2026-10-18: `text_config` keys required when `engram_layer_ids` is non-empty.
const REQUIRED_ENGRAM: [&str; 7] = [
    "engram_num_embeddings",
    "engram_max_ngram_size",
    "engram_vocab_size",
    "engram_n_heads",
    "engram_head_dim",
    "engram_pad_token_id",
    "engram_compressed_vocab_size",
];

pub fn parse_deepseek_v41(json: &str) -> Result<ModelConfig> {
    let raw: Value = serde_json::from_str(json).context("Invalid JSON in DeepSeek-V4.1 config")?;
    let text = raw
        .get("text_config")
        .filter(|v| v.is_object())
        .with_context(|| format!("{FAMILY} config.json has no `text_config` object"))?;
    for key in REQUIRED {
        require(text, key)?;
    }
    let mut config: ModelConfig = serde_json::from_value(text.clone())
        .with_context(|| format!("Failed to parse {FAMILY} text_config"))?;

    config.bos_token_id = top_u32(&raw, "bos_token_id")?;
    config.eos_token_id = top_u32(&raw, "eos_token_id")?;

    let layers = config.num_hidden_layers;
    ensure!(layers > 0, "{FAMILY}: num_hidden_layers must be > 0");
    // 2026-10-18: One latent KV head shared by every query head (MQA).
    ensure!(
        config.num_key_value_heads == 1,
        "{FAMILY}: num_key_value_heads is {}, the attention has one latent KV head",
        config.num_key_value_heads
    );
    ensure!(
        config.o_groups > 0 && config.num_attention_heads.is_multiple_of(config.o_groups),
        "{FAMILY}: num_attention_heads ({}) is not a multiple of o_groups ({})",
        config.num_attention_heads,
        config.o_groups
    );
    ensure!(
        config.head_dim > config.qk_rope_head_dim && config.qk_rope_head_dim > 0,
        "{FAMILY}: qk_rope_head_dim ({}) must be in 1..head_dim ({})",
        config.qk_rope_head_dim,
        config.head_dim
    );
    // 2026-10-18: More entries than layers is allowed (the MTP layers follow), as on the GGUF
    // route; fewer is an error.
    ensure!(
        config.compress_ratios.len() >= layers,
        "{FAMILY}: compress_ratios has {} entries, fewer than num_hidden_layers ({layers})",
        config.compress_ratios.len()
    );
    let in_range = |ids: &[usize]| ids.iter().all(|&l| l < layers);
    ensure!(
        in_range(&config.kv_source_layer_ids) && in_range(&config.index_source_layer_ids),
        "{FAMILY}: kv/index source layer ids must be in 0..{layers}"
    );
    match config.candidate_source_layer_id {
        Some(l) if l < layers => {}
        other => bail!("{FAMILY}: candidate_source_layer_id {other:?} is not in 0..{layers}"),
    }

    let scoring = str_key(text, "scoring_func")?;
    ensure!(
        scoring == "sqrtsoftplus",
        "{FAMILY}: scoring_func {scoring:?} is not implemented; only sqrtsoftplus is"
    );
    let topk_method = str_key(text, "topk_method")?;
    ensure!(
        topk_method == "noaux_tc",
        "{FAMILY}: topk_method {topk_method:?} is not implemented; only noaux_tc is"
    );
    config.use_routing_bias = true;

    read_yarn(text, &mut config)?;

    if !config.engram_layer_ids.is_empty() {
        for key in REQUIRED_ENGRAM {
            require(text, key)?;
        }
        ensure!(
            in_range(&config.engram_layer_ids),
            "{FAMILY}: engram_layer_ids must be in 0..{layers}"
        );
        ensure!(
            config.engram_num_embeddings.len() == config.engram_layer_ids.len(),
            "{FAMILY}: engram_num_embeddings has {} entries for {} Engram layers",
            config.engram_num_embeddings.len(),
            config.engram_layer_ids.len()
        );
    }

    config.num_experts = config.n_routed_experts;
    config.shared_expert_intermediate_size = config.n_shared_experts * config.moe_intermediate_size;
    // 2026-10-18: The latent KV is one `head_dim`-wide vector; values are the latent itself.
    config.kv_lora_rank = config.head_dim;
    config.v_head_dim = config.head_dim;
    config.qk_nope_head_dim = config.head_dim - config.qk_rope_head_dim;
    config.rotary_dim = config.qk_rope_head_dim;
    config.partial_rotary_factor = config.qk_rope_head_dim as f64 / config.head_dim as f64;
    config.layer_types = vec![LayerType::FullAttention; layers];

    let mtp = u64_key(text, "num_nextn_predict_layers")? as usize;
    config.num_mtp_modules = mtp;
    config.mtp_transformer_layers = 1;
    config.mtp_num_hidden_layers = mtp;
    validate_dspark_contract(&config, text)?;

    config.model_type = "deepseek_v41".to_string();
    config.nested_config = true;
    config.attn_gated = false;
    // 2026-10-18: The same prefix the GGUF route sets; the checkpoint's own tensor names
    // (`layers.N.*`, no prefix) are the loader's to map.
    config.weight_prefix = "model".to_string();
    config.vision = None;

    finalize_config(&mut config, &raw)?;
    Ok(config)
}

fn require(text: &Value, key: &str) -> Result<()> {
    ensure!(
        text.get(key).is_some_and(|v| !v.is_null()),
        "{FAMILY} text_config has no `{key}`; refusing to guess"
    );
    Ok(())
}

fn str_key<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v.get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("{FAMILY}: `{key}` must be a string"))
}

fn u64_key(v: &Value, key: &str) -> Result<u64> {
    v.get(key)
        .and_then(Value::as_u64)
        .with_context(|| format!("{FAMILY}: `{key}` must be a non-negative integer"))
}

fn top_u32(raw: &Value, key: &str) -> Result<u32> {
    u32::try_from(u64_key(raw, key)?)
        .with_context(|| format!("{FAMILY}: `{key}` does not fit in u32"))
}

/// 2026-10-18: `rope_scaling` must be YaRN with all four parameters the model reads.
fn read_yarn(text: &Value, config: &mut ModelConfig) -> Result<()> {
    let rs = text
        .get("rope_scaling")
        .filter(|v| v.is_object())
        .with_context(|| format!("{FAMILY}: `rope_scaling` must be an object"))?;
    let kind = rs
        .get("rope_type")
        .or_else(|| rs.get("type"))
        .and_then(Value::as_str)
        .with_context(|| format!("{FAMILY}: rope_scaling has no rope_type"))?;
    ensure!(
        kind == "yarn",
        "{FAMILY}: rope_scaling type {kind:?} is not yarn"
    );
    let num = |key: &str| {
        rs.get(key)
            .and_then(Value::as_f64)
            .with_context(|| format!("{FAMILY}: rope_scaling has no numeric `{key}`"))
    };
    config.yarn_factor = num("factor")? as f32;
    config.yarn_beta_fast = num("beta_fast")? as f32;
    config.yarn_beta_slow = num("beta_slow")? as f32;
    config.yarn_original_max_position_embeddings =
        u64_key(rs, "original_max_position_embeddings")? as usize;
    Ok(())
}
