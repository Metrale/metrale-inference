// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: What makes a checkpoint an embedding model rather than a generative one: how
//! its attention is masked, how token states pool into one vector, the role prompts it was
//! trained with and its per-role input limits; and the Llama-3 RoPE frequency table such a
//! model may declare. The family parser fills [`EmbeddingSpec`] from `config.json`; serve
//! fills the prompts from the checkpoint's `config_sentence_transformers.json`.
//!
//! Owner: config.
//! Invariants:
//! - A config with `embedding = Some(_)` declares `generates = false`: serve refuses
//!   generation on it and answers `/v1/embeddings` instead.
//! - Every field is read from the checkpoint; none has a default.

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;

/// 2026-10-24: How token states become one vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// 2026-10-24: The mean over every input token, prompt included, of the final-norm
    /// hidden state, accumulated in FP32.
    Mean,
}

/// 2026-10-24: Which keys a query may attend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionMask {
    /// 2026-10-24: Every token of the same input, no causal limit.
    Bidirectional,
}

/// 2026-10-24: The role an input plays to an asymmetric embedder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedRole {
    Query,
    Document,
}

impl EmbedRole {
    /// 2026-10-24: The `input_type` names `/v1/embeddings` accepts for each role. The names are
    /// the ones the checkpoint's own processor and OpenAI-compatible hosts use.
    pub fn from_input_type(s: &str) -> Option<Self> {
        match s {
            "query" => Some(Self::Query),
            "passage" | "document" => Some(Self::Document),
            _ => None,
        }
    }
}

/// 2026-10-24: The prompt prepended to each role's text, from
/// `config_sentence_transformers.json` `prompts`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingPrompts {
    pub query: String,
    pub document: String,
}

/// 2026-10-24: The embedding semantics of a checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingSpec {
    pub pooling: Pooling,
    pub attention: AttentionMask,
    /// 2026-10-24: Longest query, in tokens with BOS and prompt (`q_max_length`).
    pub query_max_tokens: usize,
    /// 2026-10-24: Longest document, in tokens with BOS and prompt (`p_max_length`).
    pub document_max_tokens: usize,
    /// 2026-10-24: Set by serve from the checkpoint directory; `None` until then, and a serve
    /// that finds none refuses the checkpoint.
    pub prompts: Option<EmbeddingPrompts>,
}

impl EmbeddingSpec {
    /// 2026-10-24: The prompt and token limit for `role`; an error until the prompts are set.
    pub fn role(&self, role: EmbedRole) -> Result<(&str, usize)> {
        let p = self
            .prompts
            .as_ref()
            .context("embedding prompts were never loaded (config_sentence_transformers.json)")?;
        Ok(match role {
            EmbedRole::Query => (p.query.as_str(), self.query_max_tokens),
            EmbedRole::Document => (p.document.as_str(), self.document_max_tokens),
        })
    }
}

/// 2026-10-24: Parse `config_sentence_transformers.json`: both `prompts.query` and
/// `prompts.document` are required.
pub fn parse_sentence_transformers_prompts(json: &str) -> Result<EmbeddingPrompts> {
    let raw: Value =
        serde_json::from_str(json).context("config_sentence_transformers.json is not JSON")?;
    let prompts = raw
        .get("prompts")
        .context("config_sentence_transformers.json has no `prompts`")?;
    let get = |k: &str| -> Result<String> {
        prompts
            .get(k)
            .and_then(Value::as_str)
            .map(str::to_string)
            .with_context(|| format!("config_sentence_transformers.json has no `prompts.{k}`"))
    };
    Ok(EmbeddingPrompts {
        query: get("query")?,
        document: get("document")?,
    })
}

/// 2026-10-24: `rope_scaling` with `rope_type = "llama3"` (transformers
/// `_compute_llama3_parameters`): low frequencies divided by `factor`, high ones kept, and a
/// smooth blend between the two wavelength cut-offs. The attention factor is 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Llama3Rope {
    pub factor: f64,
    pub low_freq_factor: f64,
    pub high_freq_factor: f64,
    pub original_max_position_embeddings: f64,
}

impl Llama3Rope {
    /// 2026-10-24: Read `rope_scaling`; any rope type other than `llama3`, or a missing key,
    /// is an error.
    pub fn from_rope_scaling(scaling: &Value) -> Result<Self> {
        let ty = scaling.get("rope_type").and_then(Value::as_str);
        ensure!(
            ty == Some("llama3"),
            "rope_scaling.rope_type {ty:?} is not llama3; no other scaling is mapped for this family"
        );
        let num = |k: &str| -> Result<f64> {
            scaling
                .get(k)
                .and_then(Value::as_f64)
                .with_context(|| format!("rope_scaling.{k} missing or not a number"))
        };
        let r = Self {
            factor: num("factor")?,
            low_freq_factor: num("low_freq_factor")?,
            high_freq_factor: num("high_freq_factor")?,
            original_max_position_embeddings: num("original_max_position_embeddings")?,
        };
        ensure!(
            r.factor > 0.0
                && r.high_freq_factor > r.low_freq_factor
                && r.low_freq_factor > 0.0
                && r.original_max_position_embeddings > 0.0,
            "rope_scaling llama3 parameters out of range: {r:?}"
        );
        Ok(r)
    }

    /// 2026-10-24: The `rotary_dim / 2` inverse frequencies at `theta`, evaluated in FP64 and
    /// rounded once to FP32.
    pub fn inv_freq(&self, theta: f64, rotary_dim: usize) -> Vec<f32> {
        let low_wavelen = self.original_max_position_embeddings / self.low_freq_factor;
        let high_wavelen = self.original_max_position_embeddings / self.high_freq_factor;
        (0..rotary_dim / 2)
            .map(|i| {
                let f = 1.0 / theta.powf((2 * i) as f64 / rotary_dim as f64);
                let wavelen = 2.0 * std::f64::consts::PI / f;
                let scaled = if wavelen < high_wavelen {
                    f
                } else if wavelen > low_wavelen {
                    f / self.factor
                } else {
                    let s = (self.original_max_position_embeddings / wavelen
                        - self.low_freq_factor)
                        / (self.high_freq_factor - self.low_freq_factor);
                    (1.0 - s) * f / self.factor + s * f
                };
                scaled as f32
            })
            .collect()
    }
}

/// 2026-10-24: `true` when `architectures` names a bidirectional Llama.
fn bidirectional(llm: &Value) -> Result<AttentionMask> {
    let arch = llm
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_str);
    match arch {
        // 2026-10-24: The checkpoint's top-level `bidirectional_attention` reads false, but its
        // remote code builds `LlamaBidirectionalModel`, which sets `is_causal = False` on every
        // layer and masks only padding. The architecture is what runs, so it decides.
        Some("LlamaBidirectionalModel") => Ok(AttentionMask::Bidirectional),
        other => bail!("llm_config.architectures {other:?} is not a mapped embedding backbone"),
    }
}

/// 2026-10-24: `pooling = "avg"`, from the top level and the text config; they must agree.
fn pooling(raw: &Value, llm: &Value) -> Result<Pooling> {
    let top = raw.get("pooling").and_then(Value::as_str);
    let inner = llm.get("pooling").and_then(Value::as_str);
    match (top, inner) {
        (Some("avg"), Some("avg") | None) => Ok(Pooling::Mean),
        other => bail!("pooling {other:?} is not mapped (only `avg`)"),
    }
}

/// 2026-10-24: The text tower of `llama_nemotron_vl` (`nvidia/llama-nemotron-embed-vl-1b-v2`):
/// a bidirectional Llama-3.2 under `llm_config`, mean-pooled. The SigLIP vision tower is not
/// served (image inputs are refused), so `vision` stays `None`.
pub(crate) fn parse_llama_nemotron_vl(raw: &Value) -> Result<crate::ModelConfig> {
    let llm = raw
        .get("llm_config")
        .context("llama_nemotron_vl config missing `llm_config`")?;
    let inner = llm.get("model_type").and_then(Value::as_str);
    ensure!(
        inner == Some("llama_bidirec"),
        "llama_nemotron_vl llm_config.model_type {inner:?} is not llama_bidirec"
    );
    for (key, want) in [("attention_bias", false), ("mlp_bias", false)] {
        let got = llm.get(key).and_then(Value::as_bool);
        ensure!(
            got == Some(want),
            "llama_nemotron_vl llm_config.{key} = {got:?}; only {want} is mapped"
        );
    }
    let act = llm.get("hidden_act").and_then(Value::as_str);
    ensure!(
        act == Some("silu"),
        "llama_nemotron_vl hidden_act {act:?} is not silu"
    );
    let attention = bidirectional(llm)?;
    let pooling = pooling(raw, llm)?;
    let max = |k: &str| -> Result<usize> {
        raw.get(k)
            .and_then(Value::as_u64)
            .filter(|&v| v > 0)
            .and_then(|v| usize::try_from(v).ok())
            .with_context(|| format!("llama_nemotron_vl config missing positive `{k}`"))
    };
    let spec = EmbeddingSpec {
        pooling,
        attention,
        query_max_tokens: max("q_max_length")?,
        document_max_tokens: max("p_max_length")?,
        prompts: None,
    };
    let rope = Llama3Rope::from_rope_scaling(
        llm.get("rope_scaling")
            .context("llama_nemotron_vl llm_config has no rope_scaling")?,
    )?;

    let mut config: crate::ModelConfig =
        serde_json::from_value(llm.clone()).context("Failed to parse llm_config")?;
    ensure!(
        config.tie_word_embeddings,
        "llama_nemotron_vl must tie its embeddings"
    );
    ensure!(
        config.head_dim > 0 && config.num_attention_heads * config.head_dim == config.hidden_size,
        "llama_nemotron_vl head_dim {} x heads {} != hidden {}",
        config.head_dim,
        config.num_attention_heads,
        config.hidden_size
    );
    ensure!(
        config.num_key_value_heads > 0
            && config
                .num_attention_heads
                .is_multiple_of(config.num_key_value_heads),
        "llama_nemotron_vl kv heads {} do not divide q heads {}",
        config.num_key_value_heads,
        config.num_attention_heads
    );
    config.model_type = "llama_nemotron_vl".to_string();
    config.nested_config = true;
    config.weight_prefix = "language_model".to_string();
    config.attn_gated = false;
    config.layer_types = vec![crate::LayerType::FullAttention; config.num_hidden_layers];
    config.full_attention_interval = 1;
    config.vision = None;
    config.rope_llama3 = Some(rope);
    config.embedding = Some(spec);
    crate::finalize_config(&mut config, raw)?;
    Ok(config)
}

#[cfg(test)]
#[path = "embedding_tests.rs"]
mod tests;
