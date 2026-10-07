// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: GPT-OSS checkpoint binding and explicit eager execution.
//! The experimental loader is not registered with the model factory. Binding and
//! launch-contract tests do not establish numerical full-forward or generation parity.

use anyhow::{Context, Result, ensure};
use metrale_config::{GptOssPolicy, LayerType, ModelConfig};
use metrale_model_layers::weight_map::PackedMxfp4Experts;
use metrale_model_weights::weights::WeightStore;
use std::collections::BTreeSet;

pub mod loader;
pub mod runtime;
mod tensors;
pub use tensors::GptOssBf16Tensor;
use tensors::{bind_bf16, checked_product};

/// 2026-10-07: The pinned checkpoint stores every nonpacked tensor in BF16.
/// FP32 is a compute policy (notably norm/scale), never an implicit storage cast.
/// The store must contain the complete checkpoint before derived tensors are added.
/// Views borrow the store and leave both weights and E8M0 scales untouched.
pub struct GptOssCheckpoint<'a> {
    pub config: &'a ModelConfig,
    pub policy: GptOssPolicy,
    pub embedding: GptOssBf16Tensor<'a>,
    pub head: GptOssBf16Tensor<'a>,
    pub final_norm: GptOssBf16Tensor<'a>,
    pub layers: Vec<GptOssLayerWeights<'a>>,
}

/// 2026-10-07: Biased BF16 linear projection in `[out, in]` checkpoint order.
pub struct GptOssLinear<'a> {
    pub weight: GptOssBf16Tensor<'a>,
    pub bias: GptOssBf16Tensor<'a>,
}

/// 2026-10-07: One layer's exact checkpoint layout. Gate/up rows stay interleaved.
/// Down bias must enter each expert before the routing-weighted reduction.
pub struct GptOssLayerWeights<'a> {
    pub kind: LayerType,
    pub input_norm: GptOssBf16Tensor<'a>,
    pub post_attention_norm: GptOssBf16Tensor<'a>,
    pub q: GptOssLinear<'a>,
    pub k: GptOssLinear<'a>,
    pub v: GptOssLinear<'a>,
    pub o: GptOssLinear<'a>,
    pub sinks: GptOssBf16Tensor<'a>,
    pub router: GptOssLinear<'a>,
    pub gate_up: PackedMxfp4Experts<'a>,
    pub gate_up_bias: GptOssBf16Tensor<'a>,
    pub down: PackedMxfp4Experts<'a>,
    pub down_bias: GptOssBf16Tensor<'a>,
}

impl<'a> GptOssCheckpoint<'a> {
    /// 2026-10-07: Validate and bind all names, dimensions, dtypes and address extents.
    /// Requires a parsed GPT-OSS policy. No GPU allocation, conversion or execution.
    pub fn bind(store: &'a WeightStore, config: &'a ModelConfig) -> Result<Self> {
        let policy = config
            .gpt_oss
            .context("GPT-OSS binding requires its explicit parsed policy")?;
        ensure!(
            config.model_type == "gpt_oss",
            "GPT-OSS binding requires model_type gpt_oss"
        );
        ensure!(
            config.weight_prefix == "model",
            "GPT-OSS binding requires model prefix"
        );
        ensure!(
            !config.tie_word_embeddings,
            "GPT-OSS checkpoint has an untied head"
        );
        ensure!(config.num_hidden_layers > 0, "GPT-OSS requires layers");
        ensure!(
            config.layer_types.len() == config.num_hidden_layers,
            "GPT-OSS requires explicit kind for every layer"
        );
        ensure!(config.num_experts > 0, "GPT-OSS requires experts");
        let q = checked_product(
            &[config.num_attention_heads, config.head_dim],
            "GPT-OSS Q width",
        )?;
        let kv = checked_product(
            &[config.num_key_value_heads, config.head_dim],
            "GPT-OSS KV width",
        )?;
        let gate_up = checked_product(&[2, config.moe_intermediate_size], "GPT-OSS gate/up width")?;
        let mut names = BTreeSet::new();
        let h = config.hidden_size;
        let embedding = bind_bf16(
            store,
            "model.embed_tokens.weight",
            &[config.vocab_size, h],
            &mut names,
        )?;
        let head = bind_bf16(store, "lm_head.weight", &[config.vocab_size, h], &mut names)?;
        let final_norm = bind_bf16(store, "model.norm.weight", &[h], &mut names)?;
        let mut layers = Vec::new();
        for (i, kind) in config.layer_types.iter().copied().enumerate() {
            ensure!(
                matches!(kind, LayerType::SlidingAttention | LayerType::FullAttention),
                "GPT-OSS layer {i}: unsupported kind {kind:?}"
            );
            let p = format!("model.layers.{i}");
            let input_norm = bind_bf16(
                store,
                &format!("{p}.input_layernorm.weight"),
                &[h],
                &mut names,
            )?;
            let post_attention_norm = bind_bf16(
                store,
                &format!("{p}.post_attention_layernorm.weight"),
                &[h],
                &mut names,
            )?;
            let q_proj = linear(store, &format!("{p}.self_attn.q_proj"), q, h, &mut names)?;
            let k_proj = linear(store, &format!("{p}.self_attn.k_proj"), kv, h, &mut names)?;
            let v_proj = linear(store, &format!("{p}.self_attn.v_proj"), kv, h, &mut names)?;
            let o_proj = linear(store, &format!("{p}.self_attn.o_proj"), h, q, &mut names)?;
            let sinks = bind_bf16(
                store,
                &format!("{p}.self_attn.sinks"),
                &[config.num_attention_heads],
                &mut names,
            )?;
            let router = linear(
                store,
                &format!("{p}.mlp.router"),
                config.num_experts,
                h,
                &mut names,
            )?;
            let experts = format!("{p}.mlp.experts");
            let gate = packed(
                store,
                &format!("{experts}.gate_up_proj"),
                config.num_experts,
                gate_up,
                h,
                &mut names,
            )?;
            let down = packed(
                store,
                &format!("{experts}.down_proj"),
                config.num_experts,
                h,
                config.moe_intermediate_size,
                &mut names,
            )?;
            let gate_up_bias = bind_bf16(
                store,
                &format!("{experts}.gate_up_proj_bias"),
                &[config.num_experts, gate_up],
                &mut names,
            )?;
            let down_bias = bind_bf16(
                store,
                &format!("{experts}.down_proj_bias"),
                &[config.num_experts, h],
                &mut names,
            )?;
            layers.push(GptOssLayerWeights {
                kind,
                input_norm,
                post_attention_norm,
                q: q_proj,
                k: k_proj,
                v: v_proj,
                o: o_proj,
                sinks,
                router,
                gate_up: gate,
                gate_up_bias,
                down,
                down_bias,
            });
        }
        for name in store.names() {
            ensure!(
                names.contains(name),
                "GPT-OSS unexpected checkpoint tensor {name}"
            );
        }
        Ok(Self {
            config,
            policy,
            embedding,
            head,
            final_norm,
            layers,
        })
    }
}

fn linear<'a>(
    store: &'a WeightStore,
    prefix: &str,
    rows: usize,
    cols: usize,
    names: &mut BTreeSet<String>,
) -> Result<GptOssLinear<'a>> {
    Ok(GptOssLinear {
        weight: bind_bf16(store, &format!("{prefix}.weight"), &[rows, cols], names)?,
        bias: bind_bf16(store, &format!("{prefix}.bias"), &[rows], names)?,
    })
}

fn packed<'a>(
    store: &'a WeightStore,
    prefix: &str,
    experts: usize,
    rows: usize,
    cols: usize,
    names: &mut BTreeSet<String>,
) -> Result<PackedMxfp4Experts<'a>> {
    let blocks = format!("{prefix}_blocks");
    let scales = format!("{prefix}_scales");
    let bound = PackedMxfp4Experts::bind(store, &blocks, &scales, experts, rows, cols)?;
    names.insert(blocks);
    names.insert(scales);
    Ok(bound)
}

#[cfg(test)]
mod tests;
