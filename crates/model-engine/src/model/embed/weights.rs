// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: The text tower's device weights, bound by name from the weight store. Every
//! tensor is checked for presence, BF16 dtype and its exact shape before the model exists, so
//! a mismatched checkpoint fails the boot instead of a kernel reading out of bounds.
//!
//! Owner: model-engine (embedding model).
//! Invariants:
//! - Each pointer is a `[rows, cols]` row-major BF16 tensor the store owns; projections are
//!   `[out, in]` (the GEMM reads `B^T`).

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_weights::weights::{WeightDtype, WeightStore};

use super::engine::EmbedShape;

pub(super) struct LayerWeights {
    pub input_norm: DevicePtr,
    pub q: DevicePtr,
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub o: DevicePtr,
    pub post_norm: DevicePtr,
    pub gate: DevicePtr,
    pub up: DevicePtr,
    pub down: DevicePtr,
}

pub(super) struct EmbedWeights {
    pub embed: DevicePtr,
    /// 2026-10-24: Rows of the embedding table: every token id must be below it.
    pub vocab_rows: usize,
    pub final_norm: DevicePtr,
    pub layers: Vec<LayerWeights>,
}

fn bf16(store: &WeightStore, name: &str, shape: &[usize]) -> Result<DevicePtr> {
    let t = store
        .get(name)
        .with_context(|| format!("embedding model: missing tensor `{name}`"))?;
    ensure!(
        t.dtype == WeightDtype::BF16,
        "embedding model: `{name}` is {:?}; the served precision is BF16",
        t.dtype
    );
    ensure!(
        t.shape == shape,
        "embedding model: `{name}` has shape {:?}, the config implies {shape:?}",
        t.shape
    );
    Ok(t.ptr)
}

impl EmbedWeights {
    /// 2026-10-24: Bind every text-tower tensor under `prefix` (the config's `weight_prefix`).
    /// The tied embedding is the input table; no LM head is loaded.
    pub(super) fn bind(store: &WeightStore, prefix: &str, s: &EmbedShape) -> Result<Self> {
        let (h, f) = (s.hidden, s.ffn);
        let qd = s.q_heads * s.head_dim;
        let kvd = s.kv_heads * s.head_dim;
        let layers = (0..s.layers)
            .map(|i| {
                let p = format!("{prefix}.layers.{i}");
                Ok(LayerWeights {
                    input_norm: bf16(store, &format!("{p}.input_layernorm.weight"), &[h])?,
                    q: bf16(store, &format!("{p}.self_attn.q_proj.weight"), &[qd, h])?,
                    k: bf16(store, &format!("{p}.self_attn.k_proj.weight"), &[kvd, h])?,
                    v: bf16(store, &format!("{p}.self_attn.v_proj.weight"), &[kvd, h])?,
                    o: bf16(store, &format!("{p}.self_attn.o_proj.weight"), &[h, qd])?,
                    post_norm: bf16(store, &format!("{p}.post_attention_layernorm.weight"), &[h])?,
                    gate: bf16(store, &format!("{p}.mlp.gate_proj.weight"), &[f, h])?,
                    up: bf16(store, &format!("{p}.mlp.up_proj.weight"), &[f, h])?,
                    down: bf16(store, &format!("{p}.mlp.down_proj.weight"), &[h, f])?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let embed_name = format!("{prefix}.embed_tokens.weight");
        let vocab_rows = store
            .get(&embed_name)
            .with_context(|| format!("embedding model: missing tensor `{embed_name}`"))?
            .shape
            .first()
            .copied()
            .unwrap_or(0);
        ensure!(
            vocab_rows > 0,
            "embedding model: `{embed_name}` has no rows"
        );
        Ok(Self {
            embed: bf16(store, &embed_name, &[vocab_rows, h])?,
            vocab_rows,
            final_norm: bf16(store, &format!("{prefix}.norm.weight"), &[h])?,
            layers,
        })
    }
}
