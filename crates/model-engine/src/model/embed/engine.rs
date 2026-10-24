// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: `EmbedGpuModel`: its shape, scratch, construction and the packed
//! `embed_batch` loop (see the module docs of `model::embed`).
//!
//! Owner: model-engine (embedding model).
//! Invariants:
//! - `embed_batch` returns vectors in input order whatever the packing.

use std::sync::Mutex;

use anyhow::{Context, Result, ensure};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::WeightStore;

use super::kernels::EmbedKernels;
use super::pack::{plan_packs, validate_inputs};
use super::weights::EmbedWeights;

/// 2026-10-24: The text tower's shape, from the config, fixed at construction.
#[derive(Debug, Clone, Copy)]
pub(in crate::model::embed) struct EmbedShape {
    pub hidden: usize,
    pub layers: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub eps: f32,
    /// 2026-10-24: Rows of one packed forward; the scratch is sized for it.
    pub max_pack_tokens: usize,
    /// 2026-10-24: Longest single input the model accepts (its longest declared role limit).
    pub max_input_tokens: usize,
}

/// 2026-10-24: Device buffers of one packed forward, sized for `max_pack_tokens` rows.
pub(in crate::model::embed) struct Scratch {
    pub x: DevicePtr,
    pub xn: DevicePtr,
    pub q: DevicePtr,
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub attn: DevicePtr,
    pub tmp: DevicePtr,
    pub gate: DevicePtr,
    pub up: DevicePtr,
    pub act: DevicePtr,
    pub ids: DevicePtr,
    pub pos: DevicePtr,
    pub cu: DevicePtr,
    pub pooled: DevicePtr,
}

pub struct EmbedGpuModel {
    pub(super) gpu: Box<dyn GpuBackend>,
    pub(super) k: EmbedKernels,
    pub(super) w: EmbedWeights,
    pub(super) s: EmbedShape,
    /// 2026-10-24: The llama3 inverse-frequency table, FP32 `[head_dim / 2]`.
    pub(super) inv_freq: DevicePtr,
    pub(super) rope_theta: f32,
    pub(super) scratch: Mutex<Scratch>,
    /// 2026-10-24: Owns the device weights `w` points into.
    pub(super) _store: WeightStore,
}

impl EmbedGpuModel {
    /// 2026-10-24: Build from the weight store. `max_pack_tokens` (serve's
    /// `--max-batch-tokens`) bounds one packed forward and must hold the longest input.
    pub fn new(
        config: &ModelConfig,
        store: WeightStore,
        gpu: Box<dyn GpuBackend>,
        max_pack_tokens: usize,
    ) -> Result<Self> {
        let spec = config
            .embedding
            .as_ref()
            .context("EmbedGpuModel needs a config with an embedding spec")?;
        let rope = config
            .rope_llama3
            .context("EmbedGpuModel: the config declares no llama3 rope")?;
        let max_input_tokens = spec.query_max_tokens.max(spec.document_max_tokens);
        ensure!(
            max_pack_tokens >= max_input_tokens,
            "--max-batch-tokens {max_pack_tokens} is below the model's longest input \
             ({max_input_tokens} tokens); one input must fit one forward"
        );
        let s = EmbedShape {
            hidden: config.hidden_size,
            layers: config.num_hidden_layers,
            q_heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            ffn: config.intermediate_size,
            eps: config.rms_norm_eps as f32,
            max_pack_tokens,
            max_input_tokens,
        };
        ensure!(
            s.head_dim == 64,
            "EmbedGpuModel runs attn_prefill_h64; head_dim {} has no instantiation",
            s.head_dim
        );
        gpu.bind_to_thread()?;
        let k = EmbedKernels::new(gpu.as_ref())?;
        let w = EmbedWeights::bind(&store, &config.weight_prefix, &s)?;
        let table = rope.inv_freq(config.rope_theta, s.head_dim);
        let inv_freq = gpu.alloc(table.len() * 4)?;
        gpu.copy_h2d(&f32_bytes(&table), inv_freq)?;
        let scratch = Scratch::new(gpu.as_ref(), &s)?;
        tracing::info!(
            "Embedding model ready: hidden={} layers={} heads={}/{} head_dim={} ffn={} \
             max_pack_tokens={} max_input_tokens={} vocab_rows={}",
            s.hidden,
            s.layers,
            s.q_heads,
            s.kv_heads,
            s.head_dim,
            s.ffn,
            s.max_pack_tokens,
            s.max_input_tokens,
            w.vocab_rows
        );
        Ok(Self {
            gpu,
            k,
            w,
            s,
            inv_freq,
            rope_theta: config.rope_theta as f32,
            scratch: Mutex::new(scratch),
            _store: store,
        })
    }

    /// 2026-10-24: Validate, pack and run `inputs`; FP32 `[inputs.len(), hidden]`.
    pub(super) fn embed_all(&self, inputs: &[Vec<u32>]) -> Result<Vec<f32>> {
        validate_inputs(inputs, self.s.max_input_tokens, self.w.vocab_rows)?;
        let lens: Vec<usize> = inputs.iter().map(Vec::len).collect();
        let mut out = vec![0f32; inputs.len() * self.s.hidden];
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| anyhow::anyhow!("embedding scratch lock poisoned"))?;
        for r in plan_packs(&lens, self.s.max_pack_tokens)? {
            let rows = &mut out[r.start * self.s.hidden..r.end * self.s.hidden];
            self.forward_pack(&mut scratch, &inputs[r], rows)?;
        }
        Ok(out)
    }
}

impl Scratch {
    fn new(gpu: &dyn GpuBackend, s: &EmbedShape) -> Result<Self> {
        let t = s.max_pack_tokens;
        let bf = |cols: usize| gpu.alloc(t * cols * 2);
        Ok(Self {
            x: bf(s.hidden)?,
            xn: bf(s.hidden)?,
            q: bf(s.q_heads * s.head_dim)?,
            k: bf(s.kv_heads * s.head_dim)?,
            v: bf(s.kv_heads * s.head_dim)?,
            attn: bf(s.q_heads * s.head_dim)?,
            tmp: bf(s.hidden)?,
            gate: bf(s.ffn)?,
            up: bf(s.ffn)?,
            act: bf(s.ffn)?,
            ids: gpu.alloc(t * 4)?,
            pos: gpu.alloc(t * 4)?,
            cu: gpu.alloc((t + 1) * 4)?,
            pooled: gpu.alloc(t * s.hidden * 4)?,
        })
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub(super) fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
