// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Single-token decode forward of one Qwen3.5 full-attention layer,
//! with a bf16 or turbo KV cache.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelArg};

use super::super::quant_weights::QuantWeights;
use super::kv::{KvStepDims, KvStepIo};
use super::{
    FullAttentionLayer, FullAttentionScratch, LayerKvCache, Qwen35ForwardConfig, Qwen35Kernels,
};

/// 2026-09-25: Runs one full-attention decoder layer for one token and returns
/// `scratch.x_out`, which holds the layer's output residual stream.
#[allow(clippy::too_many_arguments)]
pub fn forward_full_attention<Q: QuantWeights>(
    gpu: &dyn GpuBackend,
    cfg: &Qwen35ForwardConfig,
    k: &Qwen35Kernels,
    layer: &FullAttentionLayer<'_, Q>,
    scratch: &FullAttentionScratch,
    kv: &LayerKvCache,
    inv_freq_ptr: DevicePtr,
    positions_ptr: DevicePtr,
    x_in: DevicePtr,
    cache_pos: u32,
    seq_len_attn: u32,
    stream: u64,
) -> Result<DevicePtr> {
    gpu.launch_typed(
        k.rms,
        [1, 1, 1],
        [128, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&cfg.hidden.to_le_bytes()),
            KernelArg::Bytes(&cfg.rms_eps.to_le_bytes()),
            KernelArg::Buffer(x_in),
            KernelArg::Buffer(layer.input_ln),
            KernelArg::Buffer(scratch.x_norm),
        ],
    )?;
    layer
        .q_proj
        .gemv(gpu, scratch.x_norm, scratch.q_full, stream)?;
    // 2026-09-25: k_proj and v_proj read the same x_norm, so one dual-output
    // GEMV computes both.
    layer.k_proj.gemv_gate_up_with(
        layer.v_proj,
        gpu,
        scratch.x_norm,
        scratch.k,
        scratch.v,
        stream,
    )?;

    // 2026-09-25: q_proj writes `[num_heads, head_dim * 2]`, each head's row
    // `[Q_h | gate_h]`. Split Q and the gate into separate buffers.
    gpu.launch_typed(
        k.qkv_split,
        [cfg.head_dim, cfg.num_heads, 1],
        [1, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&cfg.num_heads.to_le_bytes()),
            KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
            KernelArg::Buffer(scratch.q_full),
            KernelArg::Buffer(scratch.q_split),
            KernelArg::Buffer(scratch.gate_split),
        ],
    )?;
    let gate_view = scratch.gate_split;

    // 2026-09-25: Per-head Q and K RMSNorm: each head is one row of head_dim.
    gpu.launch_typed(
        k.rms,
        [cfg.num_heads, 1, 1],
        [128, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
            KernelArg::Bytes(&cfg.rms_eps.to_le_bytes()),
            KernelArg::Buffer(scratch.q_split),
            KernelArg::Buffer(layer.q_norm),
            KernelArg::Buffer(scratch.q_norm_out),
        ],
    )?;
    gpu.launch_typed(
        k.rms,
        [cfg.num_kv_heads, 1, 1],
        [128, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
            KernelArg::Bytes(&cfg.rms_eps.to_le_bytes()),
            KernelArg::Buffer(scratch.k),
            KernelArg::Buffer(layer.k_norm),
            KernelArg::Buffer(scratch.k_norm_out),
        ],
    )?;

    // 2026-09-25: RoPE rotates q_norm_out and k_norm_out in place.
    let half_dim = cfg.rotary_dim / 2;
    let n_tokens = 1u32;
    gpu.launch_typed(
        k.rope,
        [half_dim, cfg.num_heads, 1],
        [1, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&n_tokens.to_le_bytes()),
            KernelArg::Bytes(&cfg.num_heads.to_le_bytes()),
            KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
            KernelArg::Bytes(&cfg.rotary_dim.to_le_bytes()),
            KernelArg::Buffer(positions_ptr),
            KernelArg::Buffer(inv_freq_ptr),
            KernelArg::Buffer(scratch.q_norm_out),
        ],
    )?;
    gpu.launch_typed(
        k.rope,
        [half_dim, cfg.num_kv_heads, 1],
        [1, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&n_tokens.to_le_bytes()),
            KernelArg::Bytes(&cfg.num_kv_heads.to_le_bytes()),
            KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
            KernelArg::Bytes(&cfg.rotary_dim.to_le_bytes()),
            KernelArg::Buffer(positions_ptr),
            KernelArg::Buffer(inv_freq_ptr),
            KernelArg::Buffer(scratch.k_norm_out),
        ],
    )?;

    // 2026-09-25: The KV append takes the post-RoPE k_norm_out.
    // 2026-09-25: Sparse-V threshold: the quantized attention kernels skip V rows whose
    // unnormalized weight exp(score - max) is at or below it, so 0.0 skips only zero-weight
    // rows. METRALE_SPARSE_V_THRESHOLD sets it; unset or unparsable gives 1e-3.
    let sparse_v: f32 = std::env::var("METRALE_SPARSE_V_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-3);
    super::kv::append_and_attend(
        gpu,
        k.kv.get(kv.pair)?,
        kv,
        &KvStepIo {
            q: scratch.q_norm_out,
            k: scratch.k_norm_out,
            v: scratch.v,
            out: scratch.attn_out,
        },
        &KvStepDims {
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            cache_pos,
            seq_len_attn,
            scale: 1.0 / (cfg.head_dim as f32).sqrt(),
            sparse_v,
        },
        stream,
    )?;

    let q_only = cfg.q_only();
    gpu.launch_typed(
        k.sg,
        [q_only.div_ceil(64), 1, 1],
        [64, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&q_only.to_le_bytes()),
            KernelArg::Buffer(gate_view),
            KernelArg::Buffer(scratch.attn_out),
            KernelArg::Buffer(scratch.gated_attn),
        ],
    )?;

    layer
        .o_proj
        .gemv(gpu, scratch.gated_attn, scratch.o, stream)?;

    // 2026-09-25: One kernel writes x_resid = x_in + o and x_norm2 =
    // RMSNorm(x_resid) with post_ln.
    gpu.launch_typed(
        k.add_rms,
        [1, 1, 1],
        [128, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&cfg.hidden.to_le_bytes()),
            KernelArg::Bytes(&cfg.rms_eps.to_le_bytes()),
            KernelArg::Buffer(x_in),
            KernelArg::Buffer(scratch.o),
            KernelArg::Buffer(layer.post_ln),
            KernelArg::Buffer(scratch.x_resid),
            KernelArg::Buffer(scratch.x_norm2),
        ],
    )?;
    // 2026-09-25: gate_proj and up_proj read the same x_norm2: one dual-output
    // GEMV.
    layer.gate_proj.gemv_gate_up_with(
        layer.up_proj,
        gpu,
        scratch.x_norm2,
        scratch.gate_act,
        scratch.up_act,
        stream,
    )?;
    // 2026-09-25: x_out = x_resid + down_proj @ (silu(gate_act) ⊙ up_act).
    layer.down_proj.gemv_silu_gate_resid(
        gpu,
        scratch.gate_act,
        scratch.up_act,
        scratch.x_resid,
        scratch.x_out,
        stream,
    )?;
    Ok(scratch.x_out)
}
