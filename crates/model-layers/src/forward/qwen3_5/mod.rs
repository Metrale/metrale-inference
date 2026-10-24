// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Single-token decode forward for one Qwen3.5 decoder layer, full
//! attention or GDN, through `&dyn GpuBackend` and a `QuantWeights` impl.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.
//!
//! [`forward_full_attention`] and [`forward_linear_attention`] each run norm,
//! projections, attention or the GDN recurrence, the residual add with the
//! post-attention norm, and the MLP with its residual. One token per call: the
//! KV append writes position `cache_pos`, and attention is passed `seq_len_attn`
//! as the cache length. Tokenizer, sampler and weight loading belong to the caller.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::quant_weights::QuantWeights;

mod full_attention;
mod kv;
mod linear_attention;

pub use full_attention::forward_full_attention;
pub use kv::{LayerKvCache, MetalKvKernels, layer_kv_bytes};
pub use linear_attention::forward_linear_attention;

/// 2026-09-25: Dimensions of a Qwen3.5 checkpoint, read by both layer forwards.
#[derive(Debug, Clone, Copy)]
pub struct Qwen35ForwardConfig {
    pub hidden: u32,
    pub intermediate: u32,
    pub num_layers: u32,
    pub vocab: u32,
    pub group_size: u32,
    pub rms_eps: f32,

    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub rope_theta: f32,
    /// 2026-09-25: RoPE rotates only the first `rotary_dim` elements of each head;
    /// the rest pass through unchanged (`rope_apply.metal`).
    pub rotary_dim: u32,

    pub num_k_heads_lin: u32,
    pub num_v_heads_lin: u32,
    pub k_head_dim_lin: u32,
    pub v_head_dim_lin: u32,
    pub conv_kernel_size: u32,
}

impl Qwen35ForwardConfig {
    /// 2026-09-25: The dimensions the `metal_qwen35_inference` example runs with; its
    /// default model directory is `Qwen3.5-4B-MLX-8bit`.
    pub const fn qwen3_5_4b_mlx_int8() -> Self {
        Self {
            hidden: 2560,
            intermediate: 9216,
            num_layers: 32,
            vocab: 248_320,
            group_size: 64,
            rms_eps: 1e-6,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 256,
            rope_theta: 10_000_000.0,
            rotary_dim: 64,
            num_k_heads_lin: 16,
            num_v_heads_lin: 32,
            k_head_dim_lin: 128,
            v_head_dim_lin: 128,
            conv_kernel_size: 4,
        }
    }

    /// 2026-09-25: q_proj output width. Qwen3.5 packs the attention output gate
    /// into the Q projection: each head's row is `[Q_h | gate_h]`, which
    /// `qwen35_qkv_split` separates before the norm.
    #[inline]
    pub const fn q_total(&self) -> u32 {
        self.num_heads * self.head_dim * 2
    }
    /// 2026-09-25: Q width after the gate is split off: half of [`Self::q_total`].
    #[inline]
    pub const fn q_only(&self) -> u32 {
        self.num_heads * self.head_dim
    }
    #[inline]
    pub const fn kv_dim(&self) -> u32 {
        self.num_kv_heads * self.head_dim
    }
    #[inline]
    pub const fn z_dim_lin(&self) -> u32 {
        self.num_v_heads_lin * self.v_head_dim_lin
    }
    #[inline]
    pub const fn qkv_total_lin(&self) -> u32 {
        2 * self.num_k_heads_lin * self.k_head_dim_lin + self.num_v_heads_lin * self.v_head_dim_lin
    }
    /// 2026-09-25: The number of GDN heads that gate, beta, dt_bias and A_log are
    /// indexed by.
    #[inline]
    pub const fn num_state_heads(&self) -> u32 {
        self.num_v_heads_lin
    }
}

/// 2026-09-25: Kernel handles for both layer forwards, looked up once by
/// [`Self::resolve`].
pub struct Qwen35Kernels {
    pub rms: KernelHandle,
    pub rope: KernelHandle,
    pub sg: KernelHandle,
    pub add_rms: KernelHandle,
    pub qkv_split: KernelHandle,
    pub conv1d: KernelHandle,
    pub gdn_gate: KernelHandle,
    pub sigmoid: KernelHandle,
    pub gdn_dec: KernelHandle,
    /// 2026-10-18: The KV append, decode attention and WHT kernels of every codec pair the
    /// Metal table serves. `resolve` looks them all up, so a missing one fails at startup
    /// whichever pair the run uses.
    pub kv: MetalKvKernels,
}

impl Qwen35Kernels {
    /// 2026-09-25: Looks up every kernel both layer forwards use; returns an error
    /// if any is missing.
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            rms: gpu.kernel("rms_norm", "rms_norm")?,
            rope: gpu.kernel("rope_apply", "rope_apply")?,
            sg: gpu.kernel("sigmoid_gate", "sigmoid_gate")?,
            add_rms: gpu.kernel("add_rms_norm", "add_rms_norm")?,
            qkv_split: gpu.kernel("qwen35_qkv_split", "qwen35_qkv_split")?,
            conv1d: gpu.kernel("causal_conv1d_update_l2norm", "causal_conv1d_update_l2norm")?,
            gdn_gate: gpu.kernel("gdn_helpers", "gdn_compute_gate")?,
            sigmoid: gpu.kernel("gdn_helpers", "sigmoid_bf16_to_f32")?,
            gdn_dec: gpu.kernel("gated_delta_rule_decode", "gated_delta_rule_decode")?,
            kv: MetalKvKernels::resolve(gpu)?,
        })
    }
}

pub struct FullAttentionLayer<'a, Q: QuantWeights> {
    pub input_ln: DevicePtr,
    pub q_norm: DevicePtr,
    pub k_norm: DevicePtr,
    pub post_ln: DevicePtr,
    pub q_proj: &'a Q,
    pub k_proj: &'a Q,
    pub v_proj: &'a Q,
    pub o_proj: &'a Q,
    pub gate_proj: &'a Q,
    pub up_proj: &'a Q,
    pub down_proj: &'a Q,
}

/// 2026-09-25: Scratch buffers for [`forward_full_attention`].
pub struct FullAttentionScratch {
    pub x_norm: DevicePtr,
    pub q_full: DevicePtr,
    pub q_split: DevicePtr,
    pub gate_split: DevicePtr,
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub q_norm_out: DevicePtr,
    pub k_norm_out: DevicePtr,
    pub attn_out: DevicePtr,
    pub gated_attn: DevicePtr,
    pub o: DevicePtr,
    pub x_resid: DevicePtr,
    pub x_norm2: DevicePtr,
    pub gate_act: DevicePtr,
    pub up_act: DevicePtr,
    pub x_out: DevicePtr,
}

pub struct LinearAttentionLayer<'a, Q: QuantWeights> {
    pub input_ln: DevicePtr,
    /// 2026-09-25: FP32 `[num_state_heads]`.
    pub a_log: DevicePtr,
    /// 2026-09-25: BF16 `[num_state_heads]`.
    pub dt_bias: DevicePtr,
    /// 2026-09-25: BF16 `[qkv_total_lin, conv_kernel_size]`.
    pub conv1d_weight: DevicePtr,
    pub in_proj_a: &'a Q,
    pub in_proj_b: &'a Q,
    pub in_proj_qkv: &'a Q,
    pub in_proj_z: &'a Q,
    /// 2026-09-25: BF16 `[v_head_dim_lin]`.
    pub norm_weight: DevicePtr,
    pub out_proj: &'a Q,
    pub post_ln: DevicePtr,
    pub gate_proj: &'a Q,
    pub up_proj: &'a Q,
    pub down_proj: &'a Q,
}

/// 2026-09-25: Conv and GDN state of one linear-attention layer, carried from
/// token to token. The caller allocates and zeroes it.
pub struct LinearAttentionState {
    /// 2026-09-25: FP32 `[qkv_total_lin, conv_kernel_size]`.
    pub conv1d_state: DevicePtr,
    /// 2026-09-25: FP32 `[num_v_heads_lin, k_head_dim_lin, v_head_dim_lin]`.
    pub gdn_state: DevicePtr,
}

/// 2026-09-25: Scratch buffers for [`forward_linear_attention`].
pub struct LinearAttentionScratch {
    pub x_norm: DevicePtr,
    pub dt_raw: DevicePtr,
    pub b_raw: DevicePtr,
    pub qkv: DevicePtr,
    pub qkv_smooth: DevicePtr,
    pub z: DevicePtr,
    /// 2026-09-25: FP32 `[num_state_heads]`.
    pub gate: DevicePtr,
    /// 2026-09-25: FP32 `[num_state_heads]`.
    pub beta: DevicePtr,
    pub y: DevicePtr,
    pub y_norm: DevicePtr,
    pub out: DevicePtr,
    pub x_resid: DevicePtr,
    pub x_norm2: DevicePtr,
    pub gate_act: DevicePtr,
    pub up_act: DevicePtr,
    pub x_final: DevicePtr,
}
