// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-21: The kernel handles of `pdl_chain.rs` and the lever variant it runs (split from it).
//!
//! Owner: gpu-examples.
//! Invariants: every handle resolves with `kernel` (a missing entry is an error, not a skip).

use anyhow::Result;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

/// 2026-10-21: Which side of each bit-identical launch lever the chain runs: `PDL_CHAIN_OLD`
/// (comma list, `all`) names the levers kept on their old kernels; the rest run the new ones.
/// `serial_sweep`: the routed gate/up sweep launched stream-ordered (`_sl`); `conv_to`: the conv
/// snapshot as one `causal_conv1d_window_advance_to` instead of a copy node and the advance;
/// `zero_swiglu`: `expert_out` zeroed by `glm5next_swiglu_clamp_zero` instead of a memset node.
#[derive(Clone, Copy, Debug)]
pub struct Variant {
    pub serial_sweep: bool,
    pub conv_to: bool,
    pub zero_swiglu: bool,
}

impl Variant {
    pub const OLD: Self = Self {
        serial_sweep: false,
        conv_to: false,
        zero_swiglu: false,
    };
    pub fn from_env() -> anyhow::Result<Self> {
        let old = metrale_gpu_examples::env_var("PDL_CHAIN_OLD")?.unwrap_or_default();
        let on = |name: &str| !old.split(',').any(|o| o == name || o == "all");
        Ok(Self {
            serial_sweep: on("serial_sweep"),
            conv_to: on("conv_to"),
            zero_swiglu: on("zero_swiglu"),
        })
    }
}

/// 2026-10-21: The chain's kernel handles on one backend.
pub struct Kern {
    pub pre: KernelHandle,
    pub post: KernelHandle,
    pub pack: KernelHandle,
    pub gates: KernelHandle,
    pub adv: KernelHandle,
    pub upd: KernelHandle,
    pub rec: KernelHandle,
    pub onorm: KernelHandle,
    pub few32: KernelHandle,
    pub topk: KernelHandle,
    pub quant: KernelHandle,
    pub quant64: KernelHandle,
    pub union: KernelHandle,
    pub sweep: KernelHandle,
    pub sweep64: KernelHandle,
    pub swiglu: KernelHandle,
    pub combine: KernelHandle,
    pub rank_sum: KernelHandle,
    pub adv_to: KernelHandle,
    pub swiglu_zero: KernelHandle,
    pub v: Variant,
    pub sms: u32,
}

impl Kern {
    pub fn resolve(g: &dyn GpuBackend, v: Variant) -> Result<Self> {
        let sweep = if v.serial_sweep {
            "w4a4_gemv_mx8_moe_union_sweep_sl"
        } else {
            "w4a4_gemv_mx8_moe_union_sweep"
        };
        Ok(Self {
            pre: g.kernel("glm5next_mhc_c1", "glm5next_hc_pre_norm_c1")?,
            post: g.kernel("glm5next_mhc", "glm5next_hc_post")?,
            pack: g.kernel("kda_layer_ops", "kda_pack_qkv_bf16")?,
            gates: g.kernel("kda_gates_w8a8", "kda_gates_w8a8_k128_rows")?,
            adv: g.kernel("kda_conv_tokens", "causal_conv1d_window_advance")?,
            upd: g.kernel("kda_conv_tokens", "causal_conv1d_update_l2norm_tokens")?,
            rec: g.kernel("kda_recurrent", "kda_recurrent_decode_bf16_seq_reg_snap")?,
            onorm: g.kernel("kda_layer_ops", "kda_o_norm_gated_bf16")?,
            few32: g.kernel(
                "dense_gemv_bf16_batchm_few",
                "dense_gemv_bf16_batchm_few_fp32out",
            )?,
            topk: g.kernel("glm5next_ffn", "glm5next_router_topk_rank")?,
            quant: g.kernel("w4a4_gemv_mx_moe", "w4a4_quant_rows_static")?,
            quant64: g.kernel("w4a4_gemv_mx_moe", "w4a4_quant_rows_static_k64")?,
            union: g.kernel("w4a16_gemv", "glm5next_moe_row_union")?,
            sweep: g.kernel("w4a4_gemv_mx_moe", sweep)?,
            sweep64: g.kernel("w4a4_gemv_mx_moe", "w4a4_gemv_mx8_moe_union_sweep_k64")?,
            swiglu: g.kernel("glm5next_ffn", "glm5next_swiglu_clamp")?,
            combine: g.kernel("glm5next_ffn", "glm5next_moe_combine")?,
            rank_sum: g.kernel("bf16_add", "bf16_add_rank_sum")?,
            adv_to: g.kernel("kda_conv_tokens", "causal_conv1d_window_advance_to")?,
            swiglu_zero: g.kernel("glm5next_ffn", "glm5next_swiglu_clamp_zero")?,
            v,
            sms: g.sm_count()?,
        })
    }
}
