// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: The kernel lookups on Laguna's load path that the strix-hip INT4 target does
//! not resolve, each with the reason it is absent. Generated from the
//! `laguna_lookup_inventory` table of strix_hip_laguna_int4.rs and reviewed by hand.
//!
//! Owner: metrale-kernels tests.
//! Invariants:
//! - Nothing here is a boot-gate declaration. The lookups the first gfx1151
//!   `met serve --check-kernels` made and could not resolve are MODEL.toml `[expected_absent]`
//!   declarations (or were built, dense_gemv_bf16_batchm); what remains here is on Laguna's
//!   static load path but was not looked up by that boot (the NVFP4 MoeLayer constructor, for
//!   one, is never built for packed-int experts).
//! - The list can only change with the code: an entry that starts resolving, or whose
//!   lookup disappears, fails `every_laguna_lookup_resolves_or_is_classified`.

/// 2026-10-07: Why a lookup does not resolve on the Laguna strix-hip INT4 target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapClass {
    /// 2026-10-07: NVFP4 / E8M0 / FP4-activation kernels. Laguna INT4 has no such tensors;
    /// its experts run through packed_int_gemv. The required ones are the NVFP4 MoeLayer
    /// constructor's (moe/init.rs): the INT4 expert path must not construct that layer.
    Nvfp4Only,
    /// 2026-10-07: FP8-weight kernels (E4M3 W8A16, block-scaled, W8A8, an FP8 lm_head).
    Fp8WeightOnly,
    /// 2026-10-07: GGUF Q2_0 / Q4_K kernels.
    GgufOnly,
    /// 2026-10-07: sm_90 split-K decode attention.
    HopperOnly,
    /// 2026-10-07: A feature Laguna's config never selects: hash, softmax-bias or
    /// sqrt-softplus routing, logit softcap, embedding scale, GELU, SSM state, MRoPE, BF16
    /// routed experts, the opt-in 4-row atomic NVFP4 decode.
    OtherModelFeature,
    /// 2026-10-07: Families Laguna's BF16 attention, router, shared expert, KV write or
    /// sampling may use that no strix-hip target builds. All are probes with a resolved
    /// fallback; none has a HIP build.
    HipMissing,
}

use GapClass::*;

/// 2026-10-07: One module's unresolved entry points of one class.
pub struct Gap {
    /// 2026-10-07: The class.
    pub class: GapClass,
    /// 2026-10-07: Module name as looked up.
    pub module: &'static str,
    /// 2026-10-07: Entry points.
    pub entries: &'static [&'static str],
}

/// 2026-10-07: Every unresolved Laguna lookup on strix-hip/laguna-xs-2.1/int4.
pub const GPU_GATE_GAPS: &[Gap] = &[
    Gap {
        class: Nvfp4Only,
        module: "moe_w4a16",
        entries: &[
            "moe_w4a16_down_t_k64_fp4",
            "moe_w4a16_fused_gate_up_t",
            "moe_w4a16_fused_gate_up_t_e8m0",
            "moe_w4a16_fused_gate_up_t_k64",
            "moe_w4a16_fused_gate_up_t_k64_e8m0",
            "moe_w4a16_fused_gate_up_t_k64_fp4",
            "moe_w4a16_fused_gate_up_t_k64_m128",
            "moe_w4a16_grouped_gemm_ptrtable",
            "moe_w4a16_grouped_gemm_ptrtable_e8m0",
            "moe_w4a16_grouped_gemm_ptrtable_k32",
            "moe_w4a16_grouped_gemm_ptrtable_m256",
            "moe_w4a16_grouped_gemm_ptrtable_t",
            "moe_w4a16_grouped_gemm_ptrtable_t_e8m0",
            "moe_w4a16_grouped_gemm_ptrtable_t_k64",
            "moe_w4a16_grouped_gemm_ptrtable_t_k64_e8m0",
        ],
    },
    Gap {
        class: Fp8WeightOnly,
        module: "moe_bucket_builder",
        entries: &["bucket_builder"],
    },
    Gap {
        class: Fp8WeightOnly,
        module: "moe_fp8_grouped_gemm",
        entries: &["moe_fp8_grouped_gemm"],
    },
    Gap {
        class: Fp8WeightOnly,
        module: "moe_w4a16",
        entries: &["moe_fp8_grouped_gemm_ptrtable_t"],
    },
    Gap {
        class: Fp8WeightOnly,
        module: "moe_w8a8_grouped_gemm",
        entries: &["moe_w8a8_grouped_gemm", "moe_w8a8_grouped_gemm_pm4"],
    },
    Gap {
        class: Fp8WeightOnly,
        module: "moe_w8a8_m16",
        entries: &["pm4_m16"],
    },
    Gap {
        class: Fp8WeightOnly,
        module: "w8a16_gemm_m16",
        entries: &[
            "w8a16_gemm_m16",
            "w8a16_gemm_m16_n64",
            "w8a16_gemm_m16_strided",
        ],
    },
    Gap {
        class: Fp8WeightOnly,
        module: "w8a16_gemm_pipelined_m32",
        entries: &["w8a16_gemm_pipelined_m32", "w8a16_gemm_pipelined_m64"],
    },
    Gap {
        class: Fp8WeightOnly,
        module: "w8a16_gemv_ncol",
        entries: &[
            "w8a16_gemv_batch16_ncol2",
            "w8a16_gemv_batch16_ncol2_strided",
            "w8a16_gemv_batch16_ncol4",
            "w8a16_gemv_batch16_ncol4_strided",
        ],
    },
    Gap {
        class: GgufOnly,
        module: "q2_0_mmq",
        entries: &["metrale_q2_0_mmq128_nc", "metrale_q2_0_mmq128_wc"],
    },
    Gap {
        class: HopperOnly,
        module: "paged_decode_bf16_splitk_hopper",
        entries: &[
            "paged_decode_attn_reduce_bf16_hopper",
            "paged_decode_attn_splitk_bf16_hopper",
        ],
    },
    Gap {
        class: HopperOnly,
        module: "paged_decode_fp8_splitk_hopper",
        entries: &[
            "paged_decode_attn_reduce_fp8_hopper",
            "paged_decode_attn_splitk_fp8_hopper",
        ],
    },
    Gap {
        class: OtherModelFeature,
        module: "embed_scale",
        entries: &["bf16_scale_inplace"],
    },
    Gap {
        class: OtherModelFeature,
        module: "fused_k_norm_rope_cache",
        entries: &["fused_k_norm_rope_mrope_cache_write_bf16"],
    },
    Gap {
        class: OtherModelFeature,
        module: "gelu",
        entries: &["gelu_mul"],
    },
    Gap {
        class: OtherModelFeature,
        module: "logit_softcap",
        entries: &["logit_softcap_bf16"],
    },
    Gap {
        class: OtherModelFeature,
        module: "moe_bf16_grouped_gemm",
        entries: &["moe_bf16_grouped_gemm"],
    },
    Gap {
        class: OtherModelFeature,
        module: "moe_decode_atomic_c4",
        entries: &[
            "moe_decode_atomic_c4_finalize",
            "moe_decode_atomic_c4_silu_down_accum",
        ],
    },
    Gap {
        class: OtherModelFeature,
        module: "moe_hash_route",
        entries: &["moe_hash_route", "moe_hash_route_batched"],
    },
    Gap {
        class: OtherModelFeature,
        module: "moe_topk_softmax_bias",
        entries: &[
            "moe_topk_softmax_bias",
            "moe_topk_softmax_bias_batched",
            "moe_zero_expert_add",
        ],
    },
    Gap {
        class: OtherModelFeature,
        module: "moe_topk_sqrt",
        entries: &["moe_topk_sqrtsoftplus", "moe_topk_sqrtsoftplus_batched"],
    },
    Gap {
        class: HipMissing,
        module: "argmax_feed",
        entries: &[
            "argmax_bf16_batch_feed",
            "argmax_bf16_batch_masked_host",
            "feed_resolve",
        ],
    },
    Gap {
        class: HipMissing,
        module: "dense_gemm_m16_bf16",
        entries: &["dense_gemm_m16_bf16", "dense_gemm_m16_bf16_n64"],
    },
    Gap {
        class: HipMissing,
        module: "fused_k_norm_rope_cache",
        entries: &["fused_k_norm_rope_cache_write_bf16"],
    },
    Gap {
        class: HipMissing,
        module: "gemm",
        entries: &["dense_gemm_bf16_router"],
    },
    Gap {
        class: HipMissing,
        module: "moe_router_gemm",
        entries: &["moe_router_gemm_bf16"],
    },
    Gap {
        class: HipMissing,
        module: "moe_router_gemm_prefill",
        entries: &["moe_router_gemm_rt"],
    },
    Gap {
        class: HipMissing,
        module: "moe_shared_expert_fused_bf16",
        entries: &[
            "moe_expert_gate_up_shared_bf16",
            "moe_expert_silu_down_shared_bf16",
        ],
    },
    Gap {
        class: HipMissing,
        module: "moe_shared_expert_fused_bf16_batch2",
        entries: &[
            "moe_expert_gate_up_shared_bf16_batch2",
            "moe_expert_silu_down_shared_bf16_batch2",
        ],
    },
    Gap {
        class: HipMissing,
        module: "moe_unpermute_blend",
        entries: &["moe_unpermute_blend"],
    },
    Gap {
        class: HipMissing,
        module: "paged_decode_attn_bf16_gqa",
        entries: &["paged_decode_attn_bf16_gqa"],
    },
    Gap {
        class: HipMissing,
        module: "paged_decode_attn_fp8_gqa",
        entries: &["paged_decode_attn_fp8_gqa"],
    },
    Gap {
        class: HipMissing,
        module: "reshape_and_cache_fused_k_fp8",
        entries: &["fused_k_norm_rope_cache_write_fp8_kv"],
    },
    Gap {
        class: HipMissing,
        module: "silu_mul_strided",
        entries: &["silu_mul_strided"],
    },
];
