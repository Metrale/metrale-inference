// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `DecodeKernels`: the fused K-norm/RoPE cache writes, the WHT and
//! InnerQ kernels, and the paged-decode, MLA and split-K kernels of a
//! `Qwen3AttentionLayer`. `new_with_gating` (`init.rs`) calls the three
//! resolvers in the order `ProjKernels`, `DecodeKernels`, `PrefillKernels` and
//! moves every field into the layer.
//!
//! Owner: model-layers (attention).
//! Invariants:
//! - `resolve` looks the kernels up in the order the fields are written.
//! - A kernel family behind `ArchProbes` is looked up only when the config
//!   says the model has it.
//! - Every lookup of the `fused_k_norm_rope_cache` module goes through
//!   `try_target_kernel` (pinned by `init.rs`'s `fused_kv_probe_guard` test).

use anyhow::Result;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};
use metrale_kv_codec::{KernelName, ResolvedKvKernels};

// 2026-09-26: `gate` is called by path, never through a `let`-bound function
// pointer: a call through a pointer does not pass the caller's location to a
// `#[track_caller]` fn, so the boot audit would name `gated` itself instead of
// the dispatch site below.
use super::init_arch_gates::{ArchProbes, gated as gate, present};

/// 2026-09-26: The handles `DecodeKernels::resolve` looked up, one field per
/// `Qwen3AttentionLayer` field of the same name.
pub(super) struct DecodeKernels {
    pub(super) fused_k_norm_rope_cache_write_bf16_k: KernelHandle,
    pub(super) fused_k_norm_rope_mrope_cache_write_bf16_k: KernelHandle,
    pub(super) reshape_and_cache_flash_v_only_k: KernelHandle,
    pub(super) fused_k_norm_rope_cache_write_fp8_kv_k: KernelHandle,
    pub(super) wht_bf16_k: KernelHandle,
    pub(super) wht_bf16_k_inv: KernelHandle,
    pub(super) innerq_apply_q_k: KernelHandle,
    pub(super) innerq_apply_k_k: KernelHandle,
    pub(super) paged_decode_k: KernelHandle,
    pub(super) paged_decode_512_k: Option<KernelHandle>,
    pub(super) paged_decode_mla_k: KernelHandle,
    pub(super) mla_paged_decode_k: KernelHandle,
    pub(super) mla_paged_decode_fp8_k: KernelHandle,
    pub(super) mla_batched_gemv_k: KernelHandle,
    pub(super) mla_q_rope_scatter_k: KernelHandle,
    pub(super) mla_q_rope_writeback_k: KernelHandle,
    pub(super) mla_cache_assemble_k: KernelHandle,
    pub(super) mla_q_rope_extract_batched_k: KernelHandle,
    pub(super) mla_q_rope_writeback_batched_k: KernelHandle,
    pub(super) mla_kv_assemble_batched_k: KernelHandle,
    pub(super) mla_cache_assemble_batched_k: KernelHandle,
    pub(super) prefill_attn_mla320_k: KernelHandle,
    pub(super) grouped_gemm_mla_k: KernelHandle,
    pub(super) mla_q_final_assemble_k: KernelHandle,
    pub(super) mla_fused_prefill_k: KernelHandle,
    pub(super) gemm_splitk_partial_k: KernelHandle,
    pub(super) gemm_splitk_reduce_k: KernelHandle,
    pub(super) dense_gemm_tc_k: KernelHandle,
    pub(super) paged_decode_splitk_k: Option<KernelHandle>,
    pub(super) paged_decode_reduce_k: Option<KernelHandle>,
    pub(super) paged_decode_bf16_gqa_k: Option<KernelHandle>,
    pub(super) paged_decode_fp8_gqa_k: Option<KernelHandle>,
    pub(super) paged_decode_splitk_hopper_k: Option<KernelHandle>,
    pub(super) paged_decode_reduce_hopper_k: Option<KernelHandle>,
    pub(super) paged_decode_splitk_bf16_hopper_k: Option<KernelHandle>,
    pub(super) paged_decode_reduce_bf16_hopper_k: Option<KernelHandle>,
}

impl DecodeKernels {
    /// 2026-09-26: Looks up every field's kernel, in field order. The first
    /// failed required lookup returns its error and issues no later lookup. 2026-10-18: The KV
    /// codec's decode, head_dim-512 and split-K kernels are the ones `kv_names` (the codec
    /// kernel table) names; a kernel the pair has no row for is `None`, never another pair's.
    pub(super) fn resolve(
        gpu: &dyn GpuBackend,
        probes: &ArchProbes,
        kv_names: &ResolvedKvKernels<KernelName>,
    ) -> Result<Self> {
        let hard = |name: KernelName| gpu.kernel(name.module, name.func);
        Ok(Self {
            // 2026-09-25: `try_target_kernel`: the `fused_k_norm_rope_cache`
            // module is built only from `kernels/gb10/common`, which the hopper
            // and b200 trees overlay; b300, strix, strix-hip and metal lack it.
            // A lookup there would fail, and the boot audit refuses to serve on
            // a failed lookup the target does not list as expected-absent.
            // `try_target_kernel` issues no lookup when the module is absent and
            // is the ordinary lookup, audit included, when it is present.
            fused_k_norm_rope_cache_write_bf16_k: super::super::try_target_kernel(
                gpu,
                "fused_k_norm_rope_cache",
                "fused_k_norm_rope_cache_write_bf16",
            ),
            // 2026-09-25: Same module, same reason.
            fused_k_norm_rope_mrope_cache_write_bf16_k: super::super::try_target_kernel(
                gpu,
                "fused_k_norm_rope_cache",
                "fused_k_norm_rope_mrope_cache_write_bf16",
            ),
            reshape_and_cache_flash_v_only_k: super::super::try_kernel(
                gpu,
                "reshape_and_cache",
                "reshape_and_cache_flash_v_only",
            ),
            // 2026-09-25: `try_target_kernel` for the same reason:
            // `reshape_and_cache_fused_k_fp8.cu` is only in `kernels/gb10/common`
            // (overlaid by hopper and b200). Without it the handle is 0 and
            // `fused_fp8_kv_decode_eligible` keeps decode on the un-fused chain.
            fused_k_norm_rope_cache_write_fp8_kv_k: super::super::try_target_kernel(
                gpu,
                "reshape_and_cache_fused_k_fp8",
                "fused_k_norm_rope_cache_write_fp8_kv",
            ),
            wht_bf16_k: super::super::try_kernel(gpu, "wht_bf16", "wht_bf16_inplace"),
            wht_bf16_k_inv: super::super::try_kernel(gpu, "wht_bf16", "wht_bf16_inplace_inv"),
            innerq_apply_q_k: super::super::try_kernel(
                gpu,
                "tq_plus_innerq_apply",
                "tq_plus_innerq_apply_q",
            ),
            innerq_apply_k_k: super::super::try_kernel(
                gpu,
                "tq_plus_innerq_apply",
                "tq_plus_innerq_apply_k",
            ),
            paged_decode_k: hard(kv_names.decode)?,
            // 2026-09-25: The head_dim > 256 decode arm, looked up only when
            // `probes.wide_head_dim` is set. 2026-10-18: Only the pair's own build: Turbo3 and
            // Turbo2 used to load turbo4's, and NVFP4 and the asymmetric pairs fp8's, which no
            // reader of theirs launched. Readers: `Qwen3AttentionLayer::decode_kernel_for`.
            paged_decode_512_k: kv_names
                .decode_512
                .and_then(|name| present(gate(probes.wide_head_dim, gpu, name.module, name.func))),
            paged_decode_mla_k: gate(probes.mla, gpu, "paged_decode_mla", "paged_decode_attn"),
            // 2026-09-25: MLA paged decode over the compressed KV cache, whose
            // row is `kv_lora_rank + rope` wide; looked up only for MLA models.
            mla_paged_decode_k: gate(
                probes.mla,
                gpu,
                "mla_paged_decode",
                "mla_paged_decode_nvfp4",
            ),
            mla_paged_decode_fp8_k: gate(
                probes.mla,
                gpu,
                "mla_paged_decode_fp8",
                "mla_paged_decode_fp8",
            ),
            mla_batched_gemv_k: gate(probes.mla, gpu, "mla_absorbed", "mla_batched_gemv"),
            mla_q_rope_scatter_k: gate(probes.mla, gpu, "mla_absorbed", "mla_q_rope_scatter"),
            mla_q_rope_writeback_k: gate(probes.mla, gpu, "mla_absorbed", "mla_q_rope_writeback"),
            mla_cache_assemble_k: gate(probes.mla, gpu, "mla_absorbed", "mla_cache_assemble"),
            mla_q_rope_extract_batched_k: gate(
                probes.mla,
                gpu,
                "mla_absorbed",
                "mla_q_rope_extract_batched",
            ),
            mla_q_rope_writeback_batched_k: gate(
                probes.mla,
                gpu,
                "mla_absorbed",
                "mla_q_rope_writeback_batched",
            ),
            mla_kv_assemble_batched_k: gate(
                probes.mla,
                gpu,
                "mla_absorbed",
                "mla_kv_assemble_batched",
            ),
            mla_cache_assemble_batched_k: gate(
                probes.mla,
                gpu,
                "mla_absorbed",
                "mla_cache_assemble_batched",
            ),
            prefill_attn_mla320_k: gate(
                probes.mla,
                gpu,
                "mla_prefill_attn",
                "mla_prefill_attn_320",
            ),
            grouped_gemm_mla_k: gate(probes.mla, gpu, "grouped_gemm_mla", "grouped_gemm_mla"),
            mla_q_final_assemble_k: gate(
                probes.mla,
                gpu,
                "mla_absorbed",
                "mla_q_final_assemble_batched",
            ),
            mla_fused_prefill_k: gate(probes.mla, gpu, "mla_fused_prefill", "mla_fused_prefill"),
            gemm_splitk_partial_k: super::super::try_kernel(
                gpu,
                "gemm_splitk",
                "dense_gemm_splitk_partial",
            ),
            gemm_splitk_reduce_k: super::super::try_kernel(
                gpu,
                "gemm_splitk",
                "dense_gemm_splitk_reduce",
            ),
            dense_gemm_tc_k: super::super::try_kernel(gpu, "gemm_tc", "dense_gemm_tc"),
            // 2026-10-18: A hard lookup when the codec pair has a split-K pair (FP8, NVFP4).
            // BF16 and Turbo2 used to load FP8's through a `_` arm; no reader of theirs did.
            paged_decode_splitk_k: kv_names.splitk.map(|(s, _)| hard(s)).transpose()?,
            paged_decode_reduce_k: kv_names.splitk.map(|(_, r)| hard(r)).transpose()?,
            // 2026-09-25: The GQA-packed non-split twins, from
            // `kernels/gb10/common/paged_decode_attn_{bf16,fp8}_gqa.cu`. A target
            // without them gets `None` and decode keeps the unpacked kernel.
            // They are resolved whatever `METRALE_ATTN_DECODE_GQA_PACK` says, so
            // the handle set does not depend on the environment.
            paged_decode_bf16_gqa_k: present(super::super::try_target_kernel(
                gpu,
                "paged_decode_attn_bf16_gqa",
                "paged_decode_attn_bf16_gqa",
            )),
            paged_decode_fp8_gqa_k: present(super::super::try_target_kernel(
                gpu,
                "paged_decode_attn_fp8_gqa",
                "paged_decode_attn_fp8_gqa",
            )),
            // 2026-09-25: The Hopper split-K twins, from `kernels/hopper/common`
            // only. Every other target gets `None`, and `fp8_splitk_pair` falls
            // back to the gb10 pair. They are resolved whatever the
            // `attn_decode_splitk` policy is, so the handle set does not depend
            // on the environment.
            paged_decode_splitk_hopper_k: present(super::super::try_target_kernel(
                gpu,
                "paged_decode_fp8_splitk_hopper",
                "paged_decode_attn_splitk_fp8_hopper",
            )),
            paged_decode_reduce_hopper_k: present(super::super::try_target_kernel(
                gpu,
                "paged_decode_fp8_splitk_hopper",
                "paged_decode_attn_reduce_fp8_hopper",
            )),
            paged_decode_splitk_bf16_hopper_k: present(super::super::try_target_kernel(
                gpu,
                "paged_decode_bf16_splitk_hopper",
                "paged_decode_attn_splitk_bf16_hopper",
            )),
            paged_decode_reduce_bf16_hopper_k: present(super::super::try_target_kernel(
                gpu,
                "paged_decode_bf16_splitk_hopper",
                "paged_decode_attn_reduce_bf16_hopper",
            )),
        })
    }
}
