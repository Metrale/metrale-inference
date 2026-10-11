// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Kernel handles `TransformerModel::new` resolves for the model-level ops: the
//! lm_head, argmax, embed and token-feed kernels, and the SSM-norm, softcap and
//! embed-scale kernels with their buffers.
//!
//! Owner: model-engine.
//! Invariants:
//! - An optional kernel the target lacks resolves to `KernelHandle(0)`; a required one is an error.

use anyhow::Result;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::super::ssm_pool::SsmStatePool;

/// 2026-09-26: The lm_head, argmax, embed and token-feed kernels.
pub(super) struct ModelKernels {
    pub(super) dense_gemv_kernel: KernelHandle,
    pub(super) dense_gemv_fp32out_kernel: KernelHandle,
    pub(super) w4a16_gemv_kernel: KernelHandle,
    pub(super) w4a16_gemv_logits_kernel: KernelHandle,
    pub(super) w4a16_gemm_t_kernel: KernelHandle,
    pub(super) w4a16_gemm_t_bf16_kernel: KernelHandle,
    pub(super) w4a16_gemm_kernel: KernelHandle,
    pub(super) w4a16_gemv_batch2_kernel: KernelHandle,
    pub(super) w4a16_batchm: metrale_model_layers::layers::w4a16_gemv_tiers::W4a16BatchmTiers,
    pub(super) w4a16_gemv_batch16_kernel: KernelHandle,
    pub(super) dense_gemv_fp8w_kernel: KernelHandle,
    pub(super) dense_gemv_fp8w_batch2_kernel: KernelHandle,
    pub(super) dense_gemm_kernel: KernelHandle,
    pub(super) dense_gemv_batchm_kernel: KernelHandle,
    pub(super) lm_head_m16_tc_kernel: KernelHandle,
    pub(super) lm_head_m16_tc_n64_kernel: KernelHandle,
    pub(super) argmax_kernel: KernelHandle,
    pub(super) argmax_batch_kernel: KernelHandle,
    pub(super) argmax_logits_kernel: KernelHandle,
    pub(super) batched_embed_kernel: KernelHandle,
    pub(super) fill_slots_kernel: KernelHandle,
    pub(super) argmax_feed_kernel: KernelHandle,
    pub(super) feed_resolve_kernel: KernelHandle,
}

/// 2026-09-26: Resolve [`ModelKernels`], in the order the fields are listed.
pub(super) fn resolve_model_kernels(
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<ModelKernels> {
    let dense_gemv_kernel = gpu.kernel("gemv", "dense_gemv_bf16")?;
    // 2026-09-25: Always 0. Its only launch site is the FP32-logits branch
    // of the lm_head, which `use_fp32_logits = false` (below) never takes.
    let dense_gemv_fp32out_kernel = KernelHandle(0);
    let w4a16_gemv_kernel = gpu.kernel("w4a16_gemv", "w4a16_gemv")?;
    let w4a16_gemv_logits_kernel = gpu.kernel("w4a16_gemv", "w4a16_gemv_logits")?;
    // 2026-09-25: The same resolver as the SSM and attention tile-GEMM
    // sites, which prefers the 3-deep pipeline variant when it is loaded.
    // 2026-10-07: The explicit GPT policy plus BF16 head admission makes this NVFP4-only path unreachable.
    let w4a16_gemm_t_kernel = if needs_nvfp4_head_probe(config) {
        metrale_model_layers::layers::tgemm_kernel(gpu)
    } else {
        KernelHandle(0)
    };
    // 2026-09-25: Lossless BF16-MMA lm_head variant, loaded when
    // `METRALE_LMHEAD_LOSSLESS` is set to any value. Otherwise 0, and the
    // lm_head uses `w4a16_gemm_t_kernel`.
    let w4a16_gemm_t_bf16_kernel = if std::env::var("METRALE_LMHEAD_LOSSLESS").is_ok() {
        metrale_model_layers::layers::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m128_bf16_v2")
    } else {
        metrale_gpu_runtime::gpu::KernelHandle(0)
    };
    // 2026-10-07: Both launch sites of this NVFP4 GEMM (impl_a3_lm_head.rs,
    // lm_head_batched.rs) sit under `lm_head_nvfp4 = Some(..)`, which the explicit GPT
    // BF16-head policy never builds, so it is not looked up there (the strix-hip GPT target
    // ships no w4a16 entry points). Every other family still requires it.
    let w4a16_gemm_kernel = if gpt_bf16_head_policy(config) {
        KernelHandle(0)
    } else {
        gpu.kernel("w4a16", "w4a16_gemm")?
    };
    let w4a16_gemv_batch2_kernel = gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?;
    // 2026-09-25: The narrow batched-GEMV tiers (`W4A16_BATCHM_WIDTHS`) for
    // multi-row lm_head calls. A tier the target lacks has handle 0, and
    // dispatch takes the narrowest loaded tier that covers the rows, else
    // falls back to the GEMM.
    let w4a16_batchm =
        metrale_model_layers::layers::w4a16_gemv_tiers::W4a16BatchmTiers::resolve(gpu);
    // 2026-09-25: Batched GEMV for lm_head calls of up to 16 rows; handle 0
    // when the target lacks it, and dispatch then falls back.
    let w4a16_gemv_batch16_kernel =
        metrale_model_layers::layers::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch16");
    // 2026-09-25: GEMV for the FP8 lm_head, launched only when
    // `lm_head_fp8` is set.
    let dense_gemv_fp8w_kernel = gpu.kernel("gemv_fp8w", "dense_gemv_fp8w")?;
    // 2026-09-25: FP8 GEMV over two rows; handle 0 when absent, and the FP8
    // head then runs one GEMV per row.
    let dense_gemv_fp8w_batch2_kernel = metrale_model_layers::layers::try_kernel(
        gpu,
        "dense_gemv_fp8w_batch2",
        "dense_gemv_fp8w_batch2",
    );
    let dense_gemm_kernel = gpu.kernel("gemm", "dense_gemm_bf16")?;
    let dense_gemv_batchm_kernel = gpu
        .kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")
        .unwrap_or(metrale_gpu_runtime::gpu::KernelHandle(0));
    // 2026-09-25: Tensor-core BF16 lm_head arm. `try_target_kernel` returns
    // 0 when the target's kernel tree has no `dense_gemm_m16_bf16` module
    // (only the hopper tree has one), and a 0 handle makes
    // `lm_head_m16_tc_route` decline. Where it is loaded,
    // `METRALE_LM_HEAD_M16_TC` decides whether it is launched.
    let lm_head_m16_tc_kernel = metrale_model_layers::layers::try_target_kernel(
        gpu,
        "dense_gemm_m16_bf16",
        "dense_gemm_m16_bf16",
    );
    let lm_head_m16_tc_n64_kernel = metrale_model_layers::layers::try_target_kernel(
        gpu,
        "dense_gemm_m16_bf16",
        "dense_gemm_m16_bf16_n64",
    );
    let argmax_kernel = gpu.kernel("argmax", "argmax_bf16")?;
    let argmax_batch_kernel = gpu
        .kernel("argmax", "argmax_bf16_batch")
        .unwrap_or(metrale_gpu_runtime::gpu::KernelHandle(0));
    let argmax_logits_kernel = gpu.kernel("argmax", "argmax_fp32")?;
    let batched_embed_kernel = gpu.kernel("embed_from_argmax", "batched_embed")?;
    let fill_slots_kernel = gpu.kernel("metadata_fill", "fill_slots_from_block_table")?;
    // 2026-09-25: The device token feed is optional per target: without the
    // `argmax_feed` module both handles are 0, `supports_device_token_feed`
    // is false, and the scheduler does not build the asynchronous router.
    let argmax_feed_kernel = metrale_model_layers::layers::try_target_kernel(
        gpu,
        "argmax_feed",
        "argmax_bf16_batch_feed",
    );
    let feed_resolve_kernel =
        metrale_model_layers::layers::try_target_kernel(gpu, "argmax_feed", "feed_resolve");
    Ok(ModelKernels {
        dense_gemv_kernel,
        dense_gemv_fp32out_kernel,
        w4a16_gemv_kernel,
        w4a16_gemv_logits_kernel,
        w4a16_gemm_t_kernel,
        w4a16_gemm_t_bf16_kernel,
        w4a16_gemm_kernel,
        w4a16_gemv_batch2_kernel,
        w4a16_batchm,
        w4a16_gemv_batch16_kernel,
        dense_gemv_fp8w_kernel,
        dense_gemv_fp8w_batch2_kernel,
        dense_gemm_kernel,
        dense_gemv_batchm_kernel,
        lm_head_m16_tc_kernel,
        lm_head_m16_tc_n64_kernel,
        argmax_kernel,
        argmax_batch_kernel,
        argmax_logits_kernel,
        batched_embed_kernel,
        fill_slots_kernel,
        argmax_feed_kernel,
        feed_resolve_kernel,
    })
}

/// 2026-09-26: The SSM-norm, h-dtype, softcap and embed-scale kernels, the FP32
/// logits scratch and the SSM-norm pointer table.
pub(super) struct AuxKernels {
    pub(super) ssm_norm_k: KernelHandle,
    pub(super) ssm_norm_f16_k: KernelHandle,
    pub(super) ssm_h_f32_to_f16_k: KernelHandle,
    pub(super) ssm_h_f16_to_f32_k: KernelHandle,
    pub(super) logit_softcap_kernel: KernelHandle,
    pub(super) logit_softcap_fp32_kernel: KernelHandle,
    pub(super) use_fp32_logits: bool,
    pub(super) logits_fp32_buf: DevicePtr,
    pub(super) embed_scale_kernel: KernelHandle,
    pub(super) ssm_norm_ptrs: DevicePtr,
}

/// 2026-09-26: Resolve [`AuxKernels`] and allocate its buffers.
pub(super) fn resolve_aux_kernels(
    config: &ModelConfig,
    ssm_pool: &SsmStatePool,
    gpu: &dyn GpuBackend,
) -> Result<AuxKernels> {
    let ssm_norm_k = gpu
        .kernel("ssm_state_norm", "ssm_state_clamp_norm_fused")
        .unwrap_or(KernelHandle(0));
    let ssm_norm_f16_k = gpu
        .kernel("ssm_state_norm", "ssm_state_clamp_norm_fused_f16")
        .unwrap_or(KernelHandle(0));
    let ssm_h_f32_to_f16_k =
        metrale_model_layers::layers::try_kernel(gpu, "ssm_h_dtype", "ssm_h_state_f32_to_f16");
    let ssm_h_f16_to_f32_k =
        metrale_model_layers::layers::try_kernel(gpu, "ssm_h_dtype", "ssm_h_state_f16_to_f32");

    let logit_softcap_kernel = if config.final_logit_softcapping > 0.0 {
        gpu.kernel("logit_softcap", "logit_softcap_bf16")
                .unwrap_or_else(|e| {
                    tracing::warn!(target: "metrale_model_engine::model::impl_a1", "logit_softcap kernel not found: {e}");
                    KernelHandle(0)
                })
    } else {
        KernelHandle(0)
    };
    // 2026-09-25: Always 0: only the FP32-logits path would use it.
    let logit_softcap_fp32_kernel = KernelHandle(0);
    // 2026-09-25: Always false, so the FP32 logits scratch, the FP32
    // softcap and the FP32-output lm_head GEMV are never used.
    let use_fp32_logits = false;
    let logits_fp32_buf = if use_fp32_logits {
        let bytes = config.vocab_size * 4;
        let p = gpu.alloc(bytes)?;
        tracing::info!(target: "metrale_model_engine::model::impl_a1", "FP32 LM head + softcap active (model_type={}, vocab={}). \
             Decode logits scratch: {} bytes.",
            config.model_type,
            config.vocab_size,
            bytes,
        );
        p
    } else {
        DevicePtr::NULL
    };

    let embed_scale_kernel = if config.embed_scale > 0.0 {
        gpu.kernel("embed_scale", "bf16_scale_inplace")
                .unwrap_or_else(|e| {
                    tracing::warn!(target: "metrale_model_engine::model::impl_a1", "embed_scale kernel not found: {e}");
                    KernelHandle(0)
                })
    } else {
        KernelHandle(0)
    };
    if config.embed_scale > 0.0 {
        tracing::info!(target: "metrale_model_engine::model::impl_a1", "Embedding scale: {:.4} (sqrt({}))",
            config.embed_scale,
            config.hidden_size
        );
    }
    let ssm_norm_ptrs = if ssm_pool.num_ssm_layers > 0 {
        gpu.alloc(ssm_pool.num_ssm_layers * 8)
            .unwrap_or(DevicePtr::NULL)
    } else {
        DevicePtr::NULL
    };
    Ok(AuxKernels {
        ssm_norm_k,
        ssm_norm_f16_k,
        ssm_h_f32_to_f16_k,
        ssm_h_f16_to_f32_k,
        logit_softcap_kernel,
        logit_softcap_fp32_kernel,
        use_fp32_logits,
        logits_fp32_buf,
        embed_scale_kernel,
        ssm_norm_ptrs,
    })
}

/// 2026-09-26: Returns `(feed_rows, feed_cells, feed_ids, feed_sources, feed_masks)`.
pub(super) fn alloc_feed_buffers(
    buffers: &metrale_gpu_runtime::buffers::BufferArena,
    argmax_feed_kernel: KernelHandle,
    feed_resolve_kernel: KernelHandle,
    gpu: &dyn GpuBackend,
) -> Result<(usize, DevicePtr, DevicePtr, DevicePtr, DevicePtr)> {
    let feed_rows = buffers.decode_meta().rows().max(1);
    let (feed_cells, feed_ids, feed_sources, feed_masks) =
        if argmax_feed_kernel.0 != 0 && feed_resolve_kernel.0 != 0 {
            (
                gpu.alloc(feed_rows * 4)?,
                gpu.alloc(feed_rows * 4)?,
                gpu.alloc(feed_rows * 4)?,
                gpu.alloc(feed_rows * 8)?,
            )
        } else {
            (
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
            )
        };
    Ok((feed_rows, feed_cells, feed_ids, feed_sources, feed_masks))
}

/// 2026-09-26: Start the TQ+ InnerQ calibration driver when the environment asks
/// for one; `None` when it does not or when `start()` fails.
#[cfg(feature = "cuda")]
pub(super) fn start_innerq(
    gpu: &dyn GpuBackend,
) -> Option<metrale_model_layers::layers::qwen3_attention::InnerQDriver> {
    gpu.kernel_registry().and_then(|reg| {
                let driver =
                    metrale_model_layers::layers::qwen3_attention::InnerQDriver::from_env(reg)?;
                match driver.start() {
                    Ok(()) => Some(driver),
                    Err(e) => {
                        tracing::warn!(target: "metrale_model_engine::model::impl_a1", "InnerQ calibration disabled: start() failed: {e:#}");
                        None
                    }
                }
            })
}

// 2026-10-07: The explicit GPT-OSS policy with a BF16 head: no NVFP4 head is ever built.
fn gpt_bf16_head_policy(config: &ModelConfig) -> bool {
    config.gpt_oss.is_some() && config.skip_lm_head_quantization() && !config.lm_head_fp8
}

// 2026-10-07: Other families retain their existing auto/prepacked-NVFP4 probe behavior.
fn needs_nvfp4_head_probe(config: &ModelConfig) -> bool {
    !gpt_bf16_head_policy(config)
        && metrale_model_layers::layers::tgemm_probe_ok(&config.model_type)
}

#[cfg(test)]
mod gpt_probe_tests {
    use super::*;
    #[test]
    fn explicit_native_bf16_policy_skips_only_unreachable_probe() {
        let mut config = metrale_config::parse_config(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../circuit/tests/fixtures/checkpoints/openai--gpt-oss-20b/config.json"
        )))
        .unwrap();
        assert!(!needs_nvfp4_head_probe(&config));
        assert!(gpt_bf16_head_policy(&config));
        // 2026-10-07: Known-bad control: an FP8 head request leaves the GPT BF16 policy, so
        // the NVFP4 GEMM lookup stays required.
        config.lm_head_fp8 = true;
        assert!(!gpt_bf16_head_policy(&config));
        config.lm_head_fp8 = false;
        config.gpt_oss = None;
        assert!(!gpt_bf16_head_policy(&config));
        assert_eq!(
            needs_nvfp4_head_probe(&config),
            metrale_model_layers::layers::tgemm_probe_ok(&config.model_type)
        );
        config.model_type = "qwen3".into();
        config.lm_head_bf16_override = Some(true);
        assert_eq!(
            needs_nvfp4_head_probe(&config),
            metrale_model_layers::layers::tgemm_probe_ok(&config.model_type)
        );
    }
}
