// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The per-layer MoE FFN of `qwen4_exp`: the weights from
//! `load_moe_qwen35`, the router (BF16 as shipped, or an NVFP4 copy), the shared
//! expert at BF16 where the checkpoint declares it so (2026-10-13), and,
//! with `METRALE_HOLO_MOE_GROUPED_CUTLASS=1`, the CUTLASS grouped scale
//! tables.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use anyhow::{Context, Result};
use metrale_config::{ModelConfig, WeightQuantPolicy};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;

use metrale_model_layers::layers::{FfnComponent, MoeLayer};
use metrale_model_layers::weight_map::quant_helpers::dense_auto;
use metrale_model_layers::weight_map::{
    Nvfp4MoeCopies, Nvfp4Variant, load_moe_qwen35, quantize_to_nvfp4,
};

/// 2026-10-13: The shared expert's projections, as the checkpoint names them under `.mlp`.
const SHARED_PROJS: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

/// 2026-10-13: Whether layer `lp`'s shared expert runs at its BF16 weights: the policy follows
/// the checkpoint's plan and the plan leaves all three projections unquantized (the Inferact
/// NVFP4 checkpoint lists them in its `ignore`).
pub(super) fn shared_expert_declared_bf16(policy: &WeightQuantPolicy, lp: &str) -> bool {
    policy.follows_plan()
        && SHARED_PROJS.iter().all(|p| {
            policy
                .declared(&format!("{lp}.mlp.shared_expert.{p}"))
                .weight
                .is_none()
        })
}

pub(super) fn build_moe(
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
) -> Result<FfnComponent> {
    let h = config.hidden_size;
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();

    // 2026-10-13: A shared expert the checkpoint keeps at BF16 runs at those weights (the
    // store's own tensors, no copy), and no NVFP4 copy of it is built: every fused routed
    // decode kernel zero-fills its shared term on a NULL shared weight, and the BF16 pass
    // writes it (`has_mixed_bf16_shared_expert`).
    let policy = WeightQuantPolicy::for_checkpoint(
        metrale_model_layers::layers::weight_quantization(),
        config.quantization_config.as_ref(),
        metrale_model_layers::layers::kernel_caps(),
    );
    let bf16_shared = shared_expert_declared_bf16(&policy, lp);
    let copies = Nvfp4MoeCopies {
        shared: !bf16_shared,
        ..Nvfp4MoeCopies::ALL
    };

    let weights = load_moe_qwen35(
        store,
        lp,
        config.num_experts,
        gpu,
        config,
        variant,
        absmax_k,
        quantize_k,
        stream,
        copies,
    )
    .with_context(|| format!("qwen4_exp: MoE block at {lp}"))?;

    // 2026-10-13: The router runs BF16 as the checkpoint ships it (`gate_nvfp4 = None`: the
    // dense GEMV/GEMM over `weights.gate`) unless `METRALE_QWEN4EXP_BF16_ROUTER=0`, which
    // requantizes it to NVFP4. An NVFP4 router perturbs the logits that pick the top-10 of
    // 512 experts, so it changes which experts run, not only their output.
    let bf16_router = std::env::var("METRALE_QWEN4EXP_BF16_ROUTER").as_deref() != Ok("0");
    let gate_nvfp4 = if bf16_router {
        None
    } else {
        Some(quantize_to_nvfp4(
            &weights.gate,
            config.num_experts,
            h,
            gpu,
            absmax_k,
            quantize_k,
            stream,
        )?)
    };

    let mut moe = MoeLayer::new(weights, config.num_experts, gate_nvfp4, gpu, config)?;

    if bf16_shared {
        let [gate, up, down] = SHARED_PROJS
            .map(|p| dense_auto(store, &format!("{lp}.mlp.shared_expert.{p}.weight"), gpu));
        moe.set_bf16_shared_expert(gate?, up?, down?)?;
    }

    // 2026-09-25: The grouped CUTLASS prefill arm runs only when
    // `cutlass_grouped_host` is set (`moe/forward_prefill_routed.rs`), and
    // `build_cutlass_grouped_sfb` is what sets it. Without transposed scales it
    // reads the original N-major ones.
    if std::env::var("METRALE_HOLO_MOE_GROUPED_CUTLASS")
        .ok()
        .as_deref()
        == Some("1")
    {
        moe.build_cutlass_grouped_sfb(gpu, config, stream)?;
    }

    Ok(FfnComponent::Moe(moe))
}

#[cfg(test)]
#[path = "ffn_tests.rs"]
mod tests;
