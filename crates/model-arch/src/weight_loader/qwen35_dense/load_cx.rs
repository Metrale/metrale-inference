// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The inputs `Qwen35DenseWeightLoader::load_layers` hands to its per-layer
//! helpers (`ffn_arm`, `attn_layer`, `gdn_layer`): the load-wide values in [`LoadCx`], one
//! layer's norms and FFN in [`LayerIn`], and [`Flow`], which tells the layer loop whether
//! to run its end-of-layer progress step.
//!
//! Owner: model-arch weight loader (Qwen3.5 dense).
//! Invariants:
//! - Every field of [`LoadCx`] is computed once per load, before the layer loop, and is
//!   not changed inside it.

use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};
use metrale_model_layers::layers::FfnComponent;
use metrale_model_layers::weight_map::{DenseWeight, Nvfp4Variant};
use metrale_model_weights::weights::WeightStore;

use super::fp8_residency::RouteEnv;

/// 2026-09-26: The values `load_layers` computes before its layer loop and every layer
/// reads.
pub(super) struct LoadCx<'a> {
    pub(super) store: &'a WeightStore,
    pub(super) config: &'a ModelConfig,
    pub(super) gpu: &'a dyn GpuBackend,
    pub(super) layer_kv_dtypes: &'a [KvCacheDtype],
    pub(super) variant: Nvfp4Variant,
    pub(super) absmax_k: KernelHandle,
    pub(super) quantize_k: KernelHandle,
    pub(super) stream: u64,
    pub(super) h: usize,
    pub(super) bf16_to_fp8_k: Option<KernelHandle>,
    pub(super) route_env: &'a RouteEnv,
    /// 2026-09-28: `--weight-quantization` over this checkpoint's declared plan.
    pub(super) policy: metrale_config::WeightQuantPolicy<'a>,
}

impl LoadCx<'_> {
    /// 2026-09-28: The activation stamp for `module`'s NVFP4 weight (a checkpoint module
    /// path, e.g. `{lp}.mlp.gate_proj`), from the weight-quantization policy
    /// (`WeightQuantPolicy::nvfp4_act`). The loader stamps it on the weight
    /// (`QuantizedWeight::act`), and decode and prefill dispatch follow it.
    pub(super) fn nvfp4_act(&self, module: &str) -> metrale_config::Nvfp4Act {
        self.policy.nvfp4_act(module)
    }
}

/// 2026-09-26: One layer's index, prefix, norms and built FFN, which every layer arm
/// passes to its layer constructor.
pub(super) struct LayerIn<'a> {
    pub(super) i: usize,
    pub(super) lp: &'a str,
    pub(super) input_norm: DenseWeight,
    pub(super) post_attn_norm: DenseWeight,
    pub(super) ffn: FfnComponent,
}

/// 2026-09-26: `Continue` makes the layer loop skip its end-of-layer progress step for
/// this layer; `Proceed` runs it.
pub(super) enum Flow {
    Continue,
    Proceed,
}

/// 2026-09-28: `--weight-quantization` as published, over the checkpoint's declared plan.
pub(super) fn weight_quant_policy(config: &ModelConfig) -> metrale_config::WeightQuantPolicy<'_> {
    metrale_config::WeightQuantPolicy::for_checkpoint(
        metrale_model_layers::layers::weight_quantization(),
        config.quantization_config.as_ref(),
        metrale_model_layers::layers::kernel_caps(),
    )
}

/// 2026-09-28: Under `--weight-quantization declared`, one line saying how many dense-FFN
/// layers run W4A4 as declared and how many projections the checkpoint declares FP8 (served
/// as NVFP4 with 16-bit activations until the per-row FP8 decode arms land).
pub(super) fn log_declared_plan(
    policy: &metrale_config::WeightQuantPolicy<'_>,
    config: &ModelConfig,
    layer_types: &[LayerType],
) {
    if !policy.follows_plan() {
        return;
    }
    let (mut a4, mut fp8) = (0usize, 0usize);
    for (i, lt) in layer_types.iter().enumerate() {
        let lp = config.layer_prefix(i);
        let projs: &[&str] = match lt {
            LayerType::FullAttention => &["self_attn.q_proj", "self_attn.o_proj"],
            _ => &["linear_attn.in_proj_qkv", "linear_attn.out_proj"],
        };
        a4 += usize::from(
            policy.nvfp4_act(&format!("{lp}.mlp.gate_proj")) == metrale_config::Nvfp4Act::A4,
        );
        for m in projs.iter().chain(&["mlp.gate_proj"]) {
            fp8 += usize::from(policy.wants_fp8_weights(&format!("{lp}.{m}")));
        }
    }
    tracing::info!(
        "--weight-quantization declared: {a4}/{} dense-FFN layers run W4A4 as declared; \
         {fp8} sampled projections are declared FP8 and run as NVFP4 with 16-bit activations \
         (no per-row FP8 decode arm yet)",
        layer_types.len()
    );
}
