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
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};
use metrale_model_layers::layers::FfnComponent;
use metrale_model_layers::weight_map::{DenseWeight, Nvfp4Variant};
use metrale_model_weights::weights::WeightStore;

use super::fp8_residency::RouteEnv;
use super::served_formats::{Group, Served, ServedFormats, declared_weight_bits};

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
    /// 2026-09-30: What each arm built, for the load summary (`served_formats`).
    pub(super) served: &'a std::cell::RefCell<ServedFormats>,
}

impl LoadCx<'_> {
    /// 2026-09-28: The activation stamp for `module`'s NVFP4 weight (a checkpoint module
    /// path, e.g. `{lp}.mlp.gate_proj`), from the weight-quantization policy
    /// (`WeightQuantPolicy::nvfp4_act`). The loader stamps it on the weight
    /// (`QuantizedWeight::act`), and decode and prefill dispatch follow it.
    pub(super) fn nvfp4_act(&self, module: &str) -> metrale_config::Nvfp4Act {
        self.policy.nvfp4_act(module)
    }

    /// 2026-09-30: Record that layer `layer`'s `group` decodes from `served`; `module` (e.g.
    /// `{lp}.mlp.gate_proj`) is the projection whose declaration the group is judged against.
    pub(super) fn record_served(&self, group: Group, layer: usize, module: &str, served: Served) {
        let bits = declared_weight_bits(self.policy.declared(module));
        self.served.borrow_mut().record(group, layer, served, bits);
    }

    /// 2026-09-30: The NVFP4 form of `module` as built: from the checkpoint when the store
    /// holds it as NVFP4 (a `weight_packed`, or a UInt8 `.weight`), otherwise requantized at
    /// load; `act` is the stamp the arm put on it.
    pub(super) fn nvfp4_served(&self, module: &str, act: metrale_config::Nvfp4Act) -> Served {
        let from_checkpoint = self.store.contains(&format!("{module}.weight_packed"))
            || matches!(
                self.store.get(&format!("{module}.weight")).map(|w| w.dtype),
                Ok(metrale_model_weights::weights::WeightDtype::UInt8)
            );
        Served::Nvfp4 {
            from_checkpoint,
            act,
        }
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
