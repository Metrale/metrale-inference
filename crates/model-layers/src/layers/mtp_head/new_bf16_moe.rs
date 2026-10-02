// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: Construction of the MTP drafter's MoE layer on the checkpoint's BF16 experts (an
//! MTP head excluded from quantization, as nvidia/Qwen3.6-35B-A3B-NVFP4 ships it), so the batched
//! propose runs the drafter's MoE grouped (`MoeLayer::forward_bf16_grouped_decode`) instead of
//! per-expert GEMVs per sequence (`moe_forward_generic`).
//!
//! Owner: model-layers (MTP head).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use super::MtpHead;
use crate::layers::MoeLayer;
use crate::weight_map::{DenseWeight, ExpertWeight, MoeWeights, MtpWeights};

impl MtpHead {
    /// 2026-10-02: The drafter's MoE on the checkpoint's BF16 tensors: null NVFP4 routed and shared
    /// slots, the BF16 router (`gate_nvfp4 = None`), then `MoeLayer::set_bf16_experts` with the
    /// routed experts' and the shared expert's BF16 weights (no copy). Errors when the expert count
    /// differs from the config's.
    pub(super) fn new_bf16_moe(
        weights: &MtpWeights,
        config: &metrale_config::ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<MoeLayer> {
        anyhow::ensure!(
            weights.experts.len() == config.num_experts,
            "MTP BF16 experts: {} loaded, config says {}",
            weights.experts.len(),
            config.num_experts
        );
        let moe_weights = MoeWeights {
            gate: weights.moe_gate,
            shared_expert: ExpertWeight::null(),
            shared_expert_gate: weights.shared_expert_gate,
            experts: vec![ExpertWeight::null(); config.num_experts],
            router_pre_norm: None,
            correction_bias: None,
        };
        let mut moe = MoeLayer::new(moe_weights, config.num_experts, None, gpu, config)?;
        let col =
            |f: fn(&crate::weight_map::DenseExpertWeight) -> DenseWeight| -> Vec<DenseWeight> {
                weights.experts.iter().map(f).collect()
            };
        moe.set_bf16_experts(
            &col(|e| e.gate_proj),
            &col(|e| e.up_proj),
            &col(|e| e.down_proj),
            weights.shared_expert.gate_proj.weight,
            weights.shared_expert.up_proj.weight,
            weights.shared_expert.down_proj.weight,
            gpu,
        )?;
        Ok(moe)
    }
}
