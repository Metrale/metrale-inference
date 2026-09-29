// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `CircuitBindings` for `Qwen3SsmLayer`: the GatedDeltaNet layer's weights by slot,
//! its qkvz layout, and the features the circuit does not model.
//!
//! Owner: model-layers (Qwen3 SSM layer).
//! Invariants:
//! - Every arm of `decode_inner` / `ssm_forward` other than the NVFP4 FP32-kernel one the rules
//!   encode is reported in `unmodelled`.

use std::collections::BTreeMap;

use metrale_circuit::LinearRole;

use super::Qwen3SsmLayer;
use crate::circuit_exec::{
    BoundWeight, CircuitBindings, CircuitLayer, GdnFacts, MixerFacts, WeightSlot,
};

impl CircuitBindings for Qwen3SsmLayer {
    fn circuit_prepare(
        &self,
        gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
        config: &metrale_config::ModelConfig,
        levers: &crate::layers::ops::ModelLevers,
        stream: u64,
    ) -> anyhow::Result<()> {
        self.ffn.circuit_prepare(gpu, config, levers, stream)
    }

    fn circuit_layer(
        &self,
        _config: &metrale_config::ModelConfig,
        levers: &crate::layers::ops::ModelLevers,
    ) -> Option<CircuitLayer> {
        let mut unmodelled = Vec::new();
        let arms = [
            (self.hc.is_some(), "hyper-connections"),
            (self.ple.is_some(), "per-layer embeddings"),
            (self.lora_out_proj.is_some(), "an out_proj LoRA adapter"),
            (self.qkvz_fp8w.is_some(), "an FP8 qkvz projection"),
            (self.qkvz_q2.is_some(), "a packed-Q2 qkvz projection"),
            (self.out_proj_fp8w.is_some(), "an FP8 out_proj"),
            (
                self.w8a8.is_some(),
                "declared W8A8 GDN projections (not bound yet)",
            ),
            (
                levers.gdn_fused_conv,
                "the fused GDN conv+norm kernel (METRALE_GDN_FUSED_CONV)",
            ),
            (
                self.conv1d_l2norm_f32_k.0 == 0
                    || self.gdn_f32_k.0 == 0
                    || self.gated_rms_norm_f32_k.0 == 0,
                "the BF16 GDN kernels (an FP32 GDN kernel is not loaded)",
            ),
        ];
        for (present, what) in arms {
            if present {
                unmodelled.push(what.to_string());
            }
        }
        let dense = BoundWeight::Dense;
        let mut weights = BTreeMap::from([
            (WeightSlot::InputNorm, dense(self.input_norm)),
            (WeightSlot::PostNorm, dense(self.post_attn_norm)),
            (
                WeightSlot::Linear(LinearRole::Ba),
                dense(self.ssm.in_proj_ba),
            ),
            (WeightSlot::GdnALog, dense(self.ssm.a_log)),
            (WeightSlot::GdnDtBias, dense(self.ssm.dt_bias)),
            (WeightSlot::GdnConv1d, dense(self.ssm.conv1d)),
            (WeightSlot::GdnNorm, dense(self.ssm.norm)),
        ]);
        weights.insert(
            WeightSlot::Linear(LinearRole::Qkvz),
            match self.qkvz_nvfp4 {
                Some(q) => BoundWeight::Nvfp4(q),
                None => dense(self.ssm.in_proj_qkvz),
            },
        );
        weights.insert(
            WeightSlot::Linear(LinearRole::GdnOut),
            match self.out_proj_dense {
                Some(d) => dense(d),
                None => BoundWeight::Nvfp4(self.ssm.out_proj),
            },
        );
        for (role, twin) in [
            (LinearRole::Qkvz, self.qkvz_nvfp4_t),
            (LinearRole::GdnOut, self.out_proj_nvfp4_t),
        ] {
            if let Some(t) = twin {
                weights.insert(WeightSlot::Transposed(role), BoundWeight::Nvfp4(t));
            }
        }
        self.ffn.circuit_bind(levers, &mut weights, &mut unmodelled);
        Some(CircuitLayer {
            mixer: MixerFacts::Gdn(GdnFacts {
                qkvz_deinterleaved: self.sequential_qkvz,
            }),
            weights,
            unmodelled,
        })
    }
}
