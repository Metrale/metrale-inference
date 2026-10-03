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
        // 2026-09-30: The declared W8A8 QKV|Z and out_proj, which every decode path tries first
        // (`w8a8_decode.rs`); installed only on a sequential-QKVZ layer, the layout
        // `qkvz_deinterleaved` states.
        if let Some(w) = self.w8a8 {
            if w.ctx.kernels.resolved(w.input.scale()) {
                weights.insert(
                    WeightSlot::Linear(LinearRole::Qkvz),
                    BoundWeight::W8a8(w.input, w.ctx.kernels),
                );
                weights.insert(
                    WeightSlot::Linear(LinearRole::GdnOut),
                    BoundWeight::W8a8(w.output, w.ctx.kernels),
                );
            } else {
                unmodelled.push("W8A8 GDN projections whose kernels did not resolve".to_string());
            }
        }
        for (role, twin) in [
            (LinearRole::Qkvz, self.qkvz_nvfp4_t),
            (LinearRole::GdnOut, self.out_proj_nvfp4_t),
        ] {
            if let Some(t) = twin {
                weights.insert(WeightSlot::Transposed(role), BoundWeight::Nvfp4(t));
            }
        }
        // 2026-10-03: Prefill (M6a). The unscaled E4M3 casts the prefill projections read, bound
        // only where the legacy prefill takes the cast arm (`trait_prefill_proj.rs:274`,
        // `trait_prefill_helper.rs:207`): every arm before it needs a weight this layer does not
        // hold (a packed-Q2 or row-wise FP8 qkvz, a block-scaled FP8 or BF16 out_proj), and the
        // qkvz arm writes `[Q | K | V | Z]` itself only on a sequential-QKVZ layer. Without a
        // slot the prefill emitter refuses the build; decode does not read them.
        let qkvz_cast = self.qkvz_fp8.filter(|_| {
            self.sequential_qkvz
                && self.qkvz_q2.is_none()
                && self.qkvz_fp8w_rowwise.is_none()
                && self.qkvz_fp8w.is_none()
        });
        let out_cast = self.out_proj_fp8.filter(|_| {
            self.out_proj_fp8w_rowwise.is_none()
                && self.out_proj_dense.is_none()
                && self.out_proj_fp8w.is_none()
        });
        for (role, cast) in [
            (LinearRole::Qkvz, qkvz_cast),
            (LinearRole::GdnOut, out_cast),
        ] {
            if let Some(p) = cast {
                weights.insert(
                    WeightSlot::PrefillCast(role),
                    dense(crate::weight_map::DenseWeight { weight: p }),
                );
            }
        }
        self.ffn.circuit_bind(levers, &mut weights, &mut unmodelled);
        Some(CircuitLayer {
            mixer: MixerFacts::Gdn(GdnFacts {
                qkvz_deinterleaved: self.sequential_qkvz,
                h_slot_bytes: self.h_slot_stride_bytes() as u64,
                conv_state_bytes: self.conv_state_bytes as u64,
                carry: self.carry.binding.get().copied(),
            }),
            weights,
            unmodelled,
        })
    }
}
