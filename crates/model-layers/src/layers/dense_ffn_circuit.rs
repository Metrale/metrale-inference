// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `DenseFfnLayer::circuit_bind`: the dense FFN's weights for the circuit executor,
//! and the FFN features the circuit does not model.
//!
//! Owner: model-layers (dense FFN).
//! Invariants:
//! - Only the NVFP4 SiLU FFN without an adapter is bound, and (2026-09-30) the declared W8A8
//!   one and the declared-W4A4 NVFP4 one; every other arm of `forward` is reported as
//!   unmodelled.
//! - The wide-batch arm bound is the NVFP4 MMQ one with the fused down quantize, as
//!   `fp4mmq_arms` answers it; any other `forward_prefill` arm is unmodelled.

use std::collections::BTreeMap;

use metrale_circuit::LinearRole;

use super::super::{DenseFfnLayer, FfnActivation};
use crate::circuit_exec::{BoundWeight, WeightSlot};

impl DenseFfnLayer {
    /// 2026-09-28: Add this FFN's weights to `weights`; add to `unmodelled` what the circuit
    /// cannot run.
    pub(crate) fn circuit_bind(
        &self,
        levers: &crate::layers::ops::ModelLevers,
        weights: &mut BTreeMap<WeightSlot, BoundWeight>,
        unmodelled: &mut Vec<String>,
    ) {
        let (mmq_gate_up, mmq_down) = self.fp4mmq_arms(levers);
        let arms = [
            (self.q2_weights.is_some(), "packed-Q2 FFN weights"),
            (self.fp8_weights.is_some(), "FP8 FFN weights"),
            (self.bf16_weights.is_some(), "BF16 FFN weights"),
            (
                self.activation != FfnActivation::SiLU,
                "a non-SiLU FFN activation",
            ),
            (
                // 2026-10-03: An adapter turns the MMQ arms off (`fp4mmq_arms`); with one, the
                // executor serves only the narrow rows that run no wide arm (LoRA phase 1).
                self.lora.is_none()
                    && self.w8a8.is_none()
                    && !(mmq_gate_up && mmq_down && self.nvfp4_silu_quant_k.0 != 0),
                "a wide-batch FFN arm other than NVFP4 MMQ with the fused down quantize",
            ),
        ];
        for (present, what) in arms {
            if present {
                unmodelled.push(what.to_string());
            }
        }
        // 2026-09-30: A declared W8A8 FFN runs W8A8 at every decode row count up to
        // `ops::W8A8_MAX_ROWS` (`forward_w8a8`, tried first by every entry point), so its
        // NVFP4 copies and their MMQ repacks serve no row the circuit plans.
        if let Some(w) = self.w8a8 {
            if w.ctx.kernels.resolved(w.gate.scale()) {
                for (slot, p) in [
                    (WeightSlot::FfnGate, w.gate),
                    (WeightSlot::FfnUp, w.up),
                    (WeightSlot::Linear(LinearRole::Down), w.down),
                ] {
                    weights.insert(slot, BoundWeight::W8a8(p, w.ctx.kernels));
                }
            } else {
                unmodelled.push("W8A8 FFN projections whose kernels did not resolve".to_string());
            }
            return;
        }
        weights.insert(
            WeightSlot::FfnGate,
            BoundWeight::Nvfp4(self.weights.gate_proj),
        );
        weights.insert(WeightSlot::FfnUp, BoundWeight::Nvfp4(self.weights.up_proj));
        weights.insert(
            WeightSlot::Linear(LinearRole::Down),
            BoundWeight::Nvfp4(self.weights.down_proj),
        );
        let repacked = [
            (WeightSlot::FfnGateMmq, self.fp4mmq_gate.get()),
            (WeightSlot::FfnUpMmq, self.fp4mmq_up.get()),
            (WeightSlot::FfnDownMmq, self.fp4mmq_down.get()),
        ];
        // 2026-10-03: An adapter turns the MMQ arms off (`fp4mmq_arms`), so no repack is built
        // and no rule a LoRA plan selects reads one (they state `lora_active = "off"`).
        if self.lora.is_some() {
            return;
        }
        for (slot, w) in repacked {
            match w {
                Some(w) => {
                    weights.insert(slot, BoundWeight::Mmq(w.w));
                }
                None => unmodelled.push(format!("{slot:?} is not built (no circuit_prepare)")),
            }
        }
    }

    /// 2026-09-28: Build the MMQ repacks `circuit_bind` hands out, when the MMQ arm is live
    /// (`fp4mmq_arms`); the load normally built them already (`finalize_nvfp4_mmq_load`).
    /// Waits for `stream`.
    pub(crate) fn circuit_prepare(
        &self,
        gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
        config: &metrale_config::ModelConfig,
        levers: &crate::layers::ops::ModelLevers,
        stream: u64,
    ) -> anyhow::Result<()> {
        let (gate_up, down) = self.fp4mmq_arms(levers);
        let h = u32::try_from(config.hidden_size)?;
        let inter = u32::try_from(config.intermediate_size)?;
        if gate_up {
            self.build_nvfp4_mmq_gate_up(gpu, h, inter, stream)?;
        }
        if down {
            self.ensure_nvfp4_mmq_weight(
                &self.fp4mmq_down,
                gpu,
                &self.weights.down_proj,
                h,
                inter,
                stream,
            )?;
        }
        gpu.synchronize(stream)
    }
}
