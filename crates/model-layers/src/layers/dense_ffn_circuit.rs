// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `DenseFfnLayer::circuit_bind`: the dense FFN's weights for the circuit executor,
//! and the FFN features the circuit does not model.
//!
//! Owner: model-layers (dense FFN).
//! Invariants:
//! - Only the NVFP4 SiLU FFN without an adapter is bound; every other arm of `forward` is
//!   reported as unmodelled.
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
            (self.lora.is_some(), "an FFN LoRA adapter"),
            (
                self.activation != FfnActivation::SiLU,
                "a non-SiLU FFN activation",
            ),
            (
                self.w8a8.is_some(),
                "declared W8A8 FFN projections (not bound yet)",
            ),
            (
                self.single_row_w4a4(),
                "a declared-W4A4 single-row FFN (not bound yet)",
            ),
            (
                !(mmq_gate_up && mmq_down && self.nvfp4_silu_quant_k.0 != 0),
                "a wide-batch FFN arm other than NVFP4 MMQ with the fused down quantize",
            ),
        ];
        for (present, what) in arms {
            if present {
                unmodelled.push(what.to_string());
            }
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
