// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `DenseFfnLayer::circuit_bind`: the dense FFN's weights for the circuit executor,
//! and the FFN features the circuit does not model.
//!
//! Owner: model-layers (dense FFN).
//! Invariants:
//! - Only the NVFP4 SiLU FFN without an adapter is bound; every other arm of `forward` is
//!   reported as unmodelled.

use std::collections::BTreeMap;

use metrale_circuit::LinearRole;

use super::super::{DenseFfnLayer, FfnActivation};
use crate::circuit_exec::{BoundWeight, WeightSlot};

impl DenseFfnLayer {
    /// 2026-09-28: Add this FFN's weights to `weights`; add to `unmodelled` what the circuit
    /// cannot run.
    pub(crate) fn circuit_bind(
        &self,
        weights: &mut BTreeMap<WeightSlot, BoundWeight>,
        unmodelled: &mut Vec<String>,
    ) {
        let arms = [
            (self.q2_weights.is_some(), "packed-Q2 FFN weights"),
            (self.fp8_weights.is_some(), "FP8 FFN weights"),
            (self.bf16_weights.is_some(), "BF16 FFN weights"),
            (self.lora.is_some(), "an FFN LoRA adapter"),
            (
                self.activation != FfnActivation::SiLU,
                "a non-SiLU FFN activation",
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
    }
}
