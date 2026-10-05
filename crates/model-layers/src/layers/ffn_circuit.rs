// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The FFN's circuit binding (`FfnComponent::circuit_bind`, `circuit_prepare`), split
//! from `layers/mod.rs` by exact copy.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.

use anyhow::Result;

use super::{FfnComponent, moe, ops};

impl FfnComponent {
    /// 2026-09-28: Add the FFN's weights to a circuit binding; an absent FFN is reported as
    /// unmodelled. 2026-10-03: A MoE binds through `MoeLayer::circuit_bind`, returned.
    pub(crate) fn circuit_bind(
        &self,
        config: &metrale_config::ModelConfig,
        levers: &ops::ModelLevers,
        weights: &mut std::collections::BTreeMap<
            crate::circuit_exec::WeightSlot,
            crate::circuit_exec::BoundWeight,
        >,
        unmodelled: &mut Vec<String>,
    ) -> Option<moe::MoeBinding> {
        match self {
            Self::Dense(d) => {
                d.circuit_bind(levers, weights, unmodelled);
                None
            }
            Self::Moe(m) => m.circuit_bind(config, levers, unmodelled),
            Self::None => {
                unmodelled.push("no FFN".to_string());
                None
            }
        }
    }

    /// 2026-09-28: Build what a dense FFN's binding hands out (`DenseFfnLayer::circuit_prepare`);
    /// nothing for a MoE or an absent FFN, which bind as unmodelled.
    pub(crate) fn circuit_prepare(
        &self,
        gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
        config: &metrale_config::ModelConfig,
        levers: &ops::ModelLevers,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Dense(d) => d.circuit_prepare(gpu, config, levers, stream),
            Self::Moe(_) | Self::None => Ok(()),
        }
    }
}
