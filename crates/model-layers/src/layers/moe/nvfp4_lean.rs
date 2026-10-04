// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: The lean point of the tensor-core grouped NVFP4 decode (`Nvfp4G16Lean`,
//! `kernels/gb10/common/tc_weight_formats.cuh`): the layer's declared NVFP4 experts, routed and
//! shared, are repacked IN PLACE once at load (`nvfp4_tc_lean_repack`: same bytes, same size, no
//! extra memory) into a tile-contiguous layout whose E2M1 bits a fragment extracts with rotates
//! and masks and whose scale bytes become one IMAD each. The `_lean` expert kernels then return
//! the row-major tensor-core kernels' output bytes exactly (`nvfp4_moe_grouped_microtest`,
//! `all-nvfp4-tc-lean`) at 10-17 % less expert-kernel energy at equal time.
//!
//! Why in place: the row-major tables are read by nothing else once the prefill copies exist,
//! and a second copy of every expert would cost the layer's expert memory again.
//!
//! Owner: model-layers (MoE).
//! Invariants:
//! - A layer is repacked only when its prefill reads copies (the `*_t` tables built, no CUTLASS
//!   tables, no unified layout), its experts are the checkpoint's declared NVFP4 and all local,
//!   the tensor-core decode and its lean pair resolved for its shape, and no lever sends its
//!   decode to the legacy kernels (FP32 gate or routing, grouped decode off).
//! - After the repack `Nvfp4GroupedKernels::lean` holds and only the lean pair reads the tables;
//!   every arm that would read them in the row-major layout calls `refuse_lean_layout` first.

use anyhow::Context;

use super::forward_nvfp4_grouped_decode::{Nvfp4GroupedKernels, Nvfp4GroupedLaunch};
use super::*;

impl Nvfp4GroupedKernels {
    /// 2026-10-04: The lean pair, once this layer's tables are lean.
    pub(super) fn lean_launch(&self) -> Option<Nvfp4GroupedLaunch> {
        self.lean.then_some(Nvfp4GroupedLaunch {
            gate_up: self.gate_up_tc_lean,
            gate_up_geometry: ops::NVFP4_GROUPED_GATE_UP_TC,
            down: self.down_tc_lean,
            down_geometry: ops::NVFP4_GROUPED_DOWN_TC,
            max_rows: super::NVFP4_GROUPED_DECODE_TC_MAX_ROWS,
        })
    }
}

impl MoeLayer {
    /// 2026-10-04: Whether this layer's NVFP4 expert tables are in the lean layout.
    pub fn nvfp4_experts_lean(&self) -> bool {
        self.nvfp4_grouped.lean
    }

    /// 2026-10-04: Fails an arm that would read this layer's NVFP4 expert tables in the
    /// row-major layout once they are lean (the bytes would decode as other values).
    pub(super) fn refuse_lean_layout(&self, arm: &str) -> Result<()> {
        anyhow::ensure!(
            !self.nvfp4_grouped.lean,
            "{arm}: this layer's NVFP4 experts were repacked for the lean tensor-core decode at \
             load; only `forward_nvfp4_grouped_decode` reads them (set METRALE_NO_MOE_NVFP4_TC to \
             keep the row-major layout)"
        );
        Ok(())
    }

    /// 2026-10-04: Why this layer keeps the row-major layout, or `None` when it qualifies for the
    /// lean one (module invariants). Call after the load-time prefill passes.
    pub fn nvfp4_lean_refusal(&self, config: &metrale_config::ModelConfig) -> Option<&'static str> {
        let (h, inter) = (
            config.hidden_size as u32,
            config.moe_intermediate_size as u32,
        );
        let k = &self.nvfp4_grouped;
        let levers = ops::ModelLevers::get();
        let local = |q: &QuantizedWeight| !q.weight.is_null() && !q.weight_scale.is_null();
        let checks: [(bool, &'static str); 12] = [
            (
                k.declared_experts
                    && self.fp8_shared_expert.is_none()
                    && self.fp8_down_weight_ptrs.is_none(),
                "experts are not the checkpoint's declared NVFP4 alone (the FP8 kernels would read \
                 the lifted SiLU rows)",
            ),
            (
                super::forward_nvfp4_grouped_decode::nvfp4_grouped_tc_enabled(),
                "METRALE_NO_MOE_NVFP4_TC",
            ),
            (
                k.gate_up_tc_lean.0 != 0 && k.down_tc_lean.0 != 0 && k.lean_repack.0 != 0,
                "lean kernels not in the kernel set",
            ),
            (
                ops::nvfp4_grouped_tc_shape_ok(inter, h, ops::NVFP4_GROUPED_GATE_UP_TC)
                    && ops::nvfp4_grouped_tc_shape_ok(h, inter, ops::NVFP4_GROUPED_DOWN_TC)
                    && ops::nvfp4_lean_repack_shape_ok(inter, h)
                    && ops::nvfp4_lean_repack_shape_ok(h, inter)
                    && config.shared_expert_intermediate_size == config.moe_intermediate_size,
                "shape",
            ),
            (
                self.gate_ptrs_t.is_some()
                    && self.up_ptrs_t.is_some()
                    && self.down_ptrs_t.is_some()
                    && self.shared_gate_t.is_some()
                    && self.shared_up_t.is_some()
                    && self.shared_down_t.is_some(),
                "prefill reads the row-major tables (no transposed copies)",
            ),
            (
                self.cutlass_grouped_host.is_none() && !self.unified_layout,
                "CUTLASS grouped prefill or the unified layout",
            ),
            (
                !self.weights.experts.is_empty()
                    && self
                        .weights
                        .experts
                        .iter()
                        .all(|e| local(&e.gate_proj) && local(&e.up_proj) && local(&e.down_proj))
                    && config.ep_world_size <= 1,
                "experts not all local",
            ),
            (
                local(&self.weights.shared_expert.gate_proj)
                    && local(&self.weights.shared_expert.up_proj)
                    && local(&self.weights.shared_expert.down_proj),
                "no NVFP4 shared expert",
            ),
            (
                self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
                    && self.shared_experts_scale_kind
                        == crate::weight_map::WeightQuantFormat::Nvfp4,
                "scale format",
            ),
            (
                !levers.fp32_gate && !self.fp32_routing_active(levers),
                "FP32 gate or routing",
            ),
            (
                crate::layers::moe_grouped_decode_enabled(),
                "METRALE_NO_MOE_GROUPED_DECODE",
            ),
            (self.lora.is_none(), "MoE LoRA"),
        ];
        checks.iter().find(|(ok, _)| !ok).map(|(_, why)| *why)
    }

    /// 2026-10-04: Repacks this layer's routed and shared NVFP4 experts in place for the lean
    /// decode when it qualifies (`nvfp4_lean_refusal`), on `stream`, and synchronizes it. Returns
    /// whether it did. Fails the load on a negative or NaN scale.
    pub fn repack_nvfp4_experts_lean(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &metrale_config::ModelConfig,
        stream: u64,
    ) -> Result<bool> {
        if let Some(why) = self.nvfp4_lean_refusal(config) {
            tracing::debug!("NVFP4 experts keep the row-major layout: {why}");
            return Ok(false);
        }
        let (h, inter) = (
            config.hidden_size as u32,
            config.moe_intermediate_size as u32,
        );
        let kernel = self.nvfp4_grouped.lean_repack;
        let bad = gpu.alloc(16)?;
        gpu.copy_h2d(&[0u8; 4], bad)?;
        let sh = &self.weights.shared_expert;
        let mats = self
            .weights
            .experts
            .iter()
            .chain(std::iter::once(sh))
            .flat_map(|e| {
                [
                    (&e.gate_proj, inter, h),
                    (&e.up_proj, inter, h),
                    (&e.down_proj, h, inter),
                ]
            });
        let launched = mats
            .map(|(q, n, k)| {
                ops::nvfp4_tc_lean_repack(gpu, kernel, q.weight, q.weight_scale, n, k, bad, stream)
            })
            .collect::<Result<Vec<()>>>();
        let synced = launched.and_then(|_| gpu.synchronize(stream));
        let mut flag = [0u8; 4];
        let read = synced.and_then(|()| gpu.copy_d2h(bad, &mut flag));
        gpu.free(bad)?;
        read.context("nvfp4_tc_lean_repack")?;
        anyhow::ensure!(
            flag == [0u8; 4],
            "nvfp4_tc_lean_repack: a negative or NaN E4M3 block scale in this layer's NVFP4 experts"
        );
        self.nvfp4_grouped.lean = true;
        Ok(true)
    }
}
