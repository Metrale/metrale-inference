// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The dense FFN under a fixed `--activation-quantization` for the `ffn` family: one
//! activation format at every row count, so a row's output bits do not depend on how many rows
//! share the launch. Decode sites (the one-row decode, the multi-sequence arms of both layer
//! kinds, the batched verify's GDN arm) ask [`DenseFfnLayer::fixed_ok`] first.
//!
//! Owner: model-layers (dense FFN).
//! Invariants:
//! - `fixed_ok` is false for `adaptive` row counts, so those take today's arms unchanged.
//! - `forward_fixed` launches only when `fixed_ok` holds, and writes `moe_output` rows `0..m`.

use anyhow::Result;
use metrale_config::{ActQuantFormat, ProjFamily};
use metrale_gpu_runtime::gpu::DevicePtr;

use super::DenseFfnLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

/// 2026-09-30: How a layer runs a fixed format, if it can.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fixed {
    /// 2026-09-30: The declared W8A8 layer (`dense_ffn_w8a8.rs`): per-row FP8 at every row count.
    W8a8,
    /// 2026-09-30: NVFP4 weights, NVFP4 activations (`ops::w4a4_proj::nvfp4_proj_mx`).
    W4a4,
}

impl DenseFfnLayer {
    fn fixed_route(
        &self,
        m: usize,
        gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    ) -> Option<Fixed> {
        let format = crate::layers::fixed_act(ProjFamily::Ffn, m)?;
        if self.lora.is_some() {
            return None;
        }
        if self.w8a8.is_some() {
            return matches!(format, ActQuantFormat::Declared | ActQuantFormat::Fp8)
                .then_some(Fixed::W8a8);
        }
        let nvfp4 = !self.weights.gate_proj.weight.is_null()
            && ops::w4a4_proj::max_rows(gpu) > 0
            && self.activation == super::FfnActivation::SiLU;
        let wants = match format {
            ActQuantFormat::Nvfp4 => true,
            ActQuantFormat::Declared => self.w4a16_batchm.declares_a4(&self.weights.gate_proj),
            _ => false,
        };
        (nvfp4 && wants).then_some(Fixed::W4a4)
    }

    /// 2026-09-30: Whether `m` rows run a fixed format on this layer.
    pub fn fixed_ok(&self, m: usize, ctx: &ForwardContext) -> bool {
        self.fixed_route(m, ctx.gpu).is_some()
    }

    /// 2026-09-30: The FFN of `m` rows at `input` under the fixed format, into `moe_output`.
    pub fn forward_fixed(
        &self,
        input: DevicePtr,
        m: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self.fixed_route(m, ctx.gpu) {
            Some(Fixed::W8a8) => {
                anyhow::ensure!(
                    self.forward_w8a8(input, m, ctx, stream)?,
                    "ffn: the declared W8A8 arm declined {m} rows"
                );
                Ok(())
            }
            Some(Fixed::W4a4) => {
                let h = ctx.config.hidden_size as u32;
                let inter = ctx.config.intermediate_size as u32;
                let (gate, up) = (ctx.buffers.expert_gate_out(), ctx.buffers.expert_up_out());
                let w = &self.weights;
                ops::w4a4_proj::nvfp4_proj_mx(
                    ctx.gpu,
                    input,
                    &w.gate_proj,
                    gate,
                    m as u32,
                    inter,
                    h,
                    stream,
                )?;
                ops::w4a4_proj::nvfp4_proj_mx(
                    ctx.gpu, input, &w.up_proj, up, m as u32, inter, h, stream,
                )?;
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate,
                    up,
                    gate,
                    m as u32 * inter,
                    stream,
                )?;
                ops::w4a4_proj::nvfp4_proj_mx(
                    ctx.gpu,
                    gate,
                    &w.down_proj,
                    ctx.buffers.moe_output(),
                    m as u32,
                    h,
                    inter,
                    stream,
                )
            }
            None => anyhow::bail!("ffn: no fixed format for {m} rows on this layer"),
        }
    }
}
