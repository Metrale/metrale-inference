// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The declared-W8A8 arm of the dense SiLU FFN (`crate::layers::W8a8Ffn`): the
//! gate and up projections share one quantized input, and the down projection's input,
//! `bf16(silu(gate) * up)`, is quantized in the same launch that computes it. Every entry
//! point (`forward`, `forward_k2`, `forward_k3`, `forward_km`, `forward_prefill`) tries it
//! first for `1..=ops::W8A8_MAX_ROWS` rows.
//!
//! Owner: model-layers (dense FFN).
//! Invariants:
//! - On `Ok(true)` the `[rows, hidden]` output is in `ctx.buffers.moe_output()`, as for the
//!   other arms; `expert_gate_out` / `expert_up_out` hold the gate and up rows.
//! - `Ok(false)` launches nothing: no W8A8 weights, a LoRA overlay, or a row count outside the
//!   family.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::DevicePtr;

use super::{DenseFfnLayer, FfnActivation};
use crate::layer::ForwardContext;
use crate::layers::{W8a8Ffn, ops};

impl DenseFfnLayer {
    /// 2026-09-28: Install the W8A8 gate, up and down. Refuses a GELU FFN and shapes other than
    /// `[inter, hidden]` (gate, up) and `[hidden, inter]` (down).
    pub fn set_w8a8_decode_weights(&mut self, w: W8a8Ffn, hidden: u32, inter: u32) -> Result<()> {
        ensure!(
            self.activation == FfnActivation::SiLU,
            "W8A8 FFN arm fuses SiLU only"
        );
        let shapes = [&w.gate, &w.up, &w.down].map(|p| (p.n(), p.k()));
        ensure!(
            shapes == [(inter, hidden), (inter, hidden), (hidden, inter)],
            "W8A8 FFN shapes {shapes:?}, want gate/up [{inter}, {hidden}] and down [{hidden}, {inter}]"
        );
        self.w8a8 = Some(w);
        Ok(())
    }

    /// 2026-09-28: The W8A8 FFN over `rows` rows of `input` (`[rows, hidden]` BF16).
    pub(super) fn forward_w8a8(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(ref w) = self.w8a8 else {
            return Ok(false);
        };
        if self.lora.is_some() || !w.ctx.available(&w.gate, rows) || !w.ctx.available(&w.down, rows)
        {
            return Ok(false);
        }
        let (h, inter) = (w.gate.k(), w.gate.n());
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();
        let output = ctx.buffers.moe_output();
        w.ctx
            .proj(ctx.gpu, &w.gate, input, h, rows, gate_out, inter, stream)?;
        // 2026-09-28: `up` reads the activation `gate` quantized into the shared scratch.
        ops::w8a8_gemv(
            ctx.gpu,
            &w.ctx.kernels,
            &w.up,
            &w.ctx.scratch,
            rows,
            up_out,
            inter,
            stream,
        )?;
        w.ctx.silu_proj(
            ctx.gpu, &w.down, gate_out, up_out, inter, rows, output, h, stream,
        )
    }
}
