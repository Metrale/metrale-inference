// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `METRALE_GLM_KDA_GATE_IN_PREPARE=1`: where the 64-token chunked scan
//! (`chunk64.rs`) takes a forward, its prepare computes the KDA forget gate itself
//! (`kda_chunk64_prepare_graw` / `kda_chunk64_prepare_conv_graw` in
//! `kernels/gb10/common/kda_chunk64_prepare.cu`): each (chunk, head) tile reads f_b's BF16 output
//! rows (`g_raw`) and applies `kda_gate_scalar` with the channel's `dt_bias`, `exp(A_log[h])` and
//! the lower bound, as `kda_gate_bf16` does, instead of one `kda_gate_bf16` launch writing every
//! FP32 gate row to DRAM for the prepare to read back.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - Every byte of `core`, the recurrent state and the slab factors equals the gate-kernel path's:
//!   the same element function in the same module, and the gate rows past the last row are zero
//!   as the FP32 copy's zero fill makes them; `examples/glm5next_kda_gate_in_prepare_parity.rs`
//!   compares them on a GPU.
//! - Under [`GateAt::Prepare`] nothing writes `ws.gate`, so a verify record of such a forward is
//!   refused ([`Glm5NextKdaLayer::gate_in_prepare_skips_gate`]).
//! - One decision per forward ([`Glm5NextKdaLayer::gate_at_for`]), handed to both the front end
//!   (which skips the gate launch) and the scan (which picks the `_graw` prepare).
//! - The lever set without the `_graw` kernels fails the forward before any launch.

use super::*;

/// 2026-10-14: Where a forward computes the KDA forget gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GateAt {
    /// `kda_gate_bf16` in the front end, into `ws.gate`.
    Kernel,
    /// Inside the 64-token scan's prepare, from `ws.g_raw`.
    Prepare,
}

impl Glm5NextKdaLayer {
    /// 2026-10-14: Where a forward on the 64-token scan (`chunked64`) computes the gate: in the
    /// prepare when the lever is set (an error without the kernels), else in the front end.
    pub(crate) fn gate_at_for(&self, chunked64: bool) -> Result<GateAt> {
        if !chunked64 || !self.levers.kda.gate_in_prepare {
            return Ok(GateAt::Kernel);
        }
        self.gate_in_prepare_check()?;
        Ok(GateAt::Prepare)
    }

    /// 2026-10-14: The refusal of [`GateAt::Prepare`], before any launch.
    pub(crate) fn gate_in_prepare_check(&self) -> Result<()> {
        let s = &self.kernels.seq;
        if s.chunk64_prepare_graw.0 == 0 || s.chunk64_prepare_conv_graw.0 == 0 {
            bail!(
                "METRALE_GLM_KDA_GATE_IN_PREPARE=1: needs kda_chunk64_prepare_graw and \
                 kda_chunk64_prepare_conv_graw"
            );
        }
        Ok(())
    }

    /// 2026-10-14: Whether a forward of `k` rows of one sequence leaves `ws.gate` unwritten: the
    /// lever is set and the 64-token scan applies. `record_verify_rows` refuses such a forward
    /// (it copies `ws.gate`).
    pub(crate) fn gate_in_prepare_skips_gate(&self, k: usize) -> Result<bool> {
        Ok(self.levers.kda.gate_in_prepare && self.chunk64_applies(k, self.levers.kda.chunk64))
    }

    /// 2026-10-14: f_b (head_dim -> heads * head_dim) into `ws.g_raw`, then, at
    /// [`GateAt::Kernel`], `kda_gate_bf16` into `ws.gate`.
    pub(super) fn forget_gate(
        &self,
        gpu: &dyn GpuBackend,
        t: usize,
        ws: &Glm5NextKdaWorkspace,
        gate: GateAt,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (qkv, hd) = (c.qkv_dim(), c.head_dim);
        self.gemm(
            gpu,
            ws.lowrank,
            &self.weights.f_b,
            ws.g_raw,
            t,
            qkv,
            hd,
            stream,
        )?;
        if gate == GateAt::Prepare {
            return Ok(());
        }
        KernelLaunch::new(gpu, self.kernels.gate)
            .grid([(t * c.heads) as u32, 1, 1])
            .block([BLOCK, 1, 1])
            .arg_ptr(ws.g_raw)
            .arg_ptr(self.weights.dt_bias)
            .arg_ptr(self.weights.a_log)
            .arg_ptr(ws.gate)
            .arg_u32(t as u32)
            .arg_u32(c.heads as u32)
            .arg_u32(hd as u32)
            .arg_f32(c.gate_lower_bound)
            .launch(stream)
    }

    /// 2026-10-14: The prepare's gate arguments: `ws.gate` at [`GateAt::Kernel`], else `g_raw`,
    /// `dt_bias`, `A_log` and the lower bound (the `_graw` entries' order).
    pub(super) fn gate_args<'a>(
        &self,
        l: KernelLaunch<'a>,
        ws: &Glm5NextKdaWorkspace,
        gate: GateAt,
    ) -> KernelLaunch<'a> {
        match gate {
            GateAt::Kernel => l.arg_ptr(ws.gate),
            GateAt::Prepare => l
                .arg_ptr(ws.g_raw)
                .arg_ptr(self.weights.dt_bias)
                .arg_ptr(self.weights.a_log)
                .arg_f32(self.cfg.gate_lower_bound),
        }
    }
}

#[cfg(test)]
#[path = "gate_at_tests.rs"]
mod tests;
