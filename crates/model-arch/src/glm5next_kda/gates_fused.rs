// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: One token's KDA gate chain in one launch (`METRALE_GLM_KDA_GATES_FUSED=1`):
//! `kda_gates_w8a8_k128` runs the f_b and g_b W8A8 projections of the low-rank rows, their
//! activation quantization, `kda_gate` and the beta sigmoid, where `front_end` otherwise launches
//! act-quant + f_b + gate + sigmoid + act-quant + g_b.
//! 2026-10-21: `kda_gates_w8a8_k128_rows` is the same chain for 1..=[`GATES_MAX_ROWS`] tokens
//! (the verify step's rows), taken by default (`METRALE_GLM_KDA_GATES_ROWS`, `0` turns it off).
//! On GB10 at the TP=3 shape, 34 layers with distinct weights in a graph: six launches 17.8 us
//! -> one 6.2 us a layer at 2 rows, 17.2 -> 5.7 at 1, 18.7 -> 9.5 at 4
//! (`glm5next_kda_gates_rows_gate`).
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - Both kernels write `g_raw`, `gate`, `beta` and `out_gate` byte for byte as the unfused
//!   launches do (`kernels/gb10/common/kda_gates_w8a8.cu`); the W8A8 activation scratch is not
//!   touched, so the dense registry's record of what it holds stays true.
//! - They run only when the low-rank group put `f_a(h)` and `g_a(h)` in separate buffers (`low`),
//!   with f_b and g_b on the per-row FP8 tier at `[heads * head_dim, 128]`. With the one-row
//!   lever on at one row, anything else is an error rather than a quiet unfused run; the rows
//!   kernel, on by default, declines (the unfused chain runs) where any of that does not hold.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::weight_map::Fp8Weight;

use super::{Glm5NextKdaLayer, Glm5NextKdaWorkspace};

/// 2026-10-10: The contract width the fused kernel is compiled for (the K <= 128 wxay entry).
const GATES_K: usize = 128;
/// 2026-10-10: Output rows per warp tile and warps per block of `kda_gates_w8a8_k128`.
const TILE_ROWS: usize = 16;
const WARPS: usize = 8;
/// 2026-10-21: `kdag::MAX_ROWS` in `kda_gates_w8a8.cu`: the token columns of one MMA tile.
pub const GATES_MAX_ROWS: usize = 8;

type GateWeights = (Option<Fp8Weight>, Option<Fp8Weight>);

impl Glm5NextKdaLayer {
    /// 2026-10-10: Run the fused chain when a lever asks for it at this row count; `Ok(false)`
    /// leaves `front_end` to launch the unfused chain.
    pub(super) fn gates_fused(
        &self,
        gpu: &dyn GpuBackend,
        t: usize,
        low: bool,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<bool> {
        let weights = || {
            (
                crate::glm5next_fp8_dense::row_weight(self.weights.f_b.weight),
                crate::glm5next_fp8_dense::row_weight(self.weights.g_b.weight),
            )
        };
        if self.levers.kda.gates_fused && t == 1 {
            self.launch_gates_fused(gpu, low, weights(), ws, stream)?;
            return Ok(true);
        }
        self.try_gates_rows(gpu, t, low, weights(), ws, stream)
    }

    /// 2026-10-21: The rows kernel at `t` rows when its lever is on, `t` is in range and nothing
    /// refuses it; `Ok(false)` (nothing launched) otherwise.
    pub(super) fn try_gates_rows(
        &self,
        gpu: &dyn GpuBackend,
        t: usize,
        low: bool,
        w: GateWeights,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<bool> {
        let kernel = self.kernels.gates_rows;
        if !self.levers.kda.gates_rows
            || !(1..=GATES_MAX_ROWS).contains(&t)
            || self.gates_refusal(kernel, low, w).is_some()
        {
            return Ok(false);
        }
        self.launch_gates(gpu, kernel, Some(t), w, ws, stream)?;
        Ok(true)
    }

    /// 2026-10-10: The fused launch for one row, given f_b's and g_b's FP8 copies.
    pub(super) fn launch_gates_fused(
        &self,
        gpu: &dyn GpuBackend,
        low: bool,
        w: GateWeights,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let kernel = self.kernels.gates_fused;
        if let Some(why) = self.gates_refusal(kernel, low, w) {
            bail!("METRALE_GLM_KDA_GATES_FUSED=1: {why}");
        }
        self.launch_gates(gpu, kernel, None, w, ws, stream)
    }

    /// 2026-10-21: Why `kernel` cannot run the chain here, `None` when it can.
    fn gates_refusal(&self, kernel: KernelHandle, low: bool, w: GateWeights) -> Option<String> {
        let c = &self.cfg;
        let (qkv, hd) = (c.qkv_dim(), c.head_dim);
        if kernel.0 == 0 {
            return Some("the kernel is not in this target".into());
        }
        if !low {
            return Some(
                "needs f_a and g_a in one low-rank group launch (the W4A16 seg tier)".into(),
            );
        }
        if hd != GATES_K || !qkv.is_multiple_of(TILE_ROWS) {
            return Some(format!(
                "head_dim {hd} / width {qkv}; the kernel takes K = {GATES_K} and a width in whole \
                 {TILE_ROWS}-row tiles"
            ));
        }
        let (Some(f_b), Some(g_b)) = w else {
            return Some(
                "f_b and g_b must be on the per-row FP8 tier (--dense-quantization w4a16 or fp8)"
                    .into(),
            );
        };
        [(&f_b, "f_b"), (&g_b, "g_b")]
            .into_iter()
            .find(|(w, _)| (w.n as usize, w.k as usize) != (qkv, hd))
            .map(|(w, what)| format!("{what} is [{}, {}], want [{qkv}, {hd}]", w.n, w.k))
    }

    /// 2026-10-21: One launch of `kernel` (the one-token entry when `rows` is `None`, else the
    /// rows entry with that M), after [`Self::gates_refusal`] passed.
    fn launch_gates(
        &self,
        gpu: &dyn GpuBackend,
        kernel: KernelHandle,
        rows: Option<usize>,
        w: GateWeights,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (qkv, hd) = (c.qkv_dim(), c.head_dim);
        let (Some(f_b), Some(g_b)) = w else {
            bail!("KDA gates: launched without f_b / g_b FP8 copies");
        };
        let blocks = (2 * qkv / TILE_ROWS).div_ceil(WARPS);
        let mut l = KernelLaunch::new(gpu, kernel)
            .grid([blocks as u32, 1, 1])
            .block([(WARPS * 32) as u32, 1, 1])
            .arg_ptr(ws.lowrank)
            .arg_ptr(ws.lowrank_g)
            .arg_ptr(f_b.weight)
            .arg_ptr(f_b.row_scale)
            .arg_ptr(g_b.weight)
            .arg_ptr(g_b.row_scale)
            .arg_ptr(ws.g_raw)
            .arg_ptr(ws.out_gate)
            .arg_ptr(self.weights.dt_bias)
            .arg_ptr(self.weights.a_log)
            .arg_ptr(ws.gate)
            .arg_ptr(ws.beta_bf16)
            .arg_ptr(ws.beta);
        if let Some(m) = rows {
            l = l.arg_u32(m as u32);
        }
        l.arg_u32(qkv as u32)
            .arg_u32(hd as u32)
            .arg_u32(c.heads as u32)
            .arg_f32(c.gate_lower_bound)
            .launch(stream)
    }
}

#[cfg(test)]
#[path = "gates_fused_tests.rs"]
mod tests;
