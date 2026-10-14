// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: One token's KDA gate chain in one launch (`METRALE_GLM_KDA_GATES_FUSED=1`):
//! `kda_gates_w8a8_k128` runs the f_b and g_b W8A8 projections of the low-rank rows, their
//! activation quantization, `kda_gate` and the beta sigmoid, where `front_end` otherwise launches
//! act-quant + f_b + gate + sigmoid + act-quant + g_b.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - The fused kernel writes `g_raw`, `gate`, `beta` and `out_gate` byte for byte as the unfused
//!   launches do (`kernels/gb10/common/kda_gates_w8a8.cu`); the W8A8 activation scratch is not
//!   touched, so the dense registry's record of what it holds stays true.
//! - Only when the low-rank group put `f_a(h)` and `g_a(h)` in separate buffers (`low`), with f_b
//!   and g_b on the per-row FP8 tier at `[heads * head_dim, 128]`. With the lever on at one row,
//!   anything else is an error rather than a quiet unfused run.
//! - 2026-10-14: Also 2..=`W8A8_MAX_ROWS` rows (an MTP verify group), where the unfused chain's
//!   projections run the W8A8 GEMV (`glm5next_fp8_dense::proj`); a width the dequantize + cuBLASLt
//!   path takes (`METRALE_GLM_FP8_DEQUANT_GEMM_MIN_ROWS`) keeps the unfused chain. Above one row,
//!   a front end without the low-rank group keeps it too: `g_a(h)` then overwrites `f_a(h)` in
//!   `lowrank` after f_b has read it, an order one launch cannot keep.

use anyhow::{Result, bail, ensure};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layers::ops::W8A8_MAX_ROWS;
use metrale_model_layers::weight_map::Fp8Weight;

use super::{Glm5NextKdaLayer, Glm5NextKdaWorkspace};

/// 2026-10-10: `METRALE_GLM_KDA_GATES_FUSED=1`. Off until measured end to end. Read once.
fn kda_gates_fused() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("METRALE_GLM_KDA_GATES_FUSED").as_deref() == Ok("1"))
}

/// 2026-10-10: The contract width the fused kernel is compiled for (the K <= 128 wxay entry).
const GATES_K: usize = 128;
/// 2026-10-10: Output rows per warp tile and warps per block of `kda_gates_w8a8_k128`.
const TILE_ROWS: usize = 16;
const WARPS: usize = 8;
/// 2026-10-14: Tokens per token tile (`blockIdx.y`), the MMA's n8 side.
const TOKEN_TILE: usize = 8;

/// 2026-10-14: Whether `t` rows of the unfused chain run its f_b / g_b projections on the W8A8
/// GEMV, given the dequantize + cuBLASLt path's row floor `wide_min`: only those widths fuse.
fn fused_rows(t: usize, wide_min: Option<usize>) -> bool {
    (1..=W8A8_MAX_ROWS).contains(&t) && !crate::glm5next_fp8_wide::takes(wide_min, t)
}

impl Glm5NextKdaLayer {
    /// 2026-10-10: Run the fused chain when the lever asks for it at this row count; `Ok(false)`
    /// leaves `front_end` to launch the unfused chain.
    pub(super) fn gates_fused(
        &self,
        gpu: &dyn GpuBackend,
        t: usize,
        low: bool,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<bool> {
        if !kda_gates_fused()
            || !fused_rows(t, crate::glm5next_fp8_wide::min_rows()?)
            || (t > 1 && !low)
        {
            return Ok(false);
        }
        let weights = (
            crate::glm5next_fp8_dense::row_weight(self.weights.f_b.weight),
            crate::glm5next_fp8_dense::row_weight(self.weights.g_b.weight),
        );
        self.launch_gates_fused(gpu, t, low, weights, ws, stream)?;
        Ok(true)
    }

    /// 2026-10-10: The fused launch for `t` rows (2026-10-14: was one), given f_b's and g_b's FP8
    /// copies.
    pub(super) fn launch_gates_fused(
        &self,
        gpu: &dyn GpuBackend,
        t: usize,
        low: bool,
        (f_b, g_b): (Option<Fp8Weight>, Option<Fp8Weight>),
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (qkv, hd) = (c.qkv_dim(), c.head_dim);
        ensure!(
            self.kernels.gates_fused.0 != 0,
            "METRALE_GLM_KDA_GATES_FUSED=1: kda_gates_w8a8_k128 is not in this target"
        );
        ensure!(
            low,
            "METRALE_GLM_KDA_GATES_FUSED=1: needs f_a and g_a in one low-rank group launch \
             (the W4A16 seg tier at one row)"
        );
        ensure!(
            hd == GATES_K && qkv.is_multiple_of(TILE_ROWS),
            "METRALE_GLM_KDA_GATES_FUSED=1: head_dim {hd} / width {qkv}; the kernel takes K = \
             {GATES_K} and a width in whole {TILE_ROWS}-row tiles"
        );
        let (Some(f_b), Some(g_b)) = (f_b, g_b) else {
            bail!(
                "METRALE_GLM_KDA_GATES_FUSED=1: f_b and g_b must be on the per-row FP8 tier \
                 (--dense-quantization w4a16 or fp8)"
            );
        };
        for (w, what) in [(&f_b, "f_b"), (&g_b, "g_b")] {
            ensure!(
                (w.n as usize, w.k as usize) == (qkv, hd),
                "METRALE_GLM_KDA_GATES_FUSED=1: {what} is [{}, {}], want [{qkv}, {hd}]",
                w.n,
                w.k
            );
        }
        // 2026-10-14: Once per process, at the first launch: the serve log's proof that the
        // lever reached the fused kernel, not only that the variable was set.
        static DISPATCHED: std::sync::Once = std::sync::Once::new();
        DISPATCHED.call_once(|| {
            tracing::warn!(
                "METRALE_GLM_KDA_GATES_FUSED: fused KDA gates kernel dispatched (rank-local, \
                 first call) at {t} rows"
            );
        });
        let blocks = (2 * qkv / TILE_ROWS).div_ceil(WARPS);
        KernelLaunch::new(gpu, self.kernels.gates_fused)
            .grid([blocks as u32, t.div_ceil(TOKEN_TILE) as u32, 1])
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
            .arg_ptr(ws.beta)
            .arg_u32(t as u32)
            .arg_u32(qkv as u32)
            .arg_u32(hd as u32)
            .arg_u32(c.heads as u32)
            .arg_f32(c.gate_lower_bound)
            .launch(stream)
    }
}

#[cfg(test)]
#[path = "gates_fused_tests.rs"]
mod tests;
