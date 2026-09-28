// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The W8A8 decode weights a layer carries when its checkpoint
//! declares FP8 weights and FP8 activations for a projection (compressed-tensors
//! `float-quantized` groups, HF `fp8`), and the shared per-model resources they
//! run with (`ops::w8a8_proj`).
//!
//! A layer holding one of these runs that projection W8A8 at every decode row
//! count in `1..=ops::W8A8_MAX_ROWS`, ahead of its other decode arms; wider
//! launches and prefill keep the layer's other copies. Whether a checkpoint's
//! projection gets one is the loader's decision (the declared-precision
//! policy); these types only say that it can run.
//!
//! Owner: model-layers.
//! Invariants:
//! - Every `W8a8Ctx` of one model shares one scratch; the W8A8 projections run
//!   on the decode stream only, one after another, so the quantize -> GEMV
//!   pairs never interleave.
//! - A row's output bits do not depend on how many rows share the launch
//!   (`ops::w8a8_decode`), so a W8A8 projection is row-invariant under every
//!   row-tier policy.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops::{self, W8a8Kernels, W8a8Scratch, W8a8Weight};

/// 2026-09-28: The kernels and the quantized-activation scratch one model's
/// W8A8 projections share.
#[derive(Clone, Copy, Debug)]
pub struct W8a8Ctx {
    pub kernels: W8a8Kernels,
    pub scratch: W8a8Scratch,
}

impl W8a8Ctx {
    /// 2026-09-28: Look the kernels up and allocate a scratch for a `max_k`-wide
    /// activation of `ops::W8A8_MAX_ROWS` rows.
    pub fn new(gpu: &dyn GpuBackend, max_k: u32) -> Result<Self> {
        Ok(Self {
            kernels: W8a8Kernels::load(gpu),
            scratch: W8a8Scratch::alloc(gpu, max_k)?,
        })
    }

    /// 2026-09-28: Whether `w` can run W8A8 at `rows` rows.
    pub fn available(&self, w: &W8a8Weight, rows: usize) -> bool {
        ops::w8a8_decode_available(&self.kernels, w, rows, &self.scratch)
    }

    /// 2026-09-28: `out[rows, ldc]` = W8A8 of `x[rows, ldx]` (BF16) and `w`;
    /// `Ok(false)`, launching nothing, when not [`Self::available`].
    #[allow(clippy::too_many_arguments)]
    pub fn proj(
        &self,
        gpu: &dyn GpuBackend,
        w: &W8a8Weight,
        x: DevicePtr,
        ldx: u32,
        rows: usize,
        out: DevicePtr,
        ldc: u32,
        stream: u64,
    ) -> Result<bool> {
        if !self.available(w, rows) {
            return Ok(false);
        }
        ops::w8a8_proj(
            gpu,
            &self.kernels,
            w,
            x,
            ldx,
            rows,
            out,
            ldc,
            &self.scratch,
            stream,
        )?;
        Ok(true)
    }

    /// 2026-09-28: The down projection of a SiLU FFN: `out` = W8A8 of
    /// `bf16(silu(gate) * up)` (gate and up `[rows, ld]` BF16) and `w`, with the
    /// SiLU product quantized in the same launch; `Ok(false)` when not available.
    #[allow(clippy::too_many_arguments)]
    pub fn silu_proj(
        &self,
        gpu: &dyn GpuBackend,
        w: &W8a8Weight,
        gate: DevicePtr,
        up: DevicePtr,
        ld: u32,
        rows: usize,
        out: DevicePtr,
        ldc: u32,
        stream: u64,
    ) -> Result<bool> {
        if !self.available(w, rows) {
            return Ok(false);
        }
        ops::w8a8_act_quant_silu(
            gpu,
            &self.kernels,
            w.scale(),
            gate,
            up,
            ld,
            rows,
            w.k(),
            &self.scratch,
            stream,
        )?;
        ops::w8a8_gemv(gpu, &self.kernels, w, &self.scratch, rows, out, ldc, stream)?;
        Ok(true)
    }
}

/// 2026-09-28: The input and output projections of an attention layer
/// (Q|K|V stacked, O) or a GDN layer (QKV|Z stacked, out_proj).
#[derive(Clone, Copy, Debug)]
pub struct W8a8Mixer {
    pub ctx: W8a8Ctx,
    pub input: W8a8Weight,
    pub output: W8a8Weight,
}

/// 2026-09-28: A SiLU dense FFN's gate, up and down projections.
#[derive(Clone, Copy, Debug)]
pub struct W8a8Ffn {
    pub ctx: W8a8Ctx,
    pub gate: W8a8Weight,
    pub up: W8a8Weight,
    pub down: W8a8Weight,
}
