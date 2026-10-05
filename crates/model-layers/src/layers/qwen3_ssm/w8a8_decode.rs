// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The declared-W8A8 arm of the GDN decode projections (2026-10-01: and the fixed
//! `nvfp4` arm, `pinned_qkvz` / `pinned_out`): `in_proj_qkv|z` (two
//! stacked segments, the sequential `[QKV | Z]` layout) and `out_proj`, on the W8A8 decode
//! family (`crate::layers::W8a8Mixer`). Every decode path tries it first: the single-token
//! `ssm_forward`, the batched verify (`decode_batched`) and the multi-sequence batch
//! (`ssm_batched_proj`).
//!
//! Owner: model-layers (qwen3 SSM).
//! Invariants:
//! - Installed only on a sequential-QKVZ layer at TP 1 (`set_w8a8_decode_weights`), so the
//!   stacked output is the layout every reader of the QKVZ buffer expects and no TP
//!   all-reduce is skipped.
//! - `Ok(false)` launches nothing: no W8A8 weights, or the row count is outside the family.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::DevicePtr;

use super::Qwen3SsmLayer;
use crate::layer::ForwardContext;
use crate::layers::ops::W8a8Weight;
use crate::layers::{FfnComponent, W8a8Ctx, W8a8Ffn, W8a8Mixer};
use crate::weight_map::WeightQuantFormat;

impl Qwen3SsmLayer {
    /// 2026-09-28: Install the W8A8 QKV|Z and out_proj. Refuses an interleaved-QKVZ layer or
    /// shapes that are not `[qkvz, hidden]` and `[hidden, value_dim]`.
    pub fn set_w8a8_decode_weights(
        &mut self,
        w: W8a8Mixer,
        qkvz_size: u32,
        hidden: u32,
        value_dim: u32,
    ) -> Result<()> {
        ensure!(
            self.sequential_qkvz,
            "W8A8 QKV|Z needs the sequential [QKV | Z] layout"
        );
        ensure!(
            (w.input.n(), w.input.k(), w.output.n(), w.output.k())
                == (qkvz_size, hidden, hidden, value_dim),
            "W8A8 GDN shapes [{}, {}] / [{}, {}], want [{qkvz_size}, {hidden}] / [{hidden}, {value_dim}]",
            w.input.n(),
            w.input.k(),
            w.output.n(),
            w.output.k()
        );
        self.w8a8 = Some(w);
        Ok(())
    }

    /// 2026-09-28: Install W8A8 gate/up/down on this layer's dense FFN
    /// (`DenseFfnLayer::set_w8a8_decode_weights`); refuses a MoE or absent FFN.
    pub fn set_w8a8_ffn_weights(&mut self, w: W8a8Ffn, hidden: u32, inter: u32) -> Result<()> {
        match self.ffn {
            FfnComponent::Dense(ref mut d) => d.set_w8a8_decode_weights(w, hidden, inter),
            _ => anyhow::bail!("W8A8 FFN weights need a dense FFN"),
        }
    }

    /// 2026-09-28: Run this layer's block-scaled FP8 `[QKV | Z]` and out_proj (the decode FP8
    /// weights of a native FP8 checkpoint, `set_fp8_decode_weights`) W8A8 from now on, with
    /// per-(token, 128) activation scales. `Ok(false)` installs nothing: they are absent or not
    /// block-scaled.
    pub fn adopt_fp8_block_w8a8(&mut self, ctx: W8a8Ctx, hidden: u32) -> Result<bool> {
        let block = |w: Option<crate::weight_map::Fp8Weight>| {
            w.filter(|f| f.scale_format == WeightQuantFormat::Fp8BlockScaled)
        };
        let (Some(qkvz), Some(out)) = (block(self.qkvz_fp8w), block(self.out_proj_fp8w)) else {
            return Ok(false);
        };
        let (input, output) = (W8a8Weight::new(&[qkvz])?, W8a8Weight::new(&[out])?);
        self.set_w8a8_decode_weights(W8a8Mixer { ctx, input, output }, qkvz.n, hidden, out.k)?;
        Ok(true)
    }

    /// 2026-09-28: `out[rows, ldc]` = W8A8 QKV|Z of `normed[rows, ldx]`; `Ok(false)` when
    /// this layer has no W8A8 weights or `rows` is outside the family. 2026-10-01: Under a fixed
    /// `nvfp4` GDN format (`--activation-quantization`), the NVFP4 QKV|Z of a sequential layer
    /// instead, through the row-invariant W4A4 mx path at every row count.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn pinned_qkvz(
        &self,
        ctx: &ForwardContext,
        normed: DevicePtr,
        ldx: u32,
        rows: usize,
        out: DevicePtr,
        ldc: u32,
        stream: u64,
    ) -> Result<bool> {
        if self.sequential_qkvz
            && let Some(ref nvfp4) = self.qkvz_nvfp4
            && crate::layers::ops::w4a4_proj::fixed_nvfp4_proj(
                ctx.gpu,
                metrale_config::ProjFamily::Gdn,
                nvfp4,
                normed,
                out,
                rows as u32,
                ldc,
                ldx,
                stream,
            )?
        {
            return Ok(true);
        }
        match self.w8a8 {
            Some(ref w) => w
                .ctx
                .proj(ctx.gpu, &w.input, normed, ldx, rows, out, ldc, stream),
            None => Ok(false),
        }
    }

    /// 2026-09-28: `out[rows, ldc]` = W8A8 out_proj of `normed_out[rows, ldx]`, as
    /// [`Self::pinned_qkvz`]; under a fixed `nvfp4` GDN format, the NVFP4 out_proj on the W4A4 mx
    /// path.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn pinned_out(
        &self,
        ctx: &ForwardContext,
        normed_out: DevicePtr,
        ldx: u32,
        rows: usize,
        out: DevicePtr,
        ldc: u32,
        stream: u64,
    ) -> Result<bool> {
        let nvfp4 = &self.ssm.out_proj;
        if self.out_proj_fp8w.is_none()
            && self.out_proj_dense.is_none()
            && crate::layers::ops::w4a4_proj::fixed_nvfp4_proj(
                ctx.gpu,
                metrale_config::ProjFamily::Gdn,
                nvfp4,
                normed_out,
                out,
                rows as u32,
                ldc,
                ldx,
                stream,
            )?
        {
            return Ok(true);
        }
        match self.w8a8 {
            Some(ref w) => w
                .ctx
                .proj(ctx.gpu, &w.output, normed_out, ldx, rows, out, ldc, stream),
            None => Ok(false),
        }
    }
}
