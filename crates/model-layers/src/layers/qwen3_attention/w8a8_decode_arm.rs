// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The declared-W8A8 arm of the attention decode projections (2026-10-01: and, for
//! O, the fixed `nvfp4` arm, `pinned_o`): Q|K|V (three
//! stacked segments into the `[rows, per_seq_qkv]` QKV buffer) and O, on the W8A8 decode
//! family (`crate::layers::W8a8Mixer`). The single-token `attention_forward` and the
//! multi-sequence / verify `ms_phase_qkv` and `ms_phase_o_proj` try it first.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants:
//! - `Ok(false)` launches nothing: no W8A8 weights, an MLA layer, a LoRA overlay (Q|K|V
//!   only: a q delta must fold before the gated deinterleave), or a row count outside the
//!   family.
//! - On a gated layer the Q|K|V arm leaves Q deinterleaved to `[Q_all | Gate_all]`, as every
//!   other projection arm without a q adapter does.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::DevicePtr;

use super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops::W8a8Weight;
use crate::layers::{FfnComponent, W8a8Ctx, W8a8Ffn, W8a8Mixer, ops};
use crate::weight_map::WeightQuantFormat;

impl Qwen3AttentionLayer {
    /// 2026-09-28: Install the W8A8 Q|K|V and O. Refuses shapes other than
    /// `[q_proj + 2 kv, hidden]` and `[hidden, q_dim]`, and an MLA layer.
    pub fn set_w8a8_decode_weights(
        &mut self,
        w: W8a8Mixer,
        qkv_rows: u32,
        hidden: u32,
        q_dim: u32,
    ) -> Result<()> {
        ensure!(self.mla.is_none(), "W8A8 attention arm has no MLA path");
        ensure!(
            (w.input.n(), w.input.k(), w.output.n(), w.output.k())
                == (qkv_rows, hidden, hidden, q_dim),
            "W8A8 attention shapes [{}, {}] / [{}, {}], want [{qkv_rows}, {hidden}] / [{hidden}, {q_dim}]",
            w.input.n(),
            w.input.k(),
            w.output.n(),
            w.output.k()
        );
        self.w8a8 = Some(w);
        Ok(())
    }

    /// 2026-09-28: Run this layer's block-scaled FP8 Q, K, V and O (a native FP8 checkpoint,
    /// HF `fp8` with 128x128 weight blocks) W8A8 from now on, with per-(token, 128) activation
    /// scales. `Ok(false)` installs nothing: some projection is not block-scaled FP8.
    pub fn adopt_fp8_block_w8a8(&mut self, ctx: W8a8Ctx, hidden: u32) -> Result<bool> {
        let fp8 = |w: &Option<crate::weight_map::QuantWeight>| {
            w.as_ref()
                .and_then(|w| w.as_fp8())
                .filter(|f| f.scale_format == WeightQuantFormat::Fp8BlockScaled)
                .copied()
        };
        let (Some(q), Some(k), Some(v), Some(o)) = (
            fp8(&self.q_weight),
            fp8(&self.k_weight),
            fp8(&self.v_weight),
            fp8(&self.o_weight),
        ) else {
            return Ok(false);
        };
        let (input, output) = (W8a8Weight::new(&[q, k, v])?, W8a8Weight::new(&[o])?);
        let rows = input.n();
        self.set_w8a8_decode_weights(W8a8Mixer { ctx, input, output }, rows, hidden, o.k)?;
        Ok(true)
    }

    /// 2026-09-28: Install W8A8 gate/up/down on this layer's dense FFN
    /// (`DenseFfnLayer::set_w8a8_decode_weights`); refuses a MoE or absent FFN.
    pub fn set_w8a8_ffn_weights(
        &mut self,
        w: W8a8Ffn,
        prefill: crate::layers::W8a8Ctx,
        hidden: u32,
        inter: u32,
    ) -> Result<()> {
        match self.ffn {
            FfnComponent::Dense(ref mut d) => {
                d.set_w8a8_decode_weights(w, hidden, inter)?;
                d.set_w8a8_prefill_ctx(prefill);
                Ok(())
            }
            _ => anyhow::bail!("W8A8 FFN weights need a dense FFN"),
        }
    }

    /// 2026-09-28: Whether [`Self::w8a8_qkv`] runs at `rows` rows (a pure function of the
    /// layer and `rows`, so the single-token Q and K/V steps agree on it).
    pub(super) fn w8a8_qkv_serves(&self, rows: usize) -> bool {
        self.lora.is_none()
            && rows > 0
            && self
                .w8a8
                .is_some_and(|w| w.ctx.available(&w.input, rows.min(ops::W8A8_MAX_ROWS)))
    }

    /// 2026-09-28: The single-token Q|K|V into the contiguous `[Q | K | V]` at `qkv`.
    pub(super) fn w8a8_qkv_m1(
        &self,
        ctx: &ForwardContext,
        normed: DevicePtr,
        qkv: DevicePtr,
        nq: u32,
        hd: u32,
        stream: u64,
    ) -> Result<bool> {
        let ldc = self.w8a8.map_or(0, |w| w.input.n());
        self.w8a8_qkv(ctx, normed, 1, qkv, ldc, nq, hd, stream)
    }

    /// 2026-09-28: Q|K|V of `normed[rows, hidden]` into `qkv[rows, ldc]` (BF16 elements),
    /// then the gated deinterleave of each row's Q. `Ok(false)` as the module header says.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn w8a8_qkv(
        &self,
        ctx: &ForwardContext,
        normed: DevicePtr,
        rows: usize,
        qkv: DevicePtr,
        ldc: u32,
        nq: u32,
        hd: u32,
        stream: u64,
    ) -> Result<bool> {
        let Some(ref w) = self.w8a8 else {
            return Ok(false);
        };
        if !self.w8a8_qkv_serves(rows) {
            return Ok(false);
        }
        w.ctx.proj_rows(
            ctx.gpu,
            &w.input,
            normed,
            w.input.k(),
            rows,
            qkv,
            ldc,
            stream,
        )?;
        if self.gated {
            ops::deinterleave_qg(
                ctx.gpu,
                self.deinterleave_qg_k,
                qkv,
                rows as u32,
                nq,
                hd,
                ldc,
                stream,
            )?;
        }
        Ok(true)
    }

    /// 2026-10-03: Whether O is the NVFP4 `attn.o_proj`, the weight every NVFP4 O arm reads. An
    /// NVFP4 layer leaves `o_weight` unset (`init.rs`); it holds O only for the FP8 and packed Q2
    /// formats, which take their own arms.
    pub(super) fn o_is_nvfp4(&self) -> bool {
        self.mla.is_none()
            && self.o_dense_bf16.is_none()
            && self
                .o_weight
                .as_ref()
                .is_none_or(|w| w.as_nvfp4().is_some())
            && !self.attn.o_proj.is_null()
    }

    /// 2026-09-28: O of `attn_out[rows, q_dim]` into `out[rows, hidden]`. The caller applies
    /// the o_proj adapter delta afterwards, as for its other arms. 2026-10-01: Under a fixed
    /// `nvfp4` attention format, the NVFP4 O on the row-invariant W4A4 mx path instead
    /// (`hidden` and `q_dim` are its shape).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn pinned_o(
        &self,
        ctx: &ForwardContext,
        attn_out: DevicePtr,
        rows: usize,
        out: DevicePtr,
        hidden: u32,
        q_dim: u32,
        stream: u64,
    ) -> Result<bool> {
        if self.o_is_nvfp4()
            && ops::w4a4_proj::fixed_nvfp4_proj(
                ctx.gpu,
                metrale_config::ProjFamily::Attn,
                &self.attn.o_proj,
                attn_out,
                out,
                rows as u32,
                hidden,
                q_dim,
                stream,
            )?
        {
            return Ok(true);
        }
        match self.w8a8 {
            Some(ref w) => w.ctx.proj_rows(
                ctx.gpu,
                &w.output,
                attn_out,
                w.output.k(),
                rows,
                out,
                w.output.n(),
                stream,
            ),
            None => Ok(false),
        }
    }

    /// 2026-10-05: Install the prefill's W8A8 context: the same kernels as the decode arms with
    /// an activation scratch of its own. Without it the prefill arms below launch nothing.
    pub fn set_w8a8_prefill_ctx(&mut self, ctx: crate::layers::W8a8Ctx) {
        self.w8a8_prefill = Some(ctx);
    }

    /// 2026-10-05: The W8A8 O for a prefill of `rows` rows of `attn_out[rows, q_dim]` into
    /// `out[rows, hidden]`, on the prefill's own scratch, or `Ok(false)` launching nothing.
    pub(super) fn w8a8_prefill_o(
        &self,
        ctx: &ForwardContext,
        attn_out: DevicePtr,
        rows: usize,
        out: DevicePtr,
        stream: u64,
    ) -> Result<bool> {
        let (Some(w), Some(pc)) = (self.w8a8, self.w8a8_prefill) else {
            return Ok(false);
        };
        pc.proj_rows(
            ctx.gpu,
            &w.output,
            attn_out,
            w.output.k(),
            rows,
            out,
            w.output.n(),
            stream,
        )
    }

    /// 2026-10-05: Segment `seg` (0 Q, 1 K, 2 V) of the W8A8 Q|K|V for a prefill of `rows` rows of
    /// `normed[rows, hidden]` into `out[rows, n_seg]`, on the prefill's own scratch: the prefill
    /// projections at the declared FP8, instead of an NVFP4 copy of the FP8 weight. The caller
    /// applies any adapter delta and the gated deinterleave afterwards, as for its other arms.
    /// `Ok(false)` launches nothing.
    pub(super) fn w8a8_prefill_qkv_segment(
        &self,
        ctx: &ForwardContext,
        seg: usize,
        normed: DevicePtr,
        rows: usize,
        out: DevicePtr,
        stream: u64,
    ) -> Result<bool> {
        let (Some(w), Some(pc)) = (self.w8a8, self.w8a8_prefill) else {
            return Ok(false);
        };
        let s = w.input.segment(seg)?;
        pc.proj_rows(ctx.gpu, &s, normed, s.k(), rows, out, s.n(), stream)
    }
}

#[cfg(test)]
#[path = "w8a8_decode_arm_tests.rs"]
mod tests;
