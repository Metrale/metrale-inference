// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Quantized copies of the drafter GEMMs that the `fp8` tier of
//! `--draft-weight-quantization` keeps BF16 (the ctx `fc` and fused K/V projections, the DFlash2
//! conv kernel projections and the selector's hidden projection), made on the `fp8-aux` (FP8)
//! and `nvfp4` (NVFP4) tiers, and the FP8 row-GEMM dispatch every drafter FP8 GEMM shares
//! ([`BlockDiffusionDraftHead::fp8_rows`]).
//!
//! Those BF16 GEMMs run on the 128 x 128 prefill tile, whose grid is N / 128 blocks: at a
//! drafter block of <= 8 rows the conv projections ([1024, 4096]) launched 8 blocks and took
//! ~160 us each, ten per propose, the selector projection ([256, 4096]) 2 blocks and ~165 us,
//! the ctx `fc` ([4096, 20480]) ~1.2 ms and the fused K/V ([10240, 4096]) ~0.55 ms (GB10 nsys,
//! 2026-10-09). On the row GEMVs they read half the bytes (FP8) or less across every SM.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - Only the drafter's draft numerics move; the target model is untouched, so a greedy verify
//!   still emits the target's tokens.
//! - On the tiers without them, nothing is quantized and every launch is unchanged.
//! - The copies are in the layer copies' format: the head is `Fp8Weights` or `Nvfp4Weights`
//!   when they are built, or the build fails.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::{DenseWeight, Fp8DenseWeight};

use super::BlockDiffusionDraftHead;
use super::draft_weight::{DraftQuantizer, DraftWeight};
use super::tp::FullWeight;

/// 2026-10-10: The auxiliary GEMMs' copies.
pub struct DflashAuxQuant {
    /// 2026-10-10: `fc`, `[hidden, target_layers * target_hidden]`.
    pub fc: DraftWeight,
    /// 2026-10-10: The fused ctx K/V weight, `[layers * 2 * kv_dim, hidden]`.
    pub fused_kv: DraftWeight,
    /// 2026-10-10: The selector's hidden projection, `[selector_rank, hidden]`, when the
    /// checkpoint has a selector.
    pub selector_hidden_proj: Option<DraftWeight>,
}

impl BlockDiffusionDraftHead {
    /// 2026-10-10: `dst [g, n] = src [g, k] @ W^T` on an FP8 drafter weight: the 8-row rt2 GEMV
    /// up to 8 rows, the 16-row one up to 16 (each only when it resolved, K is a multiple of
    /// 16 and `METRALE_NO_DFLASH_FP8_RT` is not `1`), else the row-scaled FP8 GEMM. The paged
    /// layer GEMMs and the [`DflashAuxQuant`] FP8 projections dispatch here (through
    /// `quant_rows`); the lm_head keeps its own order (`forward_block/tail.rs`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fp8_rows(
        &self,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        w: &Fp8DenseWeight,
        dst: DevicePtr,
        g: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        let rt = k.is_multiple_of(16) && self.levers.fp8_rt;
        if rt && self.kernels.fp8_gemv_rt2.0 != 0 && g <= 8 {
            return ops::fp8_gemv_rowscale_batch8_rt2(
                gpu,
                self.kernels.fp8_gemv_rt2,
                src,
                w,
                dst,
                g,
                n,
                k,
                stream,
            );
        }
        if rt && self.kernels.fp8_gemv_rt2_16.0 != 0 && g <= 16 {
            return ops::fp8_gemv_rowscale_batch16_rt2(
                gpu,
                self.kernels.fp8_gemv_rt2_16,
                src,
                w,
                dst,
                g,
                n,
                k,
                stream,
            );
        }
        ops::fp8_gemm_n128_row_scaled(
            gpu,
            self.kernels.fp8_gemm_n128_row_scaled,
            src,
            w,
            dst,
            g,
            n,
            k,
            stream,
        )
    }

    /// 2026-10-10: `dst [g, n] = src [g, k] @ W^T` for one of the GEMMs [`DflashAuxQuant`]
    /// covers: on its copy when there is one, else the BF16 prefill GEMM as before.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn aux_gemm(
        &self,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        bf16: &DenseWeight,
        q: Option<&DraftWeight>,
        dst: DevicePtr,
        g: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        match q {
            Some(w) => self.quant_rows(gpu, src, w, dst, g, n, k, stream),
            None => ops::dense_gemm_bf16_pipelined(
                gpu,
                self.kernels.dense_gemm_pipelined,
                src,
                bf16,
                dst,
                g,
                n,
                k,
                stream,
            ),
        }
    }

    /// 2026-10-10: Build the [`DflashAuxQuant`] copies from the head's BF16 weights with `qz`,
    /// and each layer's conv projection copies (`DflashLayer::{attention,mlp}_conv_proj_q`).
    /// The fused K/V copy's NVFP4 scale spans every layer's whole K and V (`tp_src`) under the
    /// tensor-parallel drafter. Errors unless the head's layer copies are in `qz`'s format and
    /// it has its fused K/V weight.
    pub(super) fn quantize_aux(&mut self, qz: &DraftQuantizer<'_>) -> Result<DflashAuxQuant> {
        let want = if qz.nvfp4 {
            super::DflashQuantization::Nvfp4Weights
        } else {
            super::DflashQuantization::Fp8Weights
        };
        ensure!(
            self.quant == want,
            "DFlash auxiliary GEMM copies ({want:?}) need the layer copies in the same format; \
             the head is {:?}",
            self.quant
        );
        let Some(fused) = self.fused_kv_weight else {
            anyhow::bail!("DFlash auxiliary GEMM copies: the drafter has no fused K/V weight");
        };
        let h = self.hidden_size;
        let kv_rows = self.num_layers * 2 * self.num_kv_heads * self.head_dim;
        let target_dim = self.target_layer_ids.len() * self.target_hidden_size;
        let conv_rows = 2 * self.conv_kernel_size * (h / self.conv_group_size.max(1));
        let kv_whole: Vec<FullWeight> = self
            .layers
            .iter()
            .filter_map(|l| l.tp_src)
            .flat_map(|s| [s.k, s.v])
            .collect();
        let q = |w: &DenseWeight, n: usize, k: usize| qz.copy(w, n, k, &[], None);
        for l in self.layers.iter_mut() {
            l.attention_conv_proj_q = l
                .attention_conv_proj
                .as_ref()
                .map(|w| q(w, conv_rows, h))
                .transpose()?;
            l.mlp_conv_proj_q = l
                .mlp_conv_proj
                .as_ref()
                .map(|w| q(w, conv_rows, h))
                .transpose()?;
        }
        Ok(DflashAuxQuant {
            fc: q(&self.fc, h, target_dim)?,
            fused_kv: qz.copy(&DenseWeight { weight: fused }, kv_rows, h, &kv_whole, None)?,
            selector_hidden_proj: self
                .selector_hidden_proj
                .as_ref()
                .map(|w| q(w, self.selector_rank, h))
                .transpose()?,
        })
    }
}
