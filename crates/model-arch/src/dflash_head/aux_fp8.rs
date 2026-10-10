// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `METRALE_DFLASH_AUX_FP8=1`: FP8 copies of the drafter GEMMs that stay BF16 under
//! `Fp8Weights` (the ctx `fc` and fused K/V projections, the DFlash2 conv kernel projections
//! and the selector's hidden projection), and the FP8 row-GEMM dispatch every drafter FP8 GEMM
//! shares ([`BlockDiffusionDraftHead::fp8_rows`]).
//!
//! Those BF16 GEMMs run on the 128 x 128 prefill tile, whose grid is N / 128 blocks: at a
//! drafter block of <= 8 rows the conv projections ([1024, 4096]) launched 8 blocks and took
//! ~160 us each, ten per propose, the selector projection ([256, 4096]) 2 blocks and ~165 us,
//! the ctx `fc` ([4096, 20480]) ~1.2 ms and the fused K/V ([10240, 4096]) ~0.55 ms (GB10 nsys,
//! 2026-10-09). On the FP8 row GEMVs they read half the bytes across every SM.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - Only the drafter's draft numerics move (FP8 weights with one f32 scale per row, as the
//!   layer GEMMs already use); the target model is untouched, so a greedy verify still emits
//!   the target's tokens.
//! - Off (`DFlashLevers::aux_fp8`, unset or any value but `1`), nothing is quantized and every
//!   launch is unchanged.
//! - The copies exist only with `Fp8Weights`; with the lever on and the head left BF16
//!   (`METRALE_DFLASH_DRAFTER_FP8=0`) the build fails rather than run without them.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::{DenseWeight, Fp8DenseWeight};

use super::BlockDiffusionDraftHead;

/// 2026-10-10: The FP8 copies `METRALE_DFLASH_AUX_FP8=1` adds to an `Fp8Weights` head.
pub struct DflashAuxFp8 {
    /// 2026-10-10: `fc`, `[hidden, target_layers * target_hidden]`.
    pub fc: Fp8DenseWeight,
    /// 2026-10-10: The fused ctx K/V weight, `[layers * 2 * kv_dim, hidden]`.
    pub fused_kv: Fp8DenseWeight,
    /// 2026-10-10: The selector's hidden projection, `[selector_rank, hidden]`, when the
    /// checkpoint has a selector.
    pub selector_hidden_proj: Option<Fp8DenseWeight>,
}

impl BlockDiffusionDraftHead {
    /// 2026-10-10: `dst [g, n] = src [g, k] @ W^T` on an FP8 drafter weight: the 8-row rt2 GEMV
    /// up to 8 rows, the 16-row one up to 16 (each only when it resolved, K is a multiple of
    /// 16 and `METRALE_NO_DFLASH_FP8_RT` is not `1`), else the row-scaled FP8 GEMM. The paged
    /// layer GEMMs (`forward_block_layer_paged`) and the [`DflashAuxFp8`] projections dispatch
    /// here; the lm_head keeps its own order (`forward_block/tail.rs`).
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
        let rt = k.is_multiple_of(16) && super::fp8_rt_enabled();
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

    /// 2026-10-10: `dst [g, n] = src [g, k] @ W^T` for one of the GEMMs [`DflashAuxFp8`]
    /// covers: on its FP8 copy when there is one, else the BF16 prefill GEMM as before.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn aux_gemm(
        &self,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        bf16: &DenseWeight,
        fp8: Option<&Fp8DenseWeight>,
        dst: DevicePtr,
        g: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        match fp8 {
            Some(w) => self.fp8_rows(gpu, src, w, dst, g, n, k, stream),
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

    /// 2026-10-10: Build the [`DflashAuxFp8`] copies from the head's BF16 weights with the
    /// BF16-to-FP8 quantizer `quant_k`, and each layer's conv projection copies
    /// (`DflashLayer::{attention,mlp}_conv_proj_fp8`). Errors unless the head is `Fp8Weights`
    /// and has its fused K/V weight.
    pub(super) fn quantize_aux_fp8(
        &mut self,
        gpu: &dyn GpuBackend,
        quant_k: metrale_gpu_runtime::gpu::KernelHandle,
        stream: u64,
    ) -> Result<DflashAuxFp8> {
        ensure!(
            matches!(self.quant, super::DflashQuantization::Fp8Weights),
            "METRALE_DFLASH_AUX_FP8=1 needs the FP8 drafter (METRALE_DFLASH_DRAFTER_FP8 not 0)"
        );
        let Some(fused) = self.fused_kv_weight else {
            anyhow::bail!("METRALE_DFLASH_AUX_FP8=1: the drafter has no fused K/V weight");
        };
        let h = self.hidden_size;
        let kv_rows = self.num_layers * 2 * self.num_kv_heads * self.head_dim;
        let target_dim = self.target_layer_ids.len() * self.target_hidden_size;
        let conv_rows = 2 * self.conv_kernel_size * (h / self.conv_group_size.max(1));
        let q = |w: &DenseWeight, n: usize, k: usize| w.quantize_to_fp8(gpu, quant_k, n, k, stream);
        for l in self.layers.iter_mut() {
            l.attention_conv_proj_fp8 = l
                .attention_conv_proj
                .as_ref()
                .map(|w| q(w, conv_rows, h))
                .transpose()?;
            l.mlp_conv_proj_fp8 = l
                .mlp_conv_proj
                .as_ref()
                .map(|w| q(w, conv_rows, h))
                .transpose()?;
        }
        Ok(DflashAuxFp8 {
            fc: q(&self.fc, h, target_dim)?,
            fused_kv: q(&DenseWeight { weight: fused }, kv_rows, h)?,
            selector_hidden_proj: self
                .selector_hidden_proj
                .as_ref()
                .map(|w| q(w, self.selector_rank, h))
                .transpose()?,
        })
    }
}
