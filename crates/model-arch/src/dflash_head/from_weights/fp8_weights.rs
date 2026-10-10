// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The DFlash drafter's FP8 weight copies: each layer's seven dense GEMM
//! weights and the lm_head.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use crate::dflash_head::{BlockDiffusionDraftHead, DflashQuantization};

/// 2026-09-26: Gives `head` FP8 copies of its layer weights and lm_head, and sets
/// `quant = Fp8Weights`.
pub(super) fn quantize_drafter_fp8(
    head: &mut BlockDiffusionDraftHead,
    gpu: &dyn GpuBackend,
    q_dim: usize,
    kv_dim: usize,
    lm_head_native_fp8: Option<(metrale_model_layers::weight_map::Fp8DenseWeight, usize)>,
) -> Result<()> {
    tracing::info!(
        target: "metrale_model_arch::dflash_head::from_weights",
        "DFlash Phase G: quantizing drafter weights to FP8 E4M3 ({} layers × 7 GEMMs)",
        head.num_layers
    );
    let stream = 0u64;
    let q_dim_local = q_dim;
    let kv_dim_local = kv_dim;
    let h = head.hidden_size;
    let inter = head.intermediate_size;
    let quant_k = head.kernels.quantize_bf16_to_fp8;
    for (layer_idx, layer) in head.layers.iter_mut().enumerate() {
        layer.q_proj_fp8 =
            Some(
                layer
                    .q_proj
                    .quantize_to_fp8(gpu, quant_k, q_dim_local, h, stream)?,
            );
        layer.k_proj_fp8 =
            Some(
                layer
                    .k_proj
                    .quantize_to_fp8(gpu, quant_k, kv_dim_local, h, stream)?,
            );
        layer.v_proj_fp8 =
            Some(
                layer
                    .v_proj
                    .quantize_to_fp8(gpu, quant_k, kv_dim_local, h, stream)?,
            );
        layer.o_proj_fp8 = Some(quantize_cols(
            gpu,
            quant_k,
            &layer.o_proj,
            layer.o_proj_cols,
            h,
            q_dim_local,
            stream,
        )?);
        layer.gate_proj_fp8 = Some(
            layer
                .gate_proj
                .quantize_to_fp8(gpu, quant_k, inter, h, stream)?,
        );
        layer.up_proj_fp8 = Some(
            layer
                .up_proj
                .quantize_to_fp8(gpu, quant_k, inter, h, stream)?,
        );
        layer.down_proj_fp8 = Some(quantize_cols(
            gpu,
            quant_k,
            &layer.down_proj,
            layer.down_proj_cols,
            h,
            inter,
            stream,
        )?);
        tracing::debug!(
            target: "metrale_model_arch::dflash_head::from_weights",
            "DFlash Phase G: layer {} quantized", layer_idx
        );
    }
    // 2026-09-25: The lm_head's FP8 copy: the checkpoint's own FP8 lm_head when the
    // model factory passes one (`lm_head_native_fp8`, an `Fp8DenseWeight` like the
    // layer copies, so the same tail kernels read it), else a new FP8
    // quantization of the shared BF16 lm_head in its own buffer, which leaves the
    // target's weight untouched.
    if let Some((shared, rows)) = lm_head_native_fp8 {
        anyhow::ensure!(
            rows == head.vocab_size,
            "native FP8 lm_head share rows ({rows}) != drafter vocab ({}) — \
                     the drafter's tail GEMM iterates head.vocab_size rows",
            head.vocab_size
        );
        tracing::info!(
            target: "metrale_model_arch::dflash_head::from_weights",
            "DFlash Phase G: sharing the checkpoint's NATIVE FP8 lm_head \
             [{} × {}] (1.27 GB runtime mirror skipped)",
            head.vocab_size,
            head.hidden_size
        );
        head.lm_head_shared_fp8 = Some(shared);
    } else {
        tracing::info!(
            target: "metrale_model_arch::dflash_head::from_weights",
            "DFlash Phase G: quantizing shared lm_head [{} × {}]",
            head.vocab_size,
            head.hidden_size
        );
        let lm_head_bf16 = metrale_model_layers::weight_map::DenseWeight {
            weight: head.lm_head_shared,
        };
        head.lm_head_shared_fp8 = Some(lm_head_bf16.quantize_to_fp8(
            gpu,
            quant_k,
            head.vocab_size,
            head.hidden_size,
            stream,
        )?);
    }
    head.quant = DflashQuantization::Fp8Weights;
    tracing::info!(
        target: "metrale_model_arch::dflash_head::from_weights",
        "DFlash Phase G: drafter weights ready as FP8 (quant = Fp8Weights). \
         Set METRALE_DFLASH_DRAFTER_FP8=0 to revert to BF16."
    );
    Ok(())
}

/// 2026-10-10: The FP8 copy of an `[n, k]` weight. A tensor-parallel column slice (`cols`)
/// quantizes the whole rows (`[n, cols.full_k]`), keeps this rank's `k` columns from
/// `cols.start` and the whole rows' scales, and frees the rest, so its bytes and scales are
/// the one-rank drafter's.
fn quantize_cols(
    gpu: &dyn GpuBackend,
    quant_k: metrale_gpu_runtime::gpu::KernelHandle,
    local: &metrale_model_layers::weight_map::DenseWeight,
    cols: Option<crate::dflash_head::tp::ColSlice>,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<metrale_model_layers::weight_map::Fp8DenseWeight> {
    let Some(c) = cols else {
        return local.quantize_to_fp8(gpu, quant_k, n, k, stream);
    };
    anyhow::ensure!(
        c.start + k <= c.full_k,
        "DFlash TP FP8: columns {}..{} past the whole row of {}",
        c.start,
        c.start + k,
        c.full_k
    );
    let full = c.full.quantize_to_fp8(gpu, quant_k, n, c.full_k, stream)?;
    let weight = gpu.alloc(n * k)?;
    gpu.copy_d2d_2d_async(
        full.weight.offset(c.start),
        c.full_k,
        weight,
        k,
        k,
        n,
        stream,
    )?;
    gpu.synchronize(stream)?;
    gpu.free(full.weight)?;
    Ok(metrale_model_layers::weight_map::Fp8DenseWeight {
        weight,
        row_scale: full.row_scale,
    })
}
