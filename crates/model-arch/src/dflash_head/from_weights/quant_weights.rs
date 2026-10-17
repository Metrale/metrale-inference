// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The DFlash drafter's quantized weight copies: each layer's seven dense GEMM
//! weights and the lm_head, in FP8 or (2026-10-10) NVFP4 (`--draft-weight-quantization`,
//! `draft_weight.rs`).
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: under the tensor-parallel drafter each copy is the one-rank copy's part
//! (`draft_weight.rs`): the NVFP4 scales come from the whole tensors (`DflashLayer::tp_src`,
//! `lm_head_whole`), the FP8 column slices from the whole rows.

use anyhow::Result;

use crate::dflash_head::draft_weight::{DraftQuantizer, DraftWeight};
use crate::dflash_head::tp::FullWeight;
use crate::dflash_head::{BlockDiffusionDraftHead, DflashQuantization};

/// 2026-09-26: Gives `head` copies of its layer weights and lm_head in `qz`'s format and sets
/// `quant` to match. `lm_head_native_fp8`: the target checkpoint's own FP8 LM head, shared
/// instead of a new FP8 copy (the FP8 tiers only). `lm_head_whole`: the whole LM head a
/// tensor-parallel rank's rows were cut from.
pub(super) fn quantize_drafter(
    head: &mut BlockDiffusionDraftHead,
    qz: &DraftQuantizer<'_>,
    q_dim: usize,
    kv_dim: usize,
    lm_head_native_fp8: Option<(metrale_model_layers::weight_map::Fp8DenseWeight, usize)>,
    lm_head_whole: Option<FullWeight>,
) -> Result<()> {
    let format = if qz.nvfp4 { "NVFP4" } else { "FP8 E4M3" };
    tracing::info!(
        target: "metrale_model_arch::dflash_head::from_weights",
        "DFlash Phase G: quantizing drafter weights to {format} ({} layers × 7 GEMMs)",
        head.num_layers
    );
    let h = head.hidden_size;
    let inter = head.intermediate_size;
    for (layer_idx, layer) in head.layers.iter_mut().enumerate() {
        let src = layer.tp_src;
        let whole = |f: fn(&crate::dflash_head::tp::LayerSources) -> FullWeight| {
            src.as_ref().map(|s| vec![f(s)]).unwrap_or_default()
        };
        let col_whole = |c: crate::dflash_head::tp::ColSlice, n: usize| FullWeight {
            weight: c.full,
            elems: n * c.full_k,
        };
        layer.q_proj_q = Some(qz.copy(&layer.q_proj, q_dim, h, &whole(|s| s.q), None)?);
        layer.k_proj_q = Some(qz.copy(&layer.k_proj, kv_dim, h, &whole(|s| s.k), None)?);
        layer.v_proj_q = Some(qz.copy(&layer.v_proj, kv_dim, h, &whole(|s| s.v), None)?);
        layer.o_proj_q = Some(qz.copy(
            &layer.o_proj,
            h,
            q_dim,
            &src.map(|s| vec![col_whole(s.o, h)]).unwrap_or_default(),
            src.map(|s| s.o),
        )?);
        layer.gate_proj_q = Some(qz.copy(&layer.gate_proj, inter, h, &whole(|s| s.gate), None)?);
        layer.up_proj_q = Some(qz.copy(&layer.up_proj, inter, h, &whole(|s| s.up), None)?);
        layer.down_proj_q = Some(qz.copy(
            &layer.down_proj,
            h,
            inter,
            &src.map(|s| vec![col_whole(s.down, h)]).unwrap_or_default(),
            src.map(|s| s.down),
        )?);
        tracing::debug!(
            target: "metrale_model_arch::dflash_head::from_weights",
            "DFlash Phase G: layer {} quantized", layer_idx
        );
    }
    // 2026-09-25: The lm_head's copy: on the FP8 tiers the checkpoint's own FP8 lm_head when
    // the model factory passes one (an `Fp8DenseWeight` like the layer copies, so the same tail
    // kernels read it), else a new copy of the shared BF16 lm_head in its own buffer, which
    // leaves the target's weight untouched.
    match lm_head_native_fp8 {
        Some((shared, rows)) if !qz.nvfp4 => {
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
            head.lm_head_q = Some(DraftWeight::Fp8(shared));
        }
        _ => {
            tracing::info!(
                target: "metrale_model_arch::dflash_head::from_weights",
                "DFlash Phase G: quantizing shared lm_head [{} × {}] to {format}",
                head.vocab_size,
                head.hidden_size
            );
            let lm_head_bf16 = metrale_model_layers::weight_map::DenseWeight {
                weight: head.lm_head_shared,
            };
            head.lm_head_q = Some(qz.copy(
                &lm_head_bf16,
                head.vocab_size,
                head.hidden_size,
                &lm_head_whole.into_iter().collect::<Vec<_>>(),
                None,
            )?);
        }
    }
    head.quant = if qz.nvfp4 {
        DflashQuantization::Nvfp4Weights
    } else {
        DflashQuantization::Fp8Weights
    };
    tracing::info!(
        target: "metrale_model_arch::dflash_head::from_weights",
        "DFlash Phase G: drafter weights ready as {format} (quant = {:?})",
        head.quant
    );
    Ok(())
}
