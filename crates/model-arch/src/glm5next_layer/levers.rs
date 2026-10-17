// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GLM-5.3-Flash layer launch levers: the prefill sub-chunk width, cuBLASLt for
//! wide projections, the batched DSA indexer query, and (2026-10-08) the row group of the
//! batched multi-sequence decode.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - 2026-10-17: [`GlmLevers::from_settings`] is the only reader: the loader resolves the levers
//!   once from the startup settings and hands the copy to the layer, its DSA and KDA mixers and
//!   its MLP kernels, which read plain fields. Nothing here reads the process environment.

use anyhow::{Result, anyhow};
use metrale_settings::InferenceServerSettings;

/// 2026-09-25: Default tokens per batched prefill sub-chunk, the width `Glm5NextLayer::prefill`
/// hands `Glm5NextLayer::forward_k` (overridable through `METRALE_GLM_PREFILL_ROWS`).
///
/// 16 is `DENSE_GEMV_BATCHM_MAX_M`, the widest M that `ops::dense_mm_bf16` sends to the batched
/// GEMV (`dense_gemv_bf16_batchm`), whose rows carry the same bits as the M = 1 GEMV. A wider
/// sub-chunk moves the dense projections to cuBLASLt or the tile GEMM, which accumulate in a
/// different order. Measured 2026-09-02: the 12 GLM prefill projection shapes cost 1.36-1.98x less
/// per token at M = 16 than at M = 8 in 11 of 12 (N4096 K128: 0.77x).
///
/// The routed MoE does not follow the width: `glm5next_mlp::forward::forward_moe` splits a wider
/// row group into even sub-groups of at most `MlpLevers::row_batch_max`, whose default and ceiling is
/// `MOE_ROW_BATCH_MAX_ROWS` (8).
pub(crate) const PREFILL_ROWS: usize = 16;

/// 2026-10-09: The widest `METRALE_GLM_ROW_GROUP` accepted: the GLM mHC `mix` scratch floor
/// (`MHC_MIX_MAX_TOKENS`), so every group fits `glm_hc_pre`.
pub(crate) const MAX_ROW_GROUP: usize = crate::glm5next_mhc::MHC_MIX_MAX_TOKENS;

/// 2026-10-17: The GLM layer levers, resolved once from the startup settings
/// ([`GlmLevers::from_settings`]). Each field names its variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlmLevers {
    /// 2026-09-25: `METRALE_GLM_CUBLAS_PROJ` (on unless switched off): send GLM projections
    /// with M above `DENSE_GEMV_BATCHM_MAX_M` to cuBLASLt BF16 instead of the tile GEMM. Read
    /// by the KDA, DSA and MLP blocks, always behind a `> DENSE_GEMV_BATCHM_MAX_M` row test, so
    /// the M = 1 GEMV and the batched GEMV are never replaced.
    pub cublas_wide_proj: bool,
    /// 2026-09-25: `METRALE_GLM_DSA_BATCH_QIDX`: compute the DSA indexer `wq_b` projection for
    /// all prefill rows in one cuBLASLt GEMM instead of one M = 1 GEMV per row. Prefill only: it
    /// is read in `select_rows_batched`, which runs only when `batch_select_enabled` holds
    /// (`is_prefill && k > 1`). Decode and the speculative verify keep the per-row GEMV. The
    /// GEMM accumulates in a different order from the GEMV.
    pub dsa_batch_qidx: bool,
    /// 2026-10-09: `METRALE_GLM_ROUTER_ROWS`: compute the router logits of a 2..=16-row group
    /// (the batched decode and verify) in one batched FP32-out GEMV instead of one M = 1 GEMV
    /// per row; each row's logits keep their bits (`glm5next_mlp::forward::router`).
    pub router_rows: bool,
    /// 2026-10-10: `METRALE_GLM_ROUTER_TOPK_RANK`: select the routed experts with
    /// `glm5next_router_topk_rank` (one rank count per expert) instead of `glm5next_router_topk`
    /// (one block-wide argmax per slot); same ids and weights byte for byte.
    pub router_topk_rank: bool,
    /// 2026-10-10: `METRALE_GLM_MOE_COMBINE_SPLIT`: spread each row of the MoE combine over
    /// `hidden / 256` blocks instead of one; every element's arithmetic is unchanged.
    pub moe_combine_split: bool,
    /// 2026-10-14: `METRALE_GLM_MOE_COMBINE_VEC8`: run the MoE combine as
    /// `glm5next_moe_combine[_indexed]_vec8`, eight elements per thread in 16-byte accesses;
    /// every element's arithmetic is unchanged.
    pub moe_combine_vec8: bool,
    /// 2026-10-09: `METRALE_GLM_DSA_INDEXER_ROWS`: run the DSA indexer's key and gate
    /// projections and the key norm of a batched decode or verify group once over all its rows
    /// (`Glm5NextDsaLayer::indexer_project_rows`) instead of once per row; each row is then
    /// placed from its staging row. On a captured decode the paged rows also store and select in
    /// one launch per stage (`select_paged_rows`) instead of seven launches per row; a paged
    /// prefill sub-chunk writes its latents, places its indexer rows and computes its head
    /// weights and selector query once per group of up to 16 rows. Each row keeps its bits.
    pub dsa_indexer_rows: bool,
    /// 2026-10-12: `METRALE_GLM_DSA_IDX_PROJ_WHOLE`: a DSA prefill sub-chunk's indexer key and
    /// compress gate, when the FP8 registry serves them, run over all of the sub-chunk's rows in
    /// one call each instead of 16-row groups (the W8A8 family is row-invariant: same bits).
    pub dsa_idx_proj_whole: bool,
    /// 2026-10-11: `METRALE_GLM_DSA_VERIFY_ROWS`: a captured k-row speculative verify of one
    /// sequence (`Glm5NextDsaLayer::decode_k_with`, k in 2..=16) computes the selector query and
    /// head weights of all rows in one launch each, and the indexer key, gate and key norm of
    /// all rows in one pass, as the batched decode does; each row is then placed and selected as
    /// before. Each row keeps its bits.
    pub dsa_verify_rows: bool,
    /// 2026-10-10: `METRALE_GLM_AR_FUSED_POST`: where an mHC site's all-reduce takes the
    /// one-shot path in one piece, the exchange runs alone (`CommBackend::exchange_partials`) and
    /// `glm5next_hc_post_rank_sum` sums the partials and post-mixes in one launch instead of
    /// `bf16_add_rank_sum` then `glm5next_hc_post`; every output keeps its bits.
    pub ar_fused_post: bool,
    /// 2026-10-10: `METRALE_GLM_MHC_POST_FOLD`: in the one-token decode (`forward_one`), with
    /// `mhc_fused`, the attention site's all-reduce sum and post-mix run inside the FFN site's
    /// pre-norm launch (`glm5next_hc_post_pre_norm`), the post-mixed streams written to the
    /// layer's `fold_streams` and post-mixed back into the highway by the FFN site's post; every
    /// output keeps its bits.
    pub mhc_post_fold: bool,
    /// 2026-10-09: `METRALE_GLM_MHC_FUSED`: run each mHC site's pre-mix and the RMS norm after it
    /// as one `glm5next_hc_pre_norm` launch (`glm5next_mhc_fused::glm_hc_pre_norm`) instead of
    /// `hc_mix_bf16`, `hc_finish` and `rms_norm_vanilla`; every output keeps its bits.
    pub mhc_fused: bool,
    /// 2026-09-25: `METRALE_GLM_KDA_CHUNK_PREFILL`: send a prefill sub-chunk's KDA mixer through
    /// the chunked scan (`Glm5NextKdaLayer::prefill`) instead of `decode_k`'s per-token
    /// recurrence. The chunked scan computes the recurrence chunk by chunk, in a different order,
    /// so its output is not bit-identical to the per-token walk. 2026-10-09: its conv now runs
    /// the decode's conv for all rows (`conv_tokens`); `examples/glm5next_kda_prefill_gate.rs`
    /// measures the carried state against the per-token walk.
    /// `METRALE_GLM_KDA_SEQ_TOKENS=1` is the bit-identical alternative.
    pub kda_chunk_prefill: bool,
    /// 2026-09-25: [`PREFILL_ROWS`], overridable with `METRALE_GLM_PREFILL_ROWS` (at least 1).
    /// `1` selects the per-token walk: `Glm5NextLayer::prefill` takes the batched sub-chunk
    /// path only when `rows > 1`. A width above `DENSE_GEMV_BATCHM_MAX_M` changes the
    /// projection numerics as well as the speed.
    pub prefill_rows: usize,
    /// 2026-10-08: Widest row group `Glm5NextLayer::forward_multi` runs through a layer,
    /// `METRALE_GLM_ROW_GROUP` (1..=[`MAX_ROW_GROUP`], declared default
    /// `DENSE_GEMV_BATCHM_MAX_M`, the widest M `ops::dense_mm_bf16` sends to the batched GEMV,
    /// whose rows carry the M = 1 GEMV's bits). A wider group reads each weight once per N rows
    /// instead of once per 16, but its BF16 projections run on cuBLASLt and a row's bits then
    /// depend on its batch-mates. The loader sizes every workspace for at least this many rows
    /// (`verify_k` in `glm5_next_load/loader.rs`).
    pub row_group: usize,
    /// 2026-10-12: `METRALE_GLM_DSA_WS_SHARED` (on unless switched off): the loader builds ONE
    /// `Glm5NextDsaWorkspace` for every DSA layer of the text stack (the KDA and MLP workspaces
    /// are shared the same way); off, one per layer, the pre-2026-10-12 allocation. Safe because
    /// the layers run in sequence on one stream and every field is per-call scratch (`decode_k`
    /// re-uploads its metadata each call).
    pub dsa_ws_shared: bool,
    /// 2026-09-25: `METRALE_GLM_MLP_WS_SHARED` (on unless switched off): the text layers share
    /// one MLP workspace.
    pub mlp_ws_shared: bool,
    /// 2026-09-25: `METRALE_DSA_SELECT_ROWS` (on unless switched off). Off,
    /// `Glm5NextDsaWorkspace::new` does not allocate the batched-selector buffers.
    pub dsa_select_rows: bool,
    /// 2026-09-25: `METRALE_GLM_DSA_ALLOC_PER_STEP`: a DSA layer allocates its `bt`/`sl` per call
    /// instead of using the workspace's (`Glm5NextDsaLayer::persist_bt` is its negation).
    pub dsa_alloc_per_step: bool,
    /// 2026-10-17: The mHC and SP-mHC levers (`crate::glm5next_mhc_levers`).
    pub mhc: crate::glm5next_mhc_levers::MhcLevers,
    /// 2026-10-17: The KDA mixer levers (`crate::glm5next_kda::levers`).
    pub kda: crate::glm5next_kda::KdaLevers,
    /// 2026-10-17: The DSA mixer levers (`crate::glm5next_dsa::levers`).
    pub dsa: crate::glm5next_dsa::levers::DsaLevers,
    /// 2026-10-17: The MLP routed-expert and dense launch levers
    /// (`crate::glm5next_mlp::forward::MlpLevers`).
    pub mlp: crate::glm5next_mlp::forward::MlpLevers,
    /// 2026-10-17: The dense-projection tier levers (`crate::glm5next_dense_levers`).
    pub dense: crate::glm5next_dense_levers::DenseLevers,
}

impl GlmLevers {
    /// 2026-10-17: Resolve from the startup settings. An out-of-range width is an error: the old
    /// reader ignored it and ran the default.
    pub fn from_settings(s: &InferenceServerSettings) -> Result<Self> {
        let (k, sc, m) = (&s.kernels, &s.scheduler, &s.model);
        let prefill_rows = match sc.glm_prefill_rows {
            None => PREFILL_ROWS,
            Some(r) => width("METRALE_GLM_PREFILL_ROWS", r, usize::MAX)?,
        };
        let row_group = width("METRALE_GLM_ROW_GROUP", sc.glm_row_group, MAX_ROW_GROUP)?;
        Ok(Self {
            cublas_wide_proj: m.glm_cublas_proj,
            dsa_batch_qidx: sc.glm_dsa_batch_qidx,
            router_rows: k.glm_router_rows,
            router_topk_rank: k.glm_router_topk_rank,
            moe_combine_split: k.glm_moe_combine_split,
            moe_combine_vec8: k.glm_moe_combine_vec8,
            dsa_indexer_rows: k.glm_dsa_indexer_rows,
            dsa_idx_proj_whole: k.glm_dsa_idx_proj_whole,
            dsa_verify_rows: k.glm_dsa_verify_rows,
            ar_fused_post: k.glm_ar_fused_post,
            mhc_post_fold: k.glm_mhc_post_fold,
            mhc_fused: k.glm_mhc_fused,
            kda_chunk_prefill: sc.glm_kda_chunk_prefill,
            prefill_rows,
            row_group,
            dsa_ws_shared: m.glm_dsa_ws_shared,
            mlp_ws_shared: m.glm_mlp_ws_shared,
            dsa_select_rows: s.runtime.dsa_select_rows,
            dsa_alloc_per_step: sc.glm_dsa_alloc_per_step,
            mhc: crate::glm5next_mhc_levers::MhcLevers::from_settings(s)?,
            kda: crate::glm5next_kda::KdaLevers::from_settings(s)?,
            dsa: crate::glm5next_dsa::levers::DsaLevers::from_settings(s)?,
            mlp: crate::glm5next_mlp::forward::MlpLevers::from_settings(s)?,
            dense: crate::glm5next_dense_levers::DenseLevers::from_settings(s)?,
        })
    }

    /// 2026-10-17: The log lines each lever used to print at its first read, printed once by the
    /// text-stack loader.
    pub(crate) fn announce(&self) {
        let max_m = metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M as usize;
        tracing::warn!(
            "METRALE_GLM_CUBLAS_PROJ: wide GLM projections (M > {max_m}) use {}",
            if self.cublas_wide_proj {
                "cuBLASLt BF16"
            } else {
                "the scalar tile GEMM"
            }
        );
        tracing::warn!(
            "METRALE_GLM_DSA_BATCH_QIDX: prefill DSA indexer q uses {}",
            if self.dsa_batch_qidx {
                "ONE batched cuBLASLt GEMM"
            } else {
                "one M=1 GEMV per row"
            }
        );
        if self.prefill_rows != PREFILL_ROWS {
            tracing::warn!(
                "GLM prefill sub-chunk overridden to {} rows (default {PREFILL_ROWS})",
                self.prefill_rows
            );
        }
        let mix = crate::glm5next_mhc::mhc_mix_max_tokens(self.prefill_rows);
        let floor = crate::glm5next_mhc::MHC_MIX_MAX_TOKENS;
        if mix != floor {
            tracing::warn!(
                "GLM mHC `mix` scratch widened to {mix} tokens (floor {floor}) to \
                 match METRALE_GLM_PREFILL_ROWS"
            );
        }
        if self.row_group != max_m {
            tracing::warn!(
                "METRALE_GLM_ROW_GROUP={rows}: batched GLM decode/verify groups of {rows} rows \
                 (default {max_m}); above {max_m} rows the projections leave the bit-identical \
                 batched GEMV",
                rows = self.row_group
            );
        }
        if !self.dsa_ws_shared {
            tracing::warn!(
                "GLM DSA workspace: PER-LAYER (METRALE_GLM_DSA_WS_SHARED=0), one scratch per \
                 DSA layer"
            );
        }
        if !self.mlp_ws_shared {
            tracing::warn!(
                "GLM MLP workspace: PER-LAYER (METRALE_GLM_MLP_WS_SHARED=0) — the pre-P2 \
                 allocation, one scratch per layer"
            );
        }
        if self.kda_chunk_prefill {
            tracing::warn!(
                "METRALE_GLM_KDA_CHUNK_PREFILL=1 - GLM prefill KDA uses the CHUNKED scan \
                 (kda_chunk_prepare + kda_chunk_scan). Not bit-identical to the per-token \
                 recurrent walk."
            );
        }
        self.mhc.announce();
        self.dsa.announce();
        self.mlp.announce();
    }
}

/// 2026-10-17: A row width from an integer lever: `1..=max`, else an error naming `env`.
fn width(env: &str, value: i64, max: usize) -> Result<usize> {
    usize::try_from(value)
        .ok()
        .filter(|w| (1..=max).contains(w))
        .ok_or_else(|| anyhow!("{env}={value}: expected a row count in 1..={max}"))
}

/// 2026-10-12: `METRALE_GLM_ARENA_LEAN`: off by default; on sizes the buffer arena for the
/// regions the GLM stack reads (`BufferSizes::lean_hc_stack`, applied by the engine's factory,
/// which refuses it on another model type).
pub fn arena_lean(s: &InferenceServerSettings) -> bool {
    s.model.glm_arena_lean
}

/// 2026-10-08: `rows` split into consecutive `(start, width)` groups of `cap` rows, the last one
/// shorter when `cap` does not divide `rows`. A `cap` of 0 is treated as 1. No group is empty,
/// and `rows == 0` gives none.
pub(crate) fn multi_seq_chunks(rows: usize, cap: usize) -> Vec<(usize, usize)> {
    let cap = cap.max(1);
    (0..rows)
        .step_by(cap)
        .map(|start| (start, cap.min(rows - start)))
        .collect()
}

#[cfg(test)]
#[path = "levers_settings_tests.rs"]
mod settings_tests;

#[cfg(test)]
impl GlmLevers {
    /// 2026-10-17: The levers of the declared defaults, for a test that builds a layer or a
    /// workspace.
    pub(crate) fn declared() -> Self {
        Self::from_settings(&metrale_model_layers::lever_equivalence::defaults())
            .expect("the declared defaults resolve")
    }
}
