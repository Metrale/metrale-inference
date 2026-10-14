// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GLM-5.3-Flash layer launch levers: the prefill sub-chunk width, cuBLASLt for
//! wide projections, the batched DSA indexer query, and (2026-10-08) the row group of the
//! batched multi-sequence decode.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Each function reads its environment variable once per process and caches the result.

/// 2026-09-25: Default tokens per batched prefill sub-chunk, the width `Glm5NextLayer::prefill`
/// hands `Glm5NextLayer::forward_k` (overridable through [`prefill_rows`]).
///
/// 16 is `DENSE_GEMV_BATCHM_MAX_M`, the widest M that `ops::dense_mm_bf16` sends to the batched
/// GEMV (`dense_gemv_bf16_batchm`), whose rows carry the same bits as the M = 1 GEMV. A wider
/// sub-chunk moves the dense projections to cuBLASLt or the tile GEMM, which accumulate in a
/// different order. Measured 2026-09-02: the 12 GLM prefill projection shapes cost 1.36-1.98x less
/// per token at M = 16 than at M = 8 in 11 of 12 (N4096 K128: 0.77x).
///
/// The routed MoE does not follow the width: `glm5next_mlp::forward::forward_moe` splits a wider
/// row group into even sub-groups of at most `row_batch_max()`, whose default and ceiling is
/// `MOE_ROW_BATCH_MAX_ROWS` (8).
pub(crate) const PREFILL_ROWS: usize = 16;

/// 2026-09-25: Send GLM projections with M above `DENSE_GEMV_BATCHM_MAX_M` to cuBLASLt BF16
/// instead of the tile GEMM. Read by the KDA, DSA and MLP blocks, always behind a
/// `> DENSE_GEMV_BATCHM_MAX_M` row test, so the M = 1 GEMV and the batched GEMV are never
/// replaced. On unless `METRALE_GLM_CUBLAS_PROJ=0`.
pub(crate) fn cublas_wide_proj() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_CUBLAS_PROJ").as_deref() != Ok("0");
        tracing::warn!(
            "METRALE_GLM_CUBLAS_PROJ: wide GLM projections (M > {}) use {}",
            metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M,
            if on {
                "cuBLASLt BF16"
            } else {
                "the scalar tile GEMM"
            }
        );
        on
    })
}

/// 2026-09-25: Compute the DSA indexer `wq_b` projection for all prefill rows in one cuBLASLt
/// GEMM instead of one M = 1 GEMV per row. Off unless `METRALE_GLM_DSA_BATCH_QIDX=1`.
///
/// Prefill only: it is read in `select_rows_batched`, which runs only when
/// `batch_select_enabled` holds (`is_prefill && k > 1`). Decode and the speculative verify keep
/// the per-row GEMV. The GEMM accumulates in a different order from the GEMV.
pub(crate) fn dsa_batch_qidx() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_DSA_BATCH_QIDX").as_deref() == Ok("1");
        tracing::warn!(
            "METRALE_GLM_DSA_BATCH_QIDX: prefill DSA indexer q uses {}",
            if on {
                "ONE batched cuBLASLt GEMM"
            } else {
                "one M=1 GEMV per row"
            }
        );
        on
    })
}

/// 2026-10-09: `METRALE_GLM_ROUTER_ROWS=1` computes the router logits of a 2..=16-row group
/// (the batched decode and verify) in one batched FP32-out GEMV instead of one M = 1 GEMV per
/// row; each row's logits keep their bits (`glm5next_mlp::forward::router::router_logits`).
/// Off until measured end to end. Read once.
pub(crate) fn router_rows() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("METRALE_GLM_ROUTER_ROWS").as_deref() == Ok("1"))
}

/// 2026-10-10: `METRALE_GLM_ROUTER_TOPK_RANK=1` selects the routed experts with
/// `glm5next_router_topk_rank` (one rank count per expert) instead of `glm5next_router_topk`
/// (one block-wide argmax per slot); same ids and weights byte for byte
/// (`glm5next_mlp::forward::topk_combine`). Off until measured end to end. Read once.
pub(crate) fn router_topk_rank() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("METRALE_GLM_ROUTER_TOPK_RANK").as_deref() == Ok("1"))
}

/// 2026-10-10: `METRALE_GLM_MOE_COMBINE_SPLIT=1` spreads each row of the MoE combine over
/// `hidden / 256` blocks instead of one; every element's arithmetic is unchanged
/// (`glm5next_mlp::forward::topk_combine`). Off until measured end to end. Read once.
pub(crate) fn moe_combine_split() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("METRALE_GLM_MOE_COMBINE_SPLIT").as_deref() == Ok("1"))
}

/// 2026-10-09: `METRALE_GLM_DSA_INDEXER_ROWS=1` runs the DSA indexer's key and gate
/// projections and the key norm of a batched decode or verify group once over all its rows
/// (`Glm5NextDsaLayer::indexer_project_rows`) instead of once per row; each row is then placed
/// from its staging row. On a captured decode the paged rows also store and select in one
/// launch per stage (`select_paged_rows`) instead of seven launches per row; a paged prefill
/// sub-chunk writes its latents, places its indexer rows and computes its head weights and
/// selector query once per group of up to 16 rows (`Glm5NextDsaLayer::decode_k_with`). Each
/// row keeps its bits. Off until measured end to end. Read once.
pub(crate) fn dsa_indexer_rows() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("METRALE_GLM_DSA_INDEXER_ROWS").as_deref() == Ok("1"))
}

/// 2026-10-12: `METRALE_GLM_DSA_IDX_PROJ_WHOLE=1`: a DSA prefill sub-chunk's indexer key and
/// compress gate, when the FP8 registry serves them, run over all of the sub-chunk's rows in one
/// call each instead of 16-row groups (the W8A8 family is row-invariant: same bits). Read once.
pub(crate) fn dsa_idx_proj_whole() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("METRALE_GLM_DSA_IDX_PROJ_WHOLE").as_deref() == Ok("1"))
}

/// 2026-10-11: `METRALE_GLM_DSA_VERIFY_ROWS=1`: a captured k-row speculative verify of one
/// sequence (`Glm5NextDsaLayer::decode_k_with`, k in 2..=16) computes the selector query (`wq_b`,
/// 12.6 MB BF16 on GLM-5.3) and head weights of all rows in one launch each, and the indexer
/// key, gate and key norm of all rows in one pass, as the batched decode does
/// (`selector_query_rows`, `indexer_project_rows`); each row is then placed and selected as
/// before. The per-row path read `wq_b` once per row. Each row keeps its bits (the batched
/// FP32-out GEMV's rows are the M = 1 GEMV's). Off until measured end to end. Read once.
pub(crate) fn dsa_verify_rows() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("METRALE_GLM_DSA_VERIFY_ROWS").as_deref() == Ok("1"))
}

/// 2026-10-14: `METRALE_GLM_DSA_SPAN_SELECT=1`: a captured speculative verify stores and selects
/// the paged indexer rows of all its rows in one launch per stage (`select_paged_rows`), as a
/// one-row-per-sequence batched decode already does: the multi-sequence verify's leading paged
/// spans (`Glm5NextDsaLayer::decode_spans_with`) and a one-sequence k-row verify under
/// `METRALE_GLM_DSA_VERIFY_ROWS` (`decode_k_with`). Row `r` selects only keys below its own
/// `seq_len`, so storing its span's later rows first leaves its bits. Off until measured end to
/// end. Read once.
pub(crate) fn dsa_span_select() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("METRALE_GLM_DSA_SPAN_SELECT").as_deref() == Ok("1"))
}

/// 2026-10-10: `METRALE_GLM_AR_FUSED_POST=1`: where an mHC site's all-reduce takes the one-shot
/// path in one piece, the exchange runs alone (`CommBackend::exchange_partials`) and
/// `glm5next_hc_post_rank_sum` sums the partials and post-mixes in one launch instead of
/// `bf16_add_rank_sum` then `glm5next_hc_post`; every output keeps its bits. Off until measured
/// end to end. Read once.
pub(crate) fn ar_fused_post() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("METRALE_GLM_AR_FUSED_POST").as_deref() == Ok("1"))
}

/// 2026-10-10: `METRALE_GLM_MHC_POST_FOLD=1`: in the one-token decode (`forward_one`), with
/// `METRALE_GLM_MHC_FUSED=1`, the attention site's all-reduce sum and post-mix run inside the FFN
/// site's pre-norm launch (`glm5next_hc_post_pre_norm`), the post-mixed streams written to the
/// layer's `fold_streams` and post-mixed back into the highway by the FFN site's post; every
/// output keeps its bits. Off until measured end to end. Read once.
pub(crate) fn mhc_post_fold() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("METRALE_GLM_MHC_POST_FOLD").as_deref() == Ok("1"))
}

/// 2026-10-09: `METRALE_GLM_MHC_FUSED=1` runs each mHC site's pre-mix and the RMS norm after
/// it as one `glm5next_hc_pre_norm` launch (`glm5next_mhc_fused::glm_hc_pre_norm`) instead of
/// `hc_mix_bf16`, `hc_finish` and `rms_norm_vanilla`; every output keeps its bits. Off until
/// measured end to end. Read once.
pub(crate) fn mhc_fused() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("METRALE_GLM_MHC_FUSED").as_deref() == Ok("1"))
}

/// 2026-09-25: `PREFILL_ROWS`, overridable at launch with `METRALE_GLM_PREFILL_ROWS` (values
/// below 1 or unparsable are ignored). `1` selects the per-token walk: `Glm5NextLayer::prefill`
/// takes the batched sub-chunk path only when `rows > 1`. A width above
/// `DENSE_GEMV_BATCHM_MAX_M` changes the projection numerics as well as the speed.
pub fn prefill_rows() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        let r = std::env::var("METRALE_GLM_PREFILL_ROWS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|r| *r >= 1)
            .unwrap_or(PREFILL_ROWS);
        if r != PREFILL_ROWS {
            tracing::warn!("GLM prefill sub-chunk overridden to {r} rows (default {PREFILL_ROWS})");
        }
        r
    })
}

/// 2026-10-08: Widest row group `Glm5NextLayer::forward_multi` runs through a layer:
/// `DENSE_GEMV_BATCHM_MAX_M`, the widest M `ops::dense_mm_bf16` sends to the batched GEMV, whose
/// rows carry the M = 1 GEMV's bits. A wider group would move every projection to cuBLASLt,
/// whose rows depend on their batch-mates. The loader sizes every workspace for at least this
/// many rows (`verify_k` in `glm5_next_load/loader.rs`).
///
/// 2026-10-09: `METRALE_GLM_ROW_GROUP=N` (opt-in, 1..=`MAX_ROW_GROUP`) widens the group, so a
/// batched decode or verify of more than 16 rows reads each weight once per N rows instead of
/// once per 16. Above 16 rows the BF16 projections run on cuBLASLt and a row's bits then
/// depend on its batch-mates: the default stays 16 until transcripts and accuracy clear a
/// wider group. Read once.
pub(crate) fn multi_seq_chunk_rows() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        let default = metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M as usize;
        let rows = std::env::var("METRALE_GLM_ROW_GROUP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|r| (1..=MAX_ROW_GROUP).contains(r))
            .unwrap_or(default);
        if rows != default {
            tracing::warn!(
                "METRALE_GLM_ROW_GROUP={rows}: batched GLM decode/verify groups of {rows} rows \
                 (default {default}); above {default} rows the projections leave the \
                 bit-identical batched GEMV"
            );
        }
        rows
    })
}

/// 2026-10-09: The widest `METRALE_GLM_ROW_GROUP` accepted: the GLM mHC `mix` scratch floor
/// (`MHC_MIX_MAX_TOKENS`), so every group fits `glm_hc_pre`.
pub(crate) const MAX_ROW_GROUP: usize = crate::glm5next_mhc::MHC_MIX_MAX_TOKENS;

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

/// 2026-09-25: `METRALE_GLM_KDA_CHUNK_PREFILL=1` sends a prefill sub-chunk's KDA mixer through
/// the chunked scan (`Glm5NextKdaLayer::prefill`) instead of `decode_k`'s per-token recurrence.
/// Off unless set to `1`; read once. The chunked scan computes the recurrence chunk by chunk, in
/// a different order, so its output is not bit-identical to the per-token walk.
/// 2026-10-09: Until today its conv rounded SiLU to BF16 and a separate L2 rounded again, which
/// moved q/k by up to a BF16 ulp against the decode's fused conv (one rounding) and drove the
/// carried state 2.8e-3 RMS away from the per-token walk on checkpoint weights; it now runs the
/// decode's conv for all rows (`conv_tokens`), and the state stays within 1e-6 RMS
/// (`examples/glm5next_kda_prefill_gate.rs`). `METRALE_GLM_KDA_SEQ_TOKENS=1` is the
/// bit-identical alternative and is faster on GB10.
pub(crate) fn kda_chunk_prefill() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_KDA_CHUNK_PREFILL").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "METRALE_GLM_KDA_CHUNK_PREFILL=1 - GLM prefill KDA uses the CHUNKED scan \
                 (kda_chunk_prepare + kda_chunk_scan). Not bit-identical to the per-token \
                 recurrent walk."
            );
        }
        on
    })
}

/// 2026-10-12: `METRALE_GLM_ARENA_LEAN`, read once: unset or `0` off, `1` sizes the buffer arena
/// for the regions the GLM stack reads (`BufferSizes::lean_hc_stack`, applied by the engine's
/// factory, which refuses it on another model type); another value is an error.
pub fn arena_lean() -> anyhow::Result<bool> {
    static V: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    V.get_or_init(
        || match std::env::var("METRALE_GLM_ARENA_LEAN").ok().as_deref() {
            None | Some("0") => Ok(false),
            Some("1") => Ok(true),
            Some(other) => Err(format!("METRALE_GLM_ARENA_LEAN={other:?}: expected 0 or 1")),
        },
    )
    .clone()
    .map_err(anyhow::Error::msg)
}
