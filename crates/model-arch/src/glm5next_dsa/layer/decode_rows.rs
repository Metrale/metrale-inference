// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: `Glm5NextDsaLayer::decode_rows`, one decode token for each of `n` sequences in
//! one call: the batched multi-sequence decode's DSA mixer.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Every check (row counts, metadata rows, each row's lockstep and room) runs before the
//!   first launch.
//! - Row `r` reads and writes only sequence `r`'s state and metadata row
//!   `meta_row_base + r`; the projections (the selector query and head weights among them),
//!   the latent write and the attend are the only launches that span rows, and each computes
//!   every row on its own. (2026-10-09: under `decode_spans`, "sequence `r`" is the sequence
//!   whose span holds row `r`.)
//!
//! # Why a row's output equals the single-sequence decode's
//!
//! Per row, this issues the launches `decode_k` issues for that sequence alone at `k = 1` on a
//! decode step: `indexer_forward` (M = 1 GEMVs), the device geometry when capturing, and
//! `select_row` over the row's own indexer cache. The spanning launches give each row the
//! single-row bits: the latent write runs one block per row, reading the row's metadata slot;
//! `project_in`/`project_out` run the M = 1 GEMV at one row and `dense_gemv_bf16_batchm` at
//! 2..=`DENSE_GEMV_BATCHM_MAX_M` (`kernels/gb10/common/dense_gemv_bf16_batchm.cu`: each row's
//! result is bit-identical to `dense_gemv_bf16`); from two rows on (2026-10-09) the selector
//! query and head weights run `dense_gemv_bf16_batchm_fp32out`, whose rows are the M = 1
//! `dense_gemv_bf16_fp32out`'s; the RMSNorm runs one block per row; `glm5next_dsa_mla_decode_fp8` runs
//! one block per (head, row) reading that row's block table, `seq_len` and selection row.
//! Above `DENSE_GEMV_BATCHM_MAX_M` rows the projections move to cuBLASLt and the identity is
//! lost, so the layer above hands this at most that many rows (`multi_seq_chunk_rows`).

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layer::{AttnMetadataDev, ForwardContext, LayerState};

use super::super::attend::DsaDecodePaging;
use super::super::paged::IndexerCache;
use super::super::state::Glm5NextDsaState;
use super::{Glm5NextDsaLayer, IndexerPlace, gemm};

/// 2026-10-08: Row `r`'s `Glm5NextDsaState`; errors on any other state type.
fn dsa_row<'a>(
    states: &'a mut [&mut (dyn LayerState + 'static)],
    r: usize,
) -> Result<&'a mut Glm5NextDsaState> {
    states[r]
        .as_any_mut()
        .downcast_mut::<Glm5NextDsaState>()
        .ok_or_else(|| {
            anyhow::anyhow!("Glm5NextDsaLayer: row {r} got a state that is not Glm5NextDsaState")
        })
}

/// 2026-10-09: Consecutive rows of one sequence in a [`Glm5NextDsaLayer::decode_spans`] call:
/// `rows` tokens at positions `first_pos..first_pos + rows`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DsaRowSpan {
    pub first_pos: usize,
    pub rows: usize,
}

impl Glm5NextDsaLayer {
    /// 2026-10-08: One decode token for each of `states.len()` sequences: row `r` of `hidden`
    /// is sequence `r`'s token at position `seq_lens[r]`, and its position, KV slot,
    /// `seq_len` and block table are row `meta_row_base + r` of `meta`. The output projection
    /// is written over `hidden`, as `decode_k` writes it. 2026-10-09: [`Self::decode_spans`]
    /// with one row per sequence.
    ///
    /// Errors before any launch when the row counts disagree, the rows do not fit the
    /// workspace or the metadata, a capture lacks the replay-safe kernels, or a row's
    /// indexer cache is behind its sequence or full. An indexer cache ahead of its sequence
    /// (a rejected draft) is rewound first, as in `decode_k`.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_rows(
        &self,
        hidden: DevicePtr,
        states: &mut [&mut (dyn LayerState + 'static)],
        seq_lens: &[usize],
        kv_cache: &mut PagedKvCache,
        meta: &AttnMetadataDev,
        meta_row_base: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let spans: Vec<DsaRowSpan> = seq_lens
            .iter()
            .map(|&first_pos| DsaRowSpan { first_pos, rows: 1 })
            .collect();
        self.decode_spans(
            hidden,
            states,
            &spans,
            kv_cache,
            meta,
            meta_row_base,
            ctx,
            stream,
        )
    }

    /// 2026-10-09: `spans[s].rows` consecutive tokens for each sequence `s` (`states[s]`), the
    /// rows sequence-major: the batched speculative verify's DSA mixer, and with one row per
    /// sequence the batched decode's. Per row, in row order: the indexer write and that row's
    /// selection, as `decode_k` issues them for one sequence, so a row selects over its own
    /// sequence's indexer rows up to its own position and never sees a later row's. The
    /// projections, the latent write and the attend each run once over all rows; row `r`'s
    /// position, KV slot, `seq_len` and block table are metadata row `meta_row_base + r`.
    ///
    /// Errors before any launch on the conditions [`Self::decode_rows`] lists, checked per
    /// sequence at the span's first position and for room for all its rows.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_spans(
        &self,
        hidden: DevicePtr,
        states: &mut [&mut (dyn LayerState + 'static)],
        spans: &[DsaRowSpan],
        kv_cache: &mut PagedKvCache,
        meta: &AttnMetadataDev,
        meta_row_base: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        use crate::glm5next_layer::profile;
        let n: usize = spans.iter().map(|s| s.rows).sum();
        if n == 0
            || states.len() != spans.len()
            || spans.iter().any(|s| s.rows == 0)
            || n > self.workspace.max_rows
        {
            bail!(
                "DSA layer {}: {} row states and {} spans of {n} rows for a workspace built for \
                 {} rows",
                self.layer_idx,
                states.len(),
                spans.len(),
                self.workspace.max_rows
            );
        }
        if meta_row_base + n > meta.num_seqs as usize {
            bail!(
                "DSA layer {}: rows {meta_row_base}..{} pass the {} metadata rows",
                self.layer_idx,
                meta_row_base + n,
                meta.num_seqs
            );
        }
        // 2026-10-08: A captured graph replays over later positions, so it must place the
        // indexer row and size the selection from device memory; without these two kernels
        // it would bake this step's host offsets.
        let replay_safe = ctx.graph_capture;
        if replay_safe
            && (self.select_kernels.indexer_store.0 == 0 || self.select_kernels.write_geom.0 == 0)
        {
            bail!(
                "DSA layer {}: a captured batched decode needs dsa_indexer_store and \
                 dsa_write_geom, which this target lacks",
                self.layer_idx
            );
        }
        for (s, span) in spans.iter().enumerate() {
            let st = dsa_row(states, s)?;
            self.check_lockstep(st, span.first_pos)?;
            st.ensure_room(span.rows)?;
            // 2026-10-09: A paged row is always placed on the device, at its metadata
            // position through its metadata block-table row: this call has no host tables.
            if st.cache() == IndexerCache::Paged {
                if self.select_kernels.indexer_store.0 == 0 {
                    bail!(
                        "DSA layer {}: a paged batched decode needs dsa_indexer_store, which \
                         this target lacks",
                        self.layer_idx
                    );
                }
                self.paged_layout(kv_cache)?;
            }
        }
        let bt_stride = meta.max_blocks_per_seq as usize * 4;
        let row_seq: Vec<usize> = spans
            .iter()
            .enumerate()
            .flat_map(|(s, span)| std::iter::repeat_n(s, span.rows))
            .collect();

        let gpu = ctx.gpu;
        let block_size = kv_cache.config().block_size;
        let t_proj = profile::start();
        self.project_in(gpu, hidden, n, stream)?;
        // 2026-10-08: One latent-write launch for every row: the metadata's slots are a
        // contiguous i64 array, and the kernel writes row `i` to `slot[i]`.
        self.write_latent_rows(
            gpu,
            0,
            n,
            kv_cache,
            meta.slot.offset(meta_row_base * 8),
            stream,
        )?;
        profile::end(profile::DSA_PROJ, t_proj, gpu, stream);
        // 2026-10-09: The selector query (`wq_b` over `q_resid`) and head weights
        // (`weights_proj` over the layer input) of every row in one launch each, so `wq_b`
        // (12.6 MB on GLM-5.3) is read once per group instead of once per row. Each row of
        // `dense_gemv_bf16_batchm_fp32out` is the M = 1 `gemv_f32`'s result for that row, so
        // this runs only where the single-row path would take `gemv_f32`, and only for two
        // rows or more: one row keeps the single-sequence launches.
        let w = &self.workspace;
        let (heads, idx_row) = (
            self.cfg.index_heads,
            self.cfg.index_heads * self.cfg.index_head_dim,
        );
        let batched_idx = n > 1
            && self.kernels.gemv_batchm_f32.0 != 0
            && self.kernels.gemv_f32.0 != 0
            && w.q_idx_rows.0 != 0
            && w.head_weights_rows.0 != 0;
        if batched_idx {
            // 2026-10-09: Above `DENSE_GEMV_BATCHM_MAX_M` rows (a `METRALE_GLM_ROW_GROUP` wider
            // than 16) `glm_mm`'s cuBLASLt arm writes BF16, so these FP32-out projections take
            // the FP32-out cuBLASLt call instead.
            let wide = n > metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M as usize
                && crate::glm5next_layer::cublas_wide_proj();
            let k = &self.kernels;
            for (a, wt, out, n_out, kk) in [
                (
                    w.q_resid,
                    self.weights.wq_b,
                    w.q_idx_rows,
                    idx_row,
                    self.cfg.q_lora_rank,
                ),
                (
                    hidden,
                    self.weights.weights_proj,
                    w.head_weights_rows,
                    heads,
                    self.cfg.hidden,
                ),
            ] {
                if wide {
                    metrale_model_layers::layers::ops::cublas_bf16_proj_dense_f32_out(
                        a,
                        wt,
                        out,
                        n as u32,
                        n_out as u32,
                        kk as u32,
                        stream,
                    )?;
                } else {
                    gemm(
                        gpu,
                        k.gemm_f32,
                        k.gemv_f32,
                        k.batchm_f32(),
                        a,
                        wt,
                        out,
                        n,
                        n_out,
                        kk,
                        stream,
                    )?;
                }
            }
        }
        for r in 0..n {
            let mr = meta_row_base + r;
            let t = profile::start();
            let st = dsa_row(states, row_seq[r])?;
            let bt_row = meta.block_table.offset(mr * bt_stride);
            let place = if replay_safe || st.cache() == IndexerCache::Paged {
                IndexerPlace::Device {
                    pos: meta.positions.offset(mr * 4),
                    bt: bt_row,
                }
            } else {
                // 2026-10-09: A flat row's host address is `pos * D`; it needs no table.
                IndexerPlace::Host { block_table: &[] }
            };
            self.indexer_forward_with(
                gpu,
                hidden.offset(r * self.cfg.hidden * 2),
                st,
                kv_cache,
                place,
                !batched_idx,
                stream,
            )?;
            profile::end(profile::DSA_INDEXER, t, gpu, stream);
            if replay_safe {
                self.write_geom(gpu, meta.seq_len.offset(mr * 4), stream)?;
            }
            let pre = batched_idx.then(|| {
                (
                    w.q_idx_rows.offset(r * idx_row * 4),
                    w.head_weights_rows.offset(r * heads * 4),
                )
            });
            let rows = self.indexer_rows(st, kv_cache, bt_row)?;
            self.select_row(
                gpu,
                r,
                st,
                rows,
                meta.positions.offset(mr * 4),
                replay_safe,
                pre,
                stream,
            )?;
        }

        let paging = DsaDecodePaging {
            num_seqs: n,
            num_q_heads: self.cfg.local_heads,
            num_kv_heads: 1,
            max_blocks_per_seq: meta.max_blocks_per_seq as usize,
            block_size,
            cache_stride_bytes: (block_size * self.cfg.kv_lora_rank) as u64,
        };
        let st0 = dsa_row(states, 0)?;
        self.attend_rows(
            gpu,
            n,
            st0,
            kv_cache,
            meta.block_table.offset(meta_row_base * bt_stride),
            meta.seq_len.offset(meta_row_base * 4),
            &paging,
            stream,
        )?;
        self.project_out(gpu, hidden, n, stream)
    }
}

#[cfg(test)]
#[path = "decode_rows_tests.rs"]
mod tests;
