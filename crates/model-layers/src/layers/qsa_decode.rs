// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: The QSA indexer's decode step: ingest at the device position, then,
//! past the inert bound, selection; and the host bookkeeping of a step a replayed CUDA
//! graph served.
//!
//! Owner: model-layers (QSA).
//! Invariants:
//! - After every decode step `ingested` is the step's `pos + 1` and `pooled` is
//!   `ingested / ratio`, whether the step ran eagerly, was captured, or was replayed.
//! - A replayed step lies inside the inert bound: selection's host top-k cannot be in a
//!   graph, so the model runs any step that selects eagerly
//!   (`decode_graph_unsupported_at`).
//!
//! A child module of `qsa` (via `#[path]`), so the indexer's private fields are
//! reachable without widening their visibility.

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::{QsaIndexer, QsaSelection, QsaSeqState};
use crate::layers::ops;

impl QsaIndexer {
    /// 2026-10-13: Whether the decode step at `pos` selects (`pos + 1` visible tokens
    /// past the inert bound); a step that does not runs dense attention, exactly.
    pub fn selects_at(&self, pos: usize) -> bool {
        pos + 1 > self.inert_bound()
    }

    /// 2026-10-13: Errors unless the `k` decode steps from `seq_len` can be served by a
    /// replayed graph: the counters are in sync, the keys fit, and no step selects.
    pub fn check_replay(&self, st: &QsaSeqState, seq_len: usize, k: usize) -> Result<()> {
        anyhow::ensure!(
            st.ingested == seq_len,
            "QSA: replay at {seq_len} but {} tokens ingested",
            st.ingested
        );
        anyhow::ensure!(
            k > 0 && seq_len + k <= self.max_tokens,
            "QSA: a replay of {k} steps from {seq_len} exceeds the indexer's max_tokens {} \
             (--max-seq-len)",
            self.max_tokens
        );
        anyhow::ensure!(
            !self.selects_at(seq_len + k - 1),
            "QSA: a replayed graph cannot select (step at {}); the step must run eagerly",
            seq_len + k - 1
        );
        Ok(())
    }

    /// 2026-10-13: The host bookkeeping of `k` decode steps from `seq_len` that a
    /// replayed graph served: the graph ingested and pooled on the device.
    pub fn replayed_steps(&self, st: &mut QsaSeqState, seq_len: usize, k: usize) -> Result<()> {
        self.check_replay(st, seq_len, k)?;
        self.advance_ingested(st, seq_len + k);
        Ok(())
    }

    /// 2026-10-13: Mark the start of a decode step, before any capture or replay.
    pub fn begin_step(&self, st: &mut QsaSeqState) {
        st.step_start = st.ingested;
    }

    /// 2026-10-13: Undo the host bookkeeping of a step whose CUDA-graph capture recorded
    /// it without running it, so the step re-runs eagerly from where it began.
    pub fn rearm_step(&self, st: &mut QsaSeqState) {
        self.advance_ingested(st, st.step_start);
    }

    fn advance_ingested(&self, st: &mut QsaSeqState, ingested: usize) {
        st.ingested = ingested;
        st.pooled = ingested / self.ratio as usize;
    }

    /// 2026-09-25: Decode-step ingest + selection for the token at `pos` (0-based;
    /// `pos + 1` visible). `None` inside the inert bound (dense is exact).
    /// Errors when `pos` is not the ingested count or reaches `max_tokens`.
    ///
    /// 2026-10-13: `pos_dev` holds `pos` as a u32 on the device (the step's position
    /// row); the ingest reads it there, so inside the inert bound this step is
    /// capture-safe and a replayed graph ingests at the replayed position.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_select(
        &self,
        st: &mut QsaSeqState,
        normed: DevicePtr,
        pos: usize,
        pos_dev: DevicePtr,
        k_pool: DevicePtr,
        v_pool: DevicePtr,
        block_table_dev: DevicePtr,
        block_size: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<QsaSelection>> {
        anyhow::ensure!(
            pos == st.ingested,
            "QSA: decode at pos {pos} but {} tokens ingested — the indexer \
             cache lost sync (prefix-cache skip or a rewound sequence)",
            st.ingested
        );
        anyhow::ensure!(
            pos < self.max_tokens,
            "QSA: pos {pos} >= the indexer's max_tokens {} (--max-seq-len)",
            self.max_tokens
        );

        let hd = self.hd as usize;
        let qkw = self.qk_width();
        ops::cublas_bf16_proj_dense(
            normed,
            self.qk_proj_w,
            self.qk_scratch,
            1,
            qkw as u32,
            self.hidden,
            stream,
        )
        .context("QSA qk projection (decode)")?;
        // 2026-10-13: A no-op when every complete block is pooled, which prefill
        // ingest and restore leave; were one behind, re-pooling it at a replay
        // rewrites the same bytes.
        self.pool_new_blocks(st, gpu, stream)?;
        ops::qsa_ingest_step(
            gpu,
            self.k_ingest_k,
            self.qk_scratch.offset(self.n_heads as usize * hd * 2),
            st.raw_keys,
            self.k_norm_w,
            st.block_keys,
            pos_dev,
            self.ratio,
            self.hd,
            self.rot,
            self.theta,
            self.eps,
            stream,
        )?;
        self.advance_ingested(st, pos + 1);

        let visible = pos + 1;
        let complete = visible / self.ratio as usize;
        if !self.selects_at(pos) {
            return Ok(None); // 2026-09-25: all blocks are selected, so dense is exact.
        }

        ops::qsa_qprep(
            gpu,
            self.k_qprep_k,
            self.qk_scratch,
            self.q_norm_w,
            self.q_post,
            self.n_heads,
            self.hd,
            self.rot,
            pos as u32,
            self.theta,
            self.eps,
            stream,
        )?;
        ops::qsa_score(
            gpu,
            self.k_score_k,
            self.q_post,
            st.block_keys,
            self.scores_dev,
            complete as u32,
            self.n_heads,
            self.hd,
            stream,
        )?;

        // 2026-09-25: Host top-k over the block scores. The D2H cannot be
        // captured; a layer with an indexer reports `decode_graph_unsupported`.
        let mut raw = vec![0u8; complete * 4];
        gpu.copy_d2h_on_stream(self.scores_dev, &mut raw, stream)?;
        let scores: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut order: Vec<u32> = (0..complete as u32).collect();
        // 2026-09-25: Sort by (-score, index): ties go to the lower block index.
        order.sort_by(|&a, &b| {
            scores[b as usize]
                .partial_cmp(&scores[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let mut blocks: Vec<u32> = order[..self.block_topk as usize].to_vec();
        blocks.sort_unstable();

        let ratio = self.ratio as usize;
        let mut sel: Vec<i32> = Vec::with_capacity(self.budget as usize + ratio);
        for b in &blocks {
            let base = *b as i32 * self.ratio as i32;
            for r in 0..self.ratio as i32 {
                sel.push(base + r);
            }
        }
        for t in complete * ratio..visible {
            sel.push(t as i32);
        }
        let n_sel = sel.len() as u32;

        let sel_bytes: Vec<u8> = sel.iter().flat_map(|v| v.to_le_bytes()).collect();
        gpu.copy_h2d_async(&sel_bytes, self.sel_dev, stream)?;
        ops::qsa_gather(
            gpu,
            self.k_gather_k,
            k_pool,
            v_pool,
            block_table_dev,
            self.sel_dev,
            self.k_scratch,
            self.v_scratch,
            n_sel,
            block_size,
            self.nkv_attn,
            self.hd_attn,
            stream,
        )?;

        // 2026-09-25: Identity table + seq_len for the scratch-as-paged-cache view.
        let pages = (n_sel as usize).div_ceil(block_size as usize);
        if st.table_len < pages {
            let ident: Vec<u8> = (0..pages as i32).flat_map(|v| v.to_le_bytes()).collect();
            gpu.copy_h2d_async(&ident, self.table_dev, stream)?;
            st.table_len = pages;
        }
        gpu.copy_h2d_async(&(n_sel as i32).to_le_bytes(), self.seq_len_dev, stream)?;

        Ok(Some(QsaSelection {
            k_scratch: self.k_scratch,
            v_scratch: self.v_scratch,
            table_dev: self.table_dev,
            seq_len_dev: self.seq_len_dev,
            n_sel,
            max_blocks: pages as u32,
        }))
    }
}
