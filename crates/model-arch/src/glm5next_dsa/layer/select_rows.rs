// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `select_paged_rows`, the indexer store and token selection of the first `m`
//! rows of a captured batched decode in one launch per stage (`dsa_indexer_rows.cu`), where
//! the per-row path issues seven launches per row (store, geometry, compression, scores,
//! top-k, expansion).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Taken only for rows that are each one sequence's decode row on a paged indexer cache, on
//!   the replay-safe (device-geometry) path, with the indexer rows staged
//!   (`indexer_project_rows`) and the selector query and head weights batched.
//! - Row `r` reads and writes what the per-row path's row `r` reads and writes: staging row
//!   `r`, metadata row `meta_row_base + r`, its own block-table row, query and weights row
//!   `r`, and token row `r` of the selection output; the scores, candidacy and selected pools
//!   use a region row of their own at the context ceiling's stride.
//!
//! # Why a row's selection is the per-row path's
//!
//! Per row, each rows entry runs the body the one-sequence entry runs (`dsa_indexer_body.cuh`)
//! with the same geometry (written by the same body from the same `seq_len`), and rows of
//! different sequences share no cache row. `dsa_pool_scores_rows` keeps a candidate pool's key
//! in shared memory instead of storing it to global memory and reloading it, and
//! `dsa_expand_selection_rows` recomputes the selected pools' token ids from the geometry
//! instead of reading those the compression stored; both are the same values.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layer::AttnMetadataDev;

use super::super::paged::IndexerRowsDev;
use super::super::select::{
    ROW_BLOCK, SCORES_BLOCK, contiguous_pool_count, topk_smem_for_tile, topk_tile,
};
use super::super::state::max_dsa_context;
use super::Glm5NextDsaLayer;

impl Glm5NextDsaLayer {
    /// 2026-10-09: The context ceiling's pool count, the per-row stride of the scores and
    /// candidacy regions.
    fn ceiling_pools(&self) -> usize {
        contiguous_pool_count(self.cfg.index_kpool, max_dsa_context(&self.cfg))
    }

    /// 2026-10-09: Whether `select_paged_rows` can run `m` rows: the rows kernels exist, the
    /// context ceiling has a pool, and the selection scratch holds `m` rows at the ceiling's
    /// strides.
    pub(super) fn paged_rows_ready(&self, m: usize) -> bool {
        let pools = self.ceiling_pools();
        self.select_kernels.rows.ready()
            && pools > 0
            && m <= self.workspace.max_rows
            && self
                .workspace
                .select
                .rows_regions(&self.cfg, m, pools, self.cfg.select_k(pools))
                .is_some()
    }

    /// 2026-10-09: Store and select rows `0..m`: staging row `r` placed at metadata position
    /// `meta_row_base + r` through metadata block-table row `meta_row_base + r` of the paged
    /// pool `rows0` addresses (`rows0` is row 0's addressing; the pool, block size and block
    /// stride are every paged row's), then each row's token selection into token row `r`.
    /// Advancing the states' row counters is the caller's.
    pub(super) fn select_paged_rows(
        &self,
        gpu: &dyn GpuBackend,
        m: usize,
        rows0: IndexerRowsDev,
        meta: &AttnMetadataDev,
        meta_row_base: usize,
        stream: u64,
    ) -> Result<()> {
        let pools = self.ceiling_pools();
        let sel = self.cfg.select_k(pools);
        let Some((scores, cand, selected)) = self
            .workspace
            .select
            .rows_regions(&self.cfg, m, pools, sel)
            .filter(|_| self.paged_rows_ready(m) && rows0.bt.0 != 0 && rows0.valid.0 == 0)
        else {
            bail!(
                "DSA layer {}: the rows selection cannot run {m} rows (kernels, scratch or a \
                 non-paged row 0)",
                self.layer_idx
            );
        };
        let c = &self.cfg;
        let k = &self.select_kernels.rows;
        let w = &self.workspace;
        let (d, h, kp) = (c.index_head_dim, c.index_heads, c.index_kpool);
        let bt_stride = meta.max_blocks_per_seq;
        let bt = meta
            .block_table
            .offset(meta_row_base * bt_stride as usize * 4);
        let pos = meta.positions.offset(meta_row_base * 4);
        let (m32, first_key) = (m as u32, 0i32);
        KernelLaunch::new(gpu, k.store)
            .grid([1, m32, 1])
            .block([d.min(1024) as u32, 1, 1])
            .arg_ptr(w.stage_k)
            .arg_ptr(w.stage_gate)
            .arg_ptr(pos)
            .arg_ptr(rows0.k)
            .arg_ptr(rows0.gate)
            .arg_u32(d as u32)
            .arg_ptr(bt)
            .arg_u32(bt_stride)
            .arg_u32(rows0.block_size)
            .arg_u32(rows0.blk_elems)
            .launch(stream)?;
        KernelLaunch::new(gpu, k.write_geom)
            .grid([m32, 1, 1])
            .block([1, 1, 1])
            .arg_ptr(meta.seq_len.offset(meta_row_base * 4))
            .arg_ptr(w.geom_rows)
            .arg_u32(kp as u32)
            .arg_u32(c.index_topk as u32)
            .arg_u32(topk_tile() as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, k.pool_scores)
            .grid([pools as u32, 1, m32])
            .block([SCORES_BLOCK, 1, 1])
            .shared_mem(((d + h) * 4) as u32)
            .arg_ptr(rows0.k)
            .arg_ptr(rows0.gate)
            .arg_ptr(self.weights.ape)
            .arg_ptr(w.q_idx_rows)
            .arg_ptr(w.head_weights_rows)
            .arg_ptr(pos)
            .arg_ptr(scores)
            .arg_ptr(cand)
            .arg_u32(h as u32)
            .arg_u32(d as u32)
            .arg_u32(kp as u32)
            .arg_i32(first_key)
            .arg_f32((d as f32).powf(-0.5))
            .arg_ptr(w.geom_rows)
            .arg_ptr(bt)
            .arg_u32(bt_stride)
            .arg_u32(rows0.block_size)
            .arg_u32(rows0.blk_elems)
            .arg_u32(pools as u32)
            .launch(stream)?;
        let np2 = pools.next_power_of_two().max(2).min(topk_tile());
        KernelLaunch::new(gpu, k.topk_pools)
            .grid([1, m32, 1])
            .block([ROW_BLOCK, 1, 1])
            .shared_mem(topk_smem_for_tile(np2) as u32)
            .arg_ptr(scores)
            .arg_ptr(selected)
            .arg_ptr(w.geom_rows)
            .arg_u32(pools as u32)
            .arg_u32(sel as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, k.expand_selection)
            .grid([1, m32, 1])
            .block([ROW_BLOCK, 1, 1])
            .arg_ptr(selected)
            .arg_ptr(cand)
            .arg_ptr(pos)
            .arg_ptr(w.q_mask)
            .arg_ptr(w.select.tokens())
            .arg_u32(kp as u32)
            .arg_u32(c.out_width() as u32)
            .arg_i32(first_key)
            .arg_i32(c.always_select_tail as i32)
            .arg_ptr(w.geom_rows)
            .arg_u32(pools as u32)
            .arg_u32(sel as u32)
            .launch(stream)
    }
}
