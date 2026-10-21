// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Where a DSA layer's indexer rows go: [`IndexerPlace`], the new-sequence state
//! (flat or paged), and the addressing of a state's rows for the kernels and the host.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - A flat state's addressing is its own buffers, exactly what the layer used before the
//!   paged cache existed; a paged state's is this layer's V pool through a block table.

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::super::fp8_rows::{FP8_INDEX_HEAD_DIM, Fp8IndexerLayout};
use super::super::paged::{IndexerCache, IndexerRowsDev, PagedIndexerLayout};
use super::super::state::Glm5NextDsaState;
use super::Glm5NextDsaLayer;

/// 2026-10-09: Where `indexer_forward` places a sequence's next indexer row.
#[derive(Debug, Clone, Copy)]
pub enum IndexerPlace<'a> {
    /// 2026-10-09: The host computes the address: `pos * D` for a flat cache, the row's
    /// block in `block_table` for a paged one.
    Host { block_table: &'a [u32] },
    /// 2026-10-09: The projections write the staging rows and `dsa_indexer_store` places them
    /// at the device-side position `pos`, through the device block table `bt` when the cache
    /// is paged (`bt` is ignored for a flat one). A captured graph replays it at live
    /// positions.
    Device { pos: DevicePtr, bt: DevicePtr },
}

impl Glm5NextDsaLayer {
    /// 2026-10-09: A new sequence's indexer state, flat or paged per `indexer_cache`.
    pub fn alloc_dsa_state(&self, gpu: &dyn GpuBackend) -> Result<Glm5NextDsaState> {
        match self.indexer_cache {
            IndexerCache::Flat => Glm5NextDsaState::alloc(gpu, &self.cfg),
            IndexerCache::Paged => Glm5NextDsaState::paged_as(&self.cfg, self.indexer_format),
        }
    }

    /// 2026-10-09: The paged geometry of this layer's indexer rows in `kv_cache`'s V pool;
    /// errors when a block's V side cannot hold them.
    /// 2026-10-21: With the persisted pool keys under `IndexerRowFormat::Bf16Pooled`.
    pub fn paged_layout(&self, kv_cache: &PagedKvCache) -> Result<PagedIndexerLayout> {
        let (bs, stride) = (
            kv_cache.block_size(),
            kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx),
        );
        match self.indexer_format {
            super::super::fp8_rows::IndexerRowFormat::Bf16Pooled => {
                PagedIndexerLayout::with_pool_keys(
                    bs,
                    stride,
                    self.cfg.index_head_dim,
                    self.cfg.index_kpool,
                )
            }
            _ => PagedIndexerLayout::new(bs, stride, self.cfg.index_head_dim),
        }
    }

    /// 2026-10-16: The FP8 geometry of `st`'s rows in this layer's V pool; errors when the
    /// state is not FP8 or a block's V side cannot hold them.
    pub fn fp8_layout(
        &self,
        st: &Glm5NextDsaState,
        kv_cache: &PagedKvCache,
    ) -> Result<Fp8IndexerLayout> {
        let Some(ring) = st.fp8_ring() else {
            bail!(
                "DSA layer {}: the FP8 indexer layout of a BF16 state",
                self.layer_idx
            );
        };
        Fp8IndexerLayout::new(
            kv_cache.block_size(),
            kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx),
            self.cfg.index_head_dim,
            ring.kp,
            ring.tail_sets,
        )
    }

    /// 2026-10-09: The kernel addressing of `st`'s rows: its own buffers when flat, this
    /// layer's V pool through the device block table `bt` when paged.
    pub fn indexer_rows(
        &self,
        st: &Glm5NextDsaState,
        kv_cache: &PagedKvCache,
        bt: DevicePtr,
    ) -> Result<IndexerRowsDev> {
        match st.flat_rows() {
            Some(rows) => Ok(rows),
            None if st.fp8_ring().is_some() => IndexerRowsDev::paged_fp8(
                &self.fp8_layout(st, kv_cache)?,
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                bt,
            ),
            None => IndexerRowsDev::paged(
                &self.paged_layout(kv_cache)?,
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                bt,
            ),
        }
    }

    /// 2026-10-16: Refuses a KV cache whose V blocks cannot hold `st`'s paged rows in its
    /// format.
    pub(super) fn check_paged_room(
        &self,
        st: &Glm5NextDsaState,
        kv_cache: &PagedKvCache,
    ) -> Result<()> {
        match st.fp8_ring() {
            Some(_) => self.fp8_layout(st, kv_cache).map(|_| ()),
            None => self.paged_layout(kv_cache).map(|_| ()),
        }
    }

    /// 2026-10-16: `dsa_indexer_store_fp8` over staging rows `stage_row0..stage_row0 + n`: row
    /// `r` at device position `pos[r]` through block-table row `bt + r * bt_stride` (in
    /// entries). `seg_rows` 1 means every row is its own sequence and one launch places them;
    /// otherwise the rows are segments of `seg_rows` consecutive positions of one sequence and
    /// the pool entries launch before the ring rows (the kernel's store invariant).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn store_fp8_rows(
        &self,
        gpu: &dyn GpuBackend,
        rows: &IndexerRowsDev,
        stage_row0: usize,
        pos: DevicePtr,
        bt: DevicePtr,
        bt_stride: u32,
        seg_rows: usize,
        n: usize,
        stream: u64,
    ) -> Result<()> {
        let (Some(f), store) = (rows.fp8, self.select_kernels.fp8.store) else {
            bail!("DSA layer {}: an FP8 store of BF16 rows", self.layer_idx);
        };
        let d = self.cfg.index_head_dim;
        if store.0 == 0 || d != FP8_INDEX_HEAD_DIM || n == 0 || seg_rows == 0 {
            bail!(
                "DSA layer {}: the FP8 indexer store needs dsa_indexer_store_fp8, D \
                 {FP8_INDEX_HEAD_DIM} (D {d}) and rows ({n} rows, segments of {seg_rows})",
                self.layer_idx
            );
        }
        let w = &self.workspace;
        let phases: &[u32] = if seg_rows == 1 { &[3] } else { &[1, 2] };
        for &phase in phases {
            KernelLaunch::new(gpu, store)
                .grid([1, n as u32, 1])
                .block([d as u32, 1, 1])
                .arg_ptr(w.stage_k.offset(stage_row0 * d * 2))
                .arg_ptr(w.stage_gate.offset(stage_row0 * d * 2))
                .arg_ptr(pos)
                .arg_ptr(rows.k)
                .arg_ptr(self.weights.ape)
                .arg_u32(f.kp)
                .arg_u32(f.tail_sets)
                .arg_ptr(bt)
                .arg_u32(bt_stride)
                .arg_u32(rows.block_size)
                .arg_u32(f.blk_bytes)
                .arg_u32(seg_rows as u32)
                .arg_u32(n as u32)
                .arg_u32(phase)
                .launch(stream)?;
        }
        Ok(())
    }

    /// 2026-10-09: Host-computed `(key, gate)` destinations of row `st.len()`.
    pub(super) fn host_row_ptrs(
        &self,
        st: &Glm5NextDsaState,
        kv_cache: &PagedKvCache,
        block_table: &[u32],
    ) -> Result<(DevicePtr, DevicePtr)> {
        let pos = st.len();
        if st.fp8_ring().is_some() {
            bail!(
                "DSA layer {}: an FP8 indexer row is placed by its store kernel, not at a host \
                 address",
                self.layer_idx
            );
        }
        match st.cache() {
            IndexerCache::Flat => {
                let off = st.row_offset(pos);
                Ok((st.k_normed.offset(off), st.gate.offset(off)))
            }
            IndexerCache::Paged => {
                let layout = self.paged_layout(kv_cache)?;
                let pool = kv_cache.v_pool_ptr(self.attn_layer_idx);
                let key = layout.key_row_bytes(block_table, pos)?;
                Ok((
                    pool.offset(key),
                    pool.offset(key + layout.gate_offset_bytes()),
                ))
            }
        }
    }

    /// 2026-09-26: `decode_k`'s check that the indexer cache is in lockstep with the KV
    /// cache: rewinds `st` when it is ahead of `seq_len`, fails when it is behind.
    pub(super) fn check_lockstep(&self, st: &mut Glm5NextDsaState, seq_len: usize) -> Result<()> {
        // 2026-09-25: The indexer cache must advance in lockstep with the KV cache.
        //
        // * Ahead (`len > seq_len`) follows a rejected speculative draft: rewinding to
        //   `seq_len` makes the rows past it unreachable (the selector reads `[0, len)`) and
        //   the next write overwrites them.
        // * Behind (`len < seq_len`) means rows were never written, which is an error.
        // 2026-10-09: * Behind is expected for a paged cache after a prefix-cache hit: the
        //   rows below `seq_len` are in the shared KV blocks (`adopt_kv_rows`).
        match st.len().cmp(&seq_len) {
            std::cmp::Ordering::Greater => st.rewind_to(seq_len)?,
            std::cmp::Ordering::Less if st.cache() == IndexerCache::Paged => {
                st.adopt_kv_rows(seq_len)?
            }
            std::cmp::Ordering::Less => bail!(
                "DSA layer {}: indexer cache holds {} tokens but the sequence is at {} — \
                 rows are MISSING, not merely stale. The indexer stream must advance in \
                 lockstep with the KV cache.",
                self.layer_idx,
                st.len(),
                seq_len
            ),
            std::cmp::Ordering::Equal => {}
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "paged_rows_tests.rs"]
mod tests;
