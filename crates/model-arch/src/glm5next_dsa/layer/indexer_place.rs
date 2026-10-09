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
            IndexerCache::Paged => Glm5NextDsaState::paged(&self.cfg),
        }
    }

    /// 2026-10-09: The paged geometry of this layer's indexer rows in `kv_cache`'s V pool;
    /// errors when a block's V side cannot hold them.
    pub fn paged_layout(&self, kv_cache: &PagedKvCache) -> Result<PagedIndexerLayout> {
        PagedIndexerLayout::new(
            kv_cache.block_size(),
            kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx),
            self.cfg.index_head_dim,
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
            None => IndexerRowsDev::paged(
                &self.paged_layout(kv_cache)?,
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                bt,
            ),
        }
    }

    /// 2026-10-09: Host-computed `(key, gate)` destinations of row `st.len()`.
    pub(super) fn host_row_ptrs(
        &self,
        st: &Glm5NextDsaState,
        kv_cache: &PagedKvCache,
        block_table: &[u32],
    ) -> Result<(DevicePtr, DevicePtr)> {
        let pos = st.len();
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
