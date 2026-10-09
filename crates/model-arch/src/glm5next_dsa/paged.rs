// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Where a sequence's DSA indexer rows live, and the address math that reaches
//! them.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - [`row_elem`] is the host twin of `dsa_row_elem` in `kernels/gb10/common/dsa_indexer.cu`
//!   (and its b300 fork): same inputs, same element offset. Host-side writes use it, so a row
//!   the host places and a row a kernel reads land on one address.
//! - A [`PagedIndexerLayout`] exists only when the indexer rows of one block fit the block's
//!   V-side bytes ([`PagedIndexerLayout::new`] refuses otherwise).
//!
//! # Paged
//!
//! The DSA layers' paged KV cache keeps the MLA latent on its K side. The decode attention
//! passes the K pool as both K and V (absorbed NoPE MLA: K and V are the same latent), so the
//! V side of each DSA layer's pool is allocated, charged to the KV budget, and never read. A
//! paged indexer cache puts each token's `k_normed` and `gate` rows there, in the block that
//! holds the token's latent: block `b` of the V pool is `[block_size, index_head_dim]` BF16
//! keys followed by `[block_size, index_head_dim]` BF16 gates. GLM-5.3 needs 512 B per token
//! (2 x 128 BF16) against a 512 B FP8 latent row, so the pool does not grow.
//!
//! The rows then share everything the KV blocks have: capacity across sequences, reference
//! counts, the prefix cache, block moves. There is no `valid` byte: every row below the
//! sequence length is written by the same layer call that writes its latent, so a paged
//! cache passes `valid` NULL and the kernels treat every row below `S` as valid.

use anyhow::{Result, bail, ensure};
use metrale_gpu_runtime::gpu::DevicePtr;

/// 2026-10-09: Where a DSA layer keeps its sequences' indexer rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexerCache {
    /// 2026-10-09: One contiguous buffer per sequence, reserved at `--max-seq-len`
    /// (`Glm5NextDsaState::alloc`). The MTP drafter's DSA layer and batched-decode padding
    /// rows use it.
    Flat,
    /// 2026-10-09: The V side of the layer's paged KV pool ([`PagedIndexerLayout`]).
    Paged,
}

/// 2026-10-09: Where the text stack's DSA layers keep their indexer rows, from
/// `metrale_config::glm_dsa_indexer_paged` (paged unless `METRALE_GLM_DSA_INDEXER_FLAT=1`).
///
/// The loader, the serve's per-sequence reserve and the prefix-cache gate read that one
/// function, so the reserve charges the flat buffers exactly when the layers allocate them.
pub fn text_stack_indexer_cache() -> IndexerCache {
    if metrale_config::glm_dsa_indexer_paged() {
        IndexerCache::Paged
    } else {
        IndexerCache::Flat
    }
}

/// 2026-10-09: Element offset of indexer row `raw` in a BF16 key or gate region:
/// `raw * d` when `block_table` is `None` (flat), else
/// `block_table[raw / block_size] * blk_elems + (raw % block_size) * d`.
///
/// Errors when the table has no entry for the row's block.
pub fn row_elem(
    raw: usize,
    d: usize,
    block_table: Option<&[u32]>,
    block_size: usize,
    blk_elems: usize,
) -> Result<usize> {
    let Some(bt) = block_table else {
        return Ok(raw * d);
    };
    ensure!(block_size > 0, "paged indexer row: block size 0");
    let logical = raw / block_size;
    let physical = *bt.get(logical).ok_or_else(|| {
        anyhow::anyhow!(
            "paged indexer row {raw}: block table has {} entries, needs logical block {logical}",
            bt.len()
        )
    })? as usize;
    Ok(physical * blk_elems + (raw % block_size) * d)
}

/// 2026-10-09: The geometry of paged indexer rows inside one DSA layer's V pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedIndexerLayout {
    /// 2026-10-09: Tokens per KV block.
    pub block_size: usize,
    /// 2026-10-09: Bytes between V blocks of this layer (`v_block_stride_bytes_for_layer`).
    pub block_stride_bytes: usize,
    pub index_head_dim: usize,
}

impl PagedIndexerLayout {
    /// 2026-10-09: Refuses a block whose V side cannot hold `block_size` key rows and
    /// `block_size` gate rows of `index_head_dim` BF16, and an odd stride (the kernels step
    /// blocks in BF16 elements).
    pub fn new(
        block_size: usize,
        block_stride_bytes: usize,
        index_head_dim: usize,
    ) -> Result<Self> {
        ensure!(
            block_size > 0 && index_head_dim > 0,
            "paged indexer: block size {block_size} and index_head_dim {index_head_dim} must be \
             nonzero"
        );
        let need = block_size * index_head_dim * 4;
        if block_stride_bytes < need || !block_stride_bytes.is_multiple_of(2) {
            bail!(
                "paged DSA indexer: a KV block's V side is {block_stride_bytes} B but \
                 {block_size} tokens of keys and gates need {need} B (an even count). The \
                 indexer rows live in the DSA layers' unused V pool; a KV cache dtype narrower \
                 than FP8 leaves no room. Use --kv-cache-dtype fp8 or bf16, or set \
                 METRALE_GLM_DSA_INDEXER_FLAT=1."
            );
        }
        Ok(Self {
            block_size,
            block_stride_bytes,
            index_head_dim,
        })
    }

    /// 2026-10-09: Bytes from a block's start to its gate rows.
    pub fn gate_offset_bytes(&self) -> usize {
        self.block_size * self.index_head_dim * 2
    }

    /// 2026-10-09: The block stride in BF16 elements, the kernels' `blk_elems`.
    pub fn blk_elems(&self) -> usize {
        self.block_stride_bytes / 2
    }

    /// 2026-10-09: Byte offset, from the V pool's base, of row `raw`'s key; its gate is
    /// [`Self::gate_offset_bytes`] further.
    pub fn key_row_bytes(&self, block_table: &[u32], raw: usize) -> Result<usize> {
        Ok(row_elem(
            raw,
            self.index_head_dim,
            Some(block_table),
            self.block_size,
            self.blk_elems(),
        )? * 2)
    }
}

/// 2026-10-09: The device addressing the indexer kernels take: the key, gate and validity
/// bases and, for a paged cache, the block table and its geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexerRowsDev {
    pub k: DevicePtr,
    pub gate: DevicePtr,
    /// 2026-10-09: NULL for a paged cache: every row below `S` is valid.
    pub valid: DevicePtr,
    /// 2026-10-09: i32 block table of this sequence; NULL for a flat cache.
    pub bt: DevicePtr,
    pub block_size: u32,
    pub blk_elems: u32,
}

impl IndexerRowsDev {
    /// 2026-10-09: A flat cache's buffers; the kernels then address `raw * D`.
    pub fn flat(k: DevicePtr, gate: DevicePtr, valid: DevicePtr) -> Self {
        Self {
            k,
            gate,
            valid,
            bt: DevicePtr::NULL,
            block_size: 0,
            blk_elems: 0,
        }
    }

    /// 2026-10-09: Paged rows in `v_pool` through the device block table `bt`.
    pub fn paged(layout: &PagedIndexerLayout, v_pool: DevicePtr, bt: DevicePtr) -> Result<Self> {
        ensure!(
            bt.0 != 0,
            "paged DSA indexer: no device block table to address the rows through"
        );
        Ok(Self {
            k: v_pool,
            gate: v_pool.offset(layout.gate_offset_bytes()),
            valid: DevicePtr::NULL,
            bt,
            block_size: u32::try_from(layout.block_size)?,
            blk_elems: u32::try_from(layout.blk_elems())?,
        })
    }

    pub fn is_paged(&self) -> bool {
        self.bt.0 != 0
    }
}

#[cfg(test)]
#[path = "paged_tests.rs"]
mod tests;
