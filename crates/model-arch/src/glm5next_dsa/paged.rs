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

/// 2026-10-09: Where the text stack's DSA layers keep their indexer rows: paged unless
/// `cache.glm_dsa_indexer_flat` (`METRALE_GLM_DSA_INDEXER_FLAT`). 2026-10-17: From the resolved
/// settings; the loader and its V block size read it here.
pub fn text_stack_indexer_cache(s: &metrale_settings::InferenceServerSettings) -> IndexerCache {
    text_stack_indexer_cache_given(!s.cache.glm_dsa_indexer_flat)
}

/// 2026-10-17: [`text_stack_indexer_cache`] for a resolved `paged` decision.
pub fn text_stack_indexer_cache_given(paged: bool) -> IndexerCache {
    if paged {
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
    /// 2026-10-21: Pools of this many tokens keep a persisted f32 key in the block
    /// ([`Self::with_pool_keys`], `IndexerRowFormat::Bf16Pooled`); `None` for the rows alone.
    pub pool_keys_kp: Option<usize>,
}

/// 2026-10-21: The V block bytes of BF16 rows plus persisted pool keys: `bs` key rows and `bs`
/// gate rows of `d` BF16, then `bs / kp` f32 pool keys of `d` channels (pool `p`'s key in the
/// block of its tokens, entry `(p * kp % bs) / kp`). Refuses `bs` not a multiple of `kp`.
pub fn pooled_block_bytes(bs: usize, d: usize, kp: usize) -> Result<usize> {
    ensure!(
        kp > 0 && bs.is_multiple_of(kp),
        "DSA pool keys: KV block size {bs} must be a multiple of index_kpool {kp}"
    );
    Ok(bs * d * 4 + (bs / kp) * d * 4)
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
            pool_keys_kp: None,
        })
    }

    /// 2026-10-21: [`Self::new`] with the persisted pool keys of pools of `kp` tokens after the
    /// rows; refuses a V block smaller than [`pooled_block_bytes`] or not a whole number of f32.
    pub fn with_pool_keys(
        block_size: usize,
        block_stride_bytes: usize,
        index_head_dim: usize,
        kp: usize,
    ) -> Result<Self> {
        let need = pooled_block_bytes(block_size, index_head_dim, kp)?;
        ensure!(
            block_stride_bytes >= need && block_stride_bytes.is_multiple_of(4),
            "paged DSA indexer with pool keys: a KV block's V side is {block_stride_bytes} B but \
             {block_size} tokens of rows and their pool keys need {need} B (a multiple of 4); \
             the loader sizes it (IndexerRowFormat::Bf16Pooled)"
        );
        Ok(Self {
            pool_keys_kp: Some(kp),
            ..Self::new(block_size, block_stride_bytes, index_head_dim)?
        })
    }

    /// 2026-10-21: Bytes from a block's start to its pool keys (after the key and gate rows).
    pub fn pool_keys_offset_bytes(&self) -> usize {
        self.block_size * self.index_head_dim * 4
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
    /// 2026-10-16: `Some` for an FP8 cache (`super::fp8_rows`): `k` is then the V pool's base,
    /// `gate` NULL and `blk_elems` 0, and only the FP8 kernels may read the rows.
    pub fp8: Option<Fp8RowsDev>,
    /// 2026-10-21: The persisted pool keys' region (the V pool's base plus
    /// `pool_keys_offset_bytes`), NULL when the layout keeps none; blocks are `pk_blk` f32
    /// apart.
    pub pool_keys: DevicePtr,
    pub pk_blk: u32,
}

/// 2026-10-16: The FP8 cache's geometry as its kernels take it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp8RowsDev {
    pub blk_bytes: u32,
    pub kp: u32,
    pub tail_sets: u32,
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
            fp8: None,
            pool_keys: DevicePtr::NULL,
            pk_blk: 0,
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
            fp8: None,
            pool_keys: match layout.pool_keys_kp {
                Some(_) => v_pool.offset(layout.pool_keys_offset_bytes()),
                None => DevicePtr::NULL,
            },
            pk_blk: match layout.pool_keys_kp {
                Some(_) => u32::try_from(layout.block_stride_bytes / 4)?,
                None => 0,
            },
        })
    }

    /// 2026-10-16: FP8 rows in `v_pool` through the device block table `bt`.
    pub fn paged_fp8(
        layout: &super::fp8_rows::Fp8IndexerLayout,
        v_pool: DevicePtr,
        bt: DevicePtr,
    ) -> Result<Self> {
        ensure!(
            bt.0 != 0,
            "FP8 DSA indexer: no device block table to address the rows through"
        );
        Ok(Self {
            k: v_pool,
            gate: DevicePtr::NULL,
            valid: DevicePtr::NULL,
            bt,
            block_size: u32::try_from(layout.block_size)?,
            blk_elems: 0,
            fp8: Some(Fp8RowsDev {
                blk_bytes: u32::try_from(layout.block_stride_bytes)?,
                kp: u32::try_from(layout.kp)?,
                tail_sets: u32::try_from(layout.tail_sets)?,
            }),
            pool_keys: DevicePtr::NULL,
            pk_blk: 0,
        })
    }

    pub fn is_paged(&self) -> bool {
        self.bt.0 != 0
    }
}

#[cfg(test)]
#[path = "paged_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "paged_settings_tests.rs"]
mod settings_tests;
