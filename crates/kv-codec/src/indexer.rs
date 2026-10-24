// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The pooled layout of a sparse-attention indexer codec inside one KV block: `E =
//! block_size / pool` pool entries stored as the codec's rows (packed values, then their side
//! tensors), then a BF16 ring of `tail_sets x (pool - 1)` open key rows and as many gate rows.
//! The byte count is read from the two codecs' specs, so the indexer is sized the way the
//! attention cache is.
//!
//! Owner: kv-codec.
//! Invariants:
//! - A plan exists only for an indexer codec, at a head dim its spec serves, with `pool >= 2`
//!   and `tail_sets >= 1`; anything else is a coded `KvIndexerGeometry` error.
//! - [`IndexerPlan::block_bytes`] takes only a nonzero whole number of pools.

use alloc::format;

use metrale_error::{Error, ErrorCode, error};

use crate::registry::by_name;
use crate::spec::CodecRole;
use crate::{KvCodecId, spec};

/// 2026-10-18: The BF16 codec the open ring rows are stored in.
const RING: KvCodecId = match by_name("bf16") {
    Some(id) => id,
    None => panic!("the registry has no bf16 row"),
};

/// 2026-10-18: Ring rows per open slot: one key row and one gate row.
const RING_ROWS_PER_SLOT: u64 = 2;

/// 2026-10-18: One indexer layer's pooled rows, resolved once from the serve's settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexerPlan {
    codec: KvCodecId,
    head_dim: u32,
    pool: u32,
    tail_sets: u32,
}

impl IndexerPlan {
    /// 2026-10-18: Refuses a codec that is not an indexer codec, a head dim its spec does not
    /// serve, a pool under 2 tokens and an empty ring.
    pub fn new(codec: KvCodecId, head_dim: u32, pool: u32, tail_sets: u32) -> Result<Self, Error> {
        let s = spec(codec);
        let refuse = |why: alloc::string::String| error!(ErrorCode::KvIndexerGeometry, s.name, why);
        if s.role != CodecRole::Indexer {
            return Err(refuse(format!("{} is not an indexer codec", s.name)));
        }
        if !s.head_dims.contains(&head_dim) {
            return Err(refuse(format!(
                "head dim {head_dim} is not one of {:?}",
                s.head_dims
            )));
        }
        if pool < 2 || tail_sets < 1 {
            return Err(refuse(format!(
                "pool {pool} and ring sets {tail_sets} must be at least 2 and 1"
            )));
        }
        Ok(Self {
            codec,
            head_dim,
            pool,
            tail_sets,
        })
    }

    pub const fn codec(&self) -> KvCodecId {
        self.codec
    }

    pub const fn head_dim(&self) -> u32 {
        self.head_dim
    }

    pub const fn pool(&self) -> u32 {
        self.pool
    }

    pub const fn tail_sets(&self) -> u32 {
        self.tail_sets
    }

    /// 2026-10-18: Bytes one block of `block_size` tokens needs: the pool entries as the
    /// codec's rows, then the BF16 ring. Refuses a block that is not a nonzero whole number of
    /// pools.
    pub fn block_bytes(&self, block_size: u32) -> Result<u64, Error> {
        if block_size == 0 || !block_size.is_multiple_of(self.pool) {
            return Err(error!(
                ErrorCode::KvIndexerGeometry,
                spec(self.codec).name,
                format!(
                    "block size {block_size} is not a nonzero multiple of pool {}",
                    self.pool
                )
            ));
        }
        let d = u64::from(self.head_dim);
        let entries = u64::from(block_size / self.pool) * d;
        let ring = RING_ROWS_PER_SLOT * u64::from(self.tail_sets) * u64::from(self.pool - 1) * d;
        let s = spec(self.codec);
        Ok(s.data_bytes(entries) + s.side_bytes(entries) + spec(RING).data_bytes(ring))
    }
}

#[cfg(test)]
#[path = "indexer_tests.rs"]
mod tests;
