// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The drafter's KV pool: one region of whole blocks per live proposer state.
//!
//! Before this, the pool held one sequence's blocks and every state claimed all of them, so at
//! two or more live sequences their drafter latent rows overwrote each other at the same slots
//! (only the DSA indexer cache was per state). Each state now owns a disjoint region.
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants:
//! - Two live states never hold the same block: a region is handed out at most once until
//!   it is released.
//! - A region's block table is `region * per_seq .. (region + 1) * per_seq`.

use super::*;

/// 2026-10-11: Rows per drafter KV block.
pub(super) const DRAFTER_KV_BLOCK: usize = 16;

/// 2026-10-11: One KV head of `kv_lora_rank` per token, the absorbed-MLA latent that the
/// block's `latent_write` and paged gather address.
pub(super) fn drafter_kv_config(kv_lora_rank: usize) -> KvCacheConfig {
    KvCacheConfig {
        block_size: DRAFTER_KV_BLOCK,
        num_kv_heads: 1,
        head_dim: kv_lora_rank,
        num_layers: 1,
        dtype: KvCacheDtype::Fp8,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    }
}

/// 2026-10-11: Blocks one state's region holds for `rows` drafter rows (two spare, as the
/// single-sequence pool had).
pub(super) fn blocks_per_seq(rows: usize) -> usize {
    rows / DRAFTER_KV_BLOCK + 2
}

/// 2026-10-11: Device bytes of one state's region at `--max-seq-len` `max_seq_len`, for the
/// per-sequence reserve (`seq_state_reserve`). The same capped row count, layout and block
/// count `Glm5NextMtpHead::new` allocates.
pub fn drafter_kv_bytes_per_seq(
    config: &metrale_config::ModelConfig,
    max_seq_len: usize,
) -> Result<usize> {
    let dsa = crate::glm5next_dsa::Glm5NextDsaConfig::from_config(config)?;
    let rows = drafter_context_rows(max_seq_len, &dsa);
    Ok(blocks_per_seq(rows) * drafter_kv_config(dsa.kv_lora_rank).block_bytes_kv_all_layers())
}

/// 2026-10-11: The block table of `region`: `per_seq` consecutive blocks.
pub(super) fn region_table(region: usize, per_seq: usize) -> Vec<u32> {
    (region * per_seq..(region + 1) * per_seq)
        .map(|b| b as u32)
        .collect()
}

/// 2026-10-11: The free regions of the pool. Regions are handed out lowest first.
pub(super) struct DrafterKvRegions {
    per_seq: usize,
    free: Mutex<Vec<usize>>,
}

impl DrafterKvRegions {
    pub(super) fn new(regions: usize, per_seq: usize) -> Self {
        Self {
            per_seq,
            free: Mutex::new((0..regions).rev().collect()),
        }
    }

    /// 2026-10-11: A free region and its block table. Errors when every region is held: the
    /// pool is sized for `--max-batch-size` live sequences, so a miss means a state was not
    /// released.
    pub(super) fn claim(&self) -> Result<(usize, Vec<u32>)> {
        let region = self.free.lock().pop().ok_or_else(|| {
            anyhow::anyhow!(
                "GLM MTP drafter KV pool: every per-sequence region is held (the pool is sized \
                 for --max-batch-size live sequences; a proposer state was not freed)"
            )
        })?;
        Ok((region, region_table(region, self.per_seq)))
    }

    /// 2026-10-11: Return `region`. Errors on a region that is already free.
    pub(super) fn release(&self, region: usize) -> Result<()> {
        let mut free = self.free.lock();
        if free.contains(&region) {
            bail!("GLM MTP drafter KV pool: region {region} released twice");
        }
        free.push(region);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-11: Live regions never share a block, and every table stays inside the pool.
    #[test]
    fn live_regions_are_disjoint_and_inside_the_pool() {
        let (regions, per) = (5, 7);
        let pool = DrafterKvRegions::new(regions, per);
        let tables: Vec<Vec<u32>> = (0..regions).map(|_| pool.claim().unwrap().1).collect();
        let mut all: Vec<u32> = tables.iter().flatten().copied().collect();
        assert_eq!(all.len(), regions * per);
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), regions * per, "two regions share a block");
        assert!(all.iter().all(|&b| (b as usize) < regions * per));
    }

    /// 2026-10-11: A full pool refuses instead of aliasing, and a released region is handed
    /// out again; a double release is refused.
    #[test]
    fn a_full_pool_refuses_and_a_released_region_returns() {
        let pool = DrafterKvRegions::new(2, 3);
        let (a, _) = pool.claim().unwrap();
        let (b, _) = pool.claim().unwrap();
        assert_ne!(a, b);
        assert!(pool.claim().is_err());
        pool.release(a).unwrap();
        assert!(pool.release(a).is_err());
        let (c, table) = pool.claim().unwrap();
        assert_eq!(c, a);
        assert_eq!(table, region_table(a, 3));
    }

    /// 2026-10-11: A region's table is its own run of consecutive blocks.
    #[test]
    fn a_region_table_is_its_run_of_blocks() {
        assert_eq!(region_table(0, 3), [0, 1, 2]);
        assert_eq!(region_table(2, 3), [6, 7, 8]);
    }
}
