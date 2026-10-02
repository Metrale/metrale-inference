// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: `pool_blocks`: the unclamped pool keeps the driver's share of its budget, the
//! clamped pool is cut to reachable demand, and an empty budget is refused.
//!
//! Owner: metrale-model-engine.
//! Invariants: none beyond the types.

use super::*;
use metrale_cache::kv_cache::KvCacheDtype;

/// 2026-10-02: 16 fp8 layers of 2 KV heads x 128 at 16 tokens per block: 128 KiB per block.
fn config() -> KvCacheConfig {
    KvCacheConfig {
        block_size: 16,
        num_kv_heads: 2,
        head_dim: 128,
        num_layers: 16,
        dtype: KvCacheDtype::Fp8,
        layer_dtypes: Vec::new(),
        layer_dims: Vec::new(),
        cache_blocks_per_seq: None,
    }
}

#[test]
fn an_unclamped_pool_keeps_the_drivers_share_of_its_budget() {
    let c = config();
    let block = c.block_bytes_kv_all_layers();
    let budget = 60_000 * block;
    let (blocks, budget_blocks) = pool_blocks(&c, budget, true, 8192, 16, 8).unwrap();
    assert_eq!(budget_blocks, 60_000);
    assert_eq!(
        blocks,
        (budget - budget / 1000 * PREFIX_POOL_DRIVER_PER_MILLE) / block
    );
    assert_eq!(60_000 - blocks, 180);
}

#[test]
fn a_clamped_pool_is_reachable_demand_and_keeps_no_share() {
    let c = config();
    let block = c.block_bytes_kv_all_layers();
    // 2026-10-02: 8 slots of 8192 tokens: 8 x 512 + 8 spare + 1 dummy.
    let (blocks, budget_blocks) = pool_blocks(&c, 60_000 * block, false, 8192, 16, 8).unwrap();
    assert_eq!((blocks, budget_blocks), (8 * 512 + 8 + 1, 60_000));
    // 2026-10-02: Below reachable demand the budget decides, whole.
    let (blocks, _) = pool_blocks(&c, 1_000 * block, false, 8192, 16, 8).unwrap();
    assert_eq!(blocks, 1_000);
}

#[test]
fn an_empty_budget_is_refused() {
    assert!(pool_blocks(&config(), 0, true, 8192, 16, 8).is_err());
}
