// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: Tests for `reclaim_for_prompt` on a real `RadixTree` and a `PagedKvCache` on the
//! mock GPU: a warm prompt's admission reclaim must not evict the cached prefix that prompt is
//! about to reuse.
//!
//! Owner: model-engine (prefix cache).
//! Invariants: none beyond the types.

use super::*;
use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype};
use metrale_cache::radix_tree::RadixTree;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

const BS: usize = 16;

fn pool(gpu: &MockGpuBackend, blocks: usize) -> PagedKvCache {
    let config = KvCacheConfig {
        block_size: BS,
        num_kv_heads: 1,
        head_dim: 16,
        num_layers: 1,
        dtype: KvCacheDtype::Fp8,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    };
    PagedKvCache::new(config, blocks, gpu).unwrap()
}

/// 2026-09-30: A finished sequence over `tokens`, left as `cache_sequence` then
/// `free_sequence` leave it: the cache holds the only reference on each of its blocks.
fn cache_finished(tree: &RadixTree, kv: &mut PagedKvCache, tokens: &[u32]) -> Vec<u32> {
    let blocks: Vec<u32> = (0..tokens.len() / BS)
        .map(|_| kv.alloc_block().unwrap())
        .collect();
    let acquired = tree.insert(tokens, &blocks, &[], BS, 0, 0);
    crate::model::block_mgmt::cache_acquires_refs(&acquired, kv);
    tree.release(tokens, BS, 0);
    kv.free_blocks(&blocks);
    blocks
}

/// 2026-09-30: The high-ISL warm gate in miniature. A 10-block pool holds a finished 96-token
/// (6-block) sequence with SSM checkpoints at 48 and 96 tokens. The warm prompt is those 96
/// tokens plus 4 more, so admission counts 100/16 + 1 = 7 blocks for it, and only 4 are free.
/// It reuses 6 cached blocks, so it needs 1 from the free list and nothing may be evicted.
/// Evicting for all 7 cuts the tail of its own prefix, and the restore falls back to the
/// 48-token checkpoint.
#[test]
fn warm_admission_keeps_the_prefix_it_reuses() {
    let gpu = MockGpuBackend::new();
    let mut kv = pool(&gpu, 10);
    let tree = RadixTree::new();
    let cold: Vec<u32> = (1000..1096).collect();
    let cached = cache_finished(&tree, &mut kv, &cold);
    tree.insert_intermediate_snapshot(&cold[..48], &[], &[], BS, 1, 0, 0, 0);
    tree.insert_intermediate_snapshot(&cold, &[], &[], BS, 2, 0, 0, 0);
    assert_eq!(kv.num_free_blocks(), 4);

    let mut warm = cold.clone();
    warm.extend(5000..5004);
    let got = reclaim_for_prompt(&tree, &mut kv, &warm, 0, warm.len() / BS + 1);

    assert_eq!(
        got,
        PrefixReclaim { target: 1, free: 4 },
        "only the unmatched tail is reserved, and nothing is evicted for it"
    );
    for &b in &cached {
        assert_eq!(
            kv.ref_count(b),
            1,
            "cached block {b} must keep the cache's reference"
        );
    }
    assert_eq!(tree.peek_matched_tokens(&warm, BS, 0), 96);
    let m = tree.lookup(&warm, BS, 0, 0);
    assert_eq!(
        (m.matched_tokens, m.ssm_snapshot_tokens),
        (96, 96),
        "the restore point is the deepest checkpoint within the kept prefix"
    );
}

/// 2026-09-30: When the unmatched tail does not fit even after the reuse discount, the reclaim
/// still evicts what it may (an unrelated, newer entry) and reports the shortfall, while the
/// reused prefix, older in LRU order, survives. Its pin is dropped afterwards: a later plain
/// eviction can take it.
#[test]
fn tail_that_cannot_fit_evicts_others_but_never_the_reused_prefix() {
    let gpu = MockGpuBackend::new();
    let mut kv = pool(&gpu, 10);
    let tree = RadixTree::new();
    let reused: Vec<u32> = (1000..1096).collect();
    let other: Vec<u32> = (7000..7016).collect();
    cache_finished(&tree, &mut kv, &reused);
    cache_finished(&tree, &mut kv, &other);
    assert_eq!(kv.num_free_blocks(), 3);

    let mut warm = reused.clone();
    warm.extend(5000..5080);
    let got = reclaim_for_prompt(&tree, &mut kv, &warm, 0, warm.len() / BS + 1);

    assert_eq!(got, PrefixReclaim { target: 6, free: 4 });
    assert_eq!(
        tree.peek_matched_tokens(&other, BS, 0),
        0,
        "the unrelated entry is evicted"
    );
    assert_eq!(
        tree.peek_matched_tokens(&warm, BS, 0),
        96,
        "the reused prefix is kept"
    );
    assert_eq!(
        tree.evict(6).physical.len(),
        6,
        "the pin is released after the reclaim"
    );
}

/// 2026-09-30: An empty prompt reuses nothing (prompt logprobs), so the whole count is reserved
/// and the reclaim evicts as before.
#[test]
fn a_prompt_that_reuses_nothing_reserves_the_whole_count() {
    let gpu = MockGpuBackend::new();
    let mut kv = pool(&gpu, 10);
    let tree = RadixTree::new();
    let cold: Vec<u32> = (1000..1096).collect();
    cache_finished(&tree, &mut kv, &cold);

    let got = reclaim_for_prompt(&tree, &mut kv, &[], 0, 7);

    assert_eq!(got, PrefixReclaim { target: 7, free: 7 });
    assert_eq!(tree.peek_matched_tokens(&cold, BS, 0), 48);
}
