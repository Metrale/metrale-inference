// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: Admission-time reclaim of prefix-cache blocks for one prompt, the engine side of
//! `ModelLifecycle::reclaim_prefix_blocks_for`.
//!
//! Owner: model-engine (prefix cache).
//! Invariants: none beyond the types.

use metrale_cache::kv_cache::PagedKvCache;
use metrale_telemetry::prefix_cache::PrefixCache;

use crate::traits::PrefixReclaim;

/// 2026-09-30: Evict least-recently-used prefix-cache leaves until `kv` has the free blocks a
/// prefill of `prompt` allocates. `blocks_needed` counts the whole prompt, headroom included;
/// the full blocks of `prompt`'s cached prefix are subtracted, because the prefill's lookup
/// reuses them, and that prefix is pinned while this runs so it is not evicted to make room for
/// its own request. The pin is dropped before returning; the prefill's lookup takes its own.
/// The SSM restore point is untouched: the lookup still picks the deepest checkpoint within
/// the match, and the match keeps its full length. An empty `prompt` reuses nothing.
pub(crate) fn reclaim_for_prompt(
    cache: &dyn PrefixCache,
    kv: &mut PagedKvCache,
    prompt: &[u32],
    adapter_id: u64,
    blocks_needed: usize,
) -> PrefixReclaim {
    let bs = kv.block_size();
    let pinned = cache.pin_prefix(prompt, bs, adapter_id);
    let target = blocks_needed.saturating_sub(pinned / bs);
    while kv.num_free_blocks() < target {
        let evicted = cache.evict(target - kv.num_free_blocks());
        if evicted.physical.is_empty() {
            break;
        }
        super::block_mgmt::apply_evicted_blocks(evicted, kv);
    }
    if pinned > 0 {
        cache.release_matched(prompt, bs, pinned, adapter_id);
    }
    PrefixReclaim {
        target,
        free: kv.num_free_blocks(),
    }
}

#[cfg(test)]
#[path = "prefix_reclaim_tests.rs"]
mod tests;
