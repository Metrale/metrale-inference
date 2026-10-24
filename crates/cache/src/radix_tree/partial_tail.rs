// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Whether the radix walk may match a tail that is not
//! block-aligned (`CacheTuning::prefix_subblock`, `METRALE_PREFIX_SUBBLOCK`).
//! Documentation only since 2026-10-17: the lever reaches
//! `RadixTreeInner::partial_tail_sharing` through `RadixTree::with_tuning`.
//!
//! # Why it is off by default
//!
//! Both sub-block arms of [`super::inner::RadixTreeInner::walk`] end a match
//! inside a KV block and hand out that block's index, so the requester's next
//! K/V rows are written into a block another owner still uses.
//! `PagedKvCache::inc_ref` only counts; nothing copies the block first.
//!
//! * partial-suffix arm: the block is the tail of a cached sequence's prompt,
//!   and that sequence may still be decoding into it.
//! * child-key arm: the block is a full node block that other sequences read
//!   as prompt K/V; the requester writes offsets `[remainder, block_size)`.
//!
//! Two writers that stay in lockstep write the same bytes; once they diverge,
//! readers attend over a mix of both. Measured 2026-08-21 on qwen3.8-27B with
//! DFlash2 at C=4: 0 of 4 correct, with three sequences emitting the same
//! wrong text.
//!
//! Off costs up to `block_size - 1` tokens of prefill recompute per hit.
//! `docs/ROBUSTNESS.md` records that turning the lever off moved 0 of 251
//! agentic samples.
//!
//! Insert still stores the partial tail block in `RadixNode::partial_suffix`
//! and holds a KV ref on it, so `=1` needs no insert-side change; the cost is
//! one block per cached prompt whose length is not block-aligned, held until
//! evicted.
//!
//! Owner: cache.
//! Invariants: unset and `0` mean off.
