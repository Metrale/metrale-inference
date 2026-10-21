// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The vocab-parallel LM head's split: which rows of the head this rank projects,
//! and on which calls it may. Moved out of `impl_a3_lm_head.rs` (the split, its tests and
//! `lmhead_vocab_shard`, unchanged but for the `ranks` gate).
//!
//! A sharded head ends in an all-reduce, so it is only correct on a head call that every rank
//! makes. Each head entry (`lm_head`, `lm_head_batched`, `lm_head_project_batched`,
//! `lm_head_fp8_run`) takes a [`HeadRanks`] from its caller, which knows whether the other
//! ranks run the same call; `lm_head_vocab_tp_audit_tests.rs` pins every caller's choice.
//!
//! Owner: model-engine (LM head).
//! Invariants:
//! - `lmhead_vocab_shard(_, HeadRanks::Local)` is `None`: a rank-local head call never issues
//!   a collective.

use super::types::TransformerModel;

/// 2026-10-10: Which ranks make a given LM-head call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HeadRanks {
    /// 2026-10-10: Every rank makes this call, in the same order: it ends a forward whose layers
    /// already ran their collectives on every rank (decode, verify, prefill). The head may
    /// shard by vocab and all-reduce.
    Every,
    /// 2026-10-10: Only this rank makes it (rank 0's prompt-logprob collection, a head run
    /// with no layer before it). The head projects the whole vocab and issues no collective;
    /// the logits are the same bits either way.
    Local,
}

impl TransformerModel {
    /// 2026-09-25: `(begin, len)` rows of the LM head this rank computes, or `None` for the
    /// whole-vocab projection.
    ///
    /// `None` for a [`HeadRanks::Local`] call (2026-10-10: a rank-0-only call that shards waits
    /// forever in the all-reduce), without a communicator of at least 2 ranks, or when
    /// `METRALE_NO_LMHEAD_VOCAB_TP=1` (resolved once into the settings). The BF16 dense-head paths and
    /// (2026-10-09) the FP8 head (`lm_head_fp8_run`) call it. 2026-10-08: the rows split with
    /// `tp_split` in [`LMHEAD_VOCAB_SPLIT_ALIGN`]-row units, so a vocab that does not divide by
    /// the world size (154,880 rows over 3 ranks) still shards; an evenly dividing vocab gets
    /// the same ranges as before. The pieces meet in a zeroed buffer summed by an all-reduce,
    /// so uneven lengths need nothing else, and adding the other ranks' zeros is exact.
    pub(super) fn lmhead_vocab_shard(&self, v: u32, ranks: HeadRanks) -> Option<(usize, usize)> {
        let off = self.settings.kernels.no_lmhead_vocab_tp;
        let comm = self.comm_ref().map(|c| (c.world_size(), c.rank()));
        head_vocab_shard(v as usize, ranks, off, comm)
    }
}

/// 2026-10-10: The decision `lmhead_vocab_shard` makes, from the call's ranks, the kill switch
/// and the communicator's `(world, rank)`.
pub(super) fn head_vocab_shard(
    v: usize,
    ranks: HeadRanks,
    off: bool,
    comm: Option<(usize, usize)>,
) -> Option<(usize, usize)> {
    if ranks == HeadRanks::Local || off {
        return None;
    }
    let (world, rank) = comm?;
    lmhead_vocab_split(v, world, rank)
}

/// 2026-10-08: Row granularity of the vocab-parallel LM head split: the shard starts stay
/// 64-row aligned, as the even splits of every vocab served so far already were.
const LMHEAD_VOCAB_SPLIT_ALIGN: usize = 64;

/// 2026-10-08: `(begin, len)` of `rank`'s rows of a `v`-row head over `world` ranks, or `None`
/// at one rank or when the vocab is not a whole number of split units (then every rank
/// computes the whole head, as before).
fn lmhead_vocab_split(v: usize, world: usize, rank: usize) -> Option<(usize, usize)> {
    if world < 2 {
        return None;
    }
    if v.is_multiple_of(world) {
        // 2026-10-08: The previous rule, byte for byte.
        return Some((rank * (v / world), v / world));
    }
    // 2026-10-09: A vocab that is not a whole number of units (the tokenizer caps GLM-5.3's
    // 154,880-row head to 154,856) splits as if padded to the next unit; the last range stops
    // at `v`. Without this the cap silently sent every rank back to the whole head.
    let padded = v.div_ceil(LMHEAD_VOCAB_SPLIT_ALIGN) * LMHEAD_VOCAB_SPLIT_ALIGN;
    let s = metrale_config::tp_split(padded, world, rank, LMHEAD_VOCAB_SPLIT_ALIGN).ok()?;
    (s.start < v).then(|| (s.start, s.len.min(v - s.start)))
}

#[cfg(test)]
mod vocab_split_tests {
    use super::{HeadRanks, head_vocab_shard, lmhead_vocab_split};

    #[test]
    fn only_an_every_rank_call_shards() {
        for rank in 0..3 {
            let every = head_vocab_shard(154_856, HeadRanks::Every, false, Some((3, rank)));
            assert_eq!(every, lmhead_vocab_split(154_856, 3, rank));
            assert!(every.is_some());
            let local = head_vocab_shard(154_856, HeadRanks::Local, false, Some((3, rank)));
            assert_eq!(
                local, None,
                "a rank-local head call must never shard (rank {rank})"
            );
        }
        assert_eq!(
            head_vocab_shard(154_856, HeadRanks::Every, true, Some((3, 0))),
            None
        );
        assert_eq!(
            head_vocab_shard(154_856, HeadRanks::Every, false, None),
            None
        );
    }

    #[test]
    fn an_even_vocab_keeps_the_even_split() {
        assert_eq!(lmhead_vocab_split(248_320, 2, 1), Some((124_160, 124_160)));
        assert_eq!(lmhead_vocab_split(154_880, 2, 0), Some((0, 77_440)));
        assert_eq!(lmhead_vocab_split(154_880, 1, 0), None);
    }

    #[test]
    fn glm_vocab_over_three_ranks_partitions_in_aligned_pieces() {
        let parts: Vec<_> = (0..3)
            .map(|r| lmhead_vocab_split(154_880, 3, r).unwrap())
            .collect();
        let mut next = 0;
        for (b, l) in &parts {
            assert_eq!(*b, next);
            assert_eq!(b % 64, 0);
            next = b + l;
        }
        assert_eq!(next, 154_880);
        assert_eq!(
            parts,
            vec![(0, 51_648), (51_648, 51_648), (103_296, 51_584)]
        );
    }

    #[test]
    fn a_tokenizer_capped_vocab_still_shards_and_covers_every_row() {
        let parts: Vec<_> = (0..3)
            .map(|r| lmhead_vocab_split(154_856, 3, r).unwrap())
            .collect();
        assert_eq!(
            parts,
            vec![(0, 51_648), (51_648, 51_648), (103_296, 51_560)]
        );
        assert_eq!(lmhead_vocab_split(100, 2, 1), Some((50, 50)));
        let tail: Vec<_> = (0..2)
            .map(|r| lmhead_vocab_split(129, 2, r).unwrap())
            .collect();
        assert_eq!(tail, vec![(0, 128), (128, 1)]);
    }
}

#[cfg(test)]
#[path = "lm_head_vocab_tp_audit_tests.rs"]
mod audit_tests;
