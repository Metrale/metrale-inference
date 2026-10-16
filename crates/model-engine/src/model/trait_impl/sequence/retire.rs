// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: The prefix-cache insert of a retiring sequence (`cache_sequence_dispatch`),
//! with or without the finish-leaf SSM snapshot.
//!
//! Owner: model-engine.
//! Invariants: the insert always runs; the finish-leaf snapshot is taken only on one rank.

use metrale_telemetry::prefix_cache::{InsertAcquired, PrefixCache};

use crate::traits::SequenceState;

/// 2026-10-16: Insert `seq` (prompt and generated tokens) into `cache`, and on a single rank
/// register the finish-leaf snapshot `finish_leaf` takes for its full length; a displaced
/// snapshot goes to `free_displaced`. On a multi-rank world (`multi_rank`) `finish_leaf` is not
/// called: only rank 0 retires through here (workers get the free command alone), so a finish
/// leaf there is one no worker holds. The all-or-nothing snapshot agreement can never restore
/// it, its slots LRU-evict the prompt checkpoints every rank does hold (votes `[0, 192, 192]`),
/// and on a multi-turn prompt a deeper rank-0 leaf disagrees with the workers' prompt
/// checkpoint, so the vote yields `None` and a restore every rank could share is lost.
/// Returns the blocks the cache now references.
pub(super) fn retire_insert(
    cache: &dyn PrefixCache,
    seq: &SequenceState,
    block_size: usize,
    multi_rank: bool,
    finish_leaf: impl FnOnce() -> Option<usize>,
    free_displaced: impl FnOnce(usize),
) -> InsertAcquired {
    let snap = if multi_rank { None } else { finish_leaf() };
    match snap {
        Some(snap_id) => {
            let (displaced, acquired) = cache.insert_with_snapshot(
                &seq.tokens,
                &seq.block_table,
                &seq.disk_block_ids,
                block_size,
                snap_id,
                seq.session_hash,
                seq.prompt_len,
                seq.adapter_id,
            );
            if let Some(old) = displaced {
                free_displaced(old);
            }
            acquired
        }
        None => cache.insert(
            &seq.tokens,
            &seq.block_table,
            &seq.disk_block_ids,
            block_size,
            seq.prompt_len,
            seq.adapter_id,
        ),
    }
}
