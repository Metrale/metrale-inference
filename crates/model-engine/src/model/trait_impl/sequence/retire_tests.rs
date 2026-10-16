// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: Tests for `retire_insert` on a real `RadixTree`: a multi-rank retire takes no
//! finish-leaf snapshot and still inserts the sequence; a single-rank retire is the insert it
//! always was (snapshot taken and registered, a displaced one freed, a failed save inserted
//! plain).
//!
//! Owner: model-engine (prefix cache).
//! Invariants: none beyond the types.

use std::cell::{Cell, RefCell};

use metrale_cache::radix_tree::RadixTree;
use metrale_telemetry::prefix_cache::{InsertAcquired, PrefixCache};

use super::retire::retire_insert;
use crate::traits::SequenceState;

const BS: usize = 16;

/// 2026-10-16: A retiring 48-token sequence (a 16-token prompt and 32 generated tokens) over
/// blocks 10..13.
fn retiring() -> SequenceState {
    let mut seq = SequenceState::host_only(0);
    seq.tokens = (100..148).collect();
    seq.block_table = vec![10, 11, 12];
    seq.prompt_len = 16;
    seq
}

/// 2026-10-16: Run `retire_insert` with a finish-leaf save that answers `snap`; returns the
/// insert's blocks, how many saves were taken, and the displaced snapshots freed.
fn retire(
    tree: &RadixTree,
    multi_rank: bool,
    snap: Option<usize>,
) -> (InsertAcquired, usize, Vec<usize>) {
    let saves = Cell::new(0);
    let freed = RefCell::new(Vec::new());
    let acquired = retire_insert(
        tree,
        &retiring(),
        BS,
        multi_rank,
        || {
            saves.set(saves.get() + 1);
            snap
        },
        |old| freed.borrow_mut().push(old),
    );
    (acquired, saves.get(), freed.into_inner())
}

/// 2026-10-16: The next turn's prompt: the retired sequence plus four new tokens.
fn next_turn() -> Vec<u32> {
    (100..152).collect()
}

/// 2026-10-16: The blocks a plain insert of the retiring sequence reports on an empty tree.
fn plain_insert_blocks() -> Vec<u32> {
    let seq = retiring();
    RadixTree::new()
        .insert(&seq.tokens, &seq.block_table, &[], BS, seq.prompt_len, 0)
        .blocks
}

#[test]
fn multi_rank_takes_no_finish_leaf_and_still_inserts() {
    let tree = RadixTree::new();
    let (acquired, saves, freed) = retire(&tree, true, Some(7));
    assert_eq!(saves, 0, "no finish-leaf snapshot on a multi-rank world");
    assert!(freed.is_empty());
    assert_eq!(acquired.blocks, plain_insert_blocks());
    assert_eq!(tree.peek_matched_tokens(&next_turn(), BS, 0), 48);
    let m = tree.lookup(&next_turn(), BS, 0, 0);
    assert_eq!((m.matched_tokens, m.ssm_snapshot), (48, None));
}

#[test]
fn single_rank_takes_and_registers_the_finish_leaf() {
    let tree = RadixTree::new();
    let (acquired, saves, freed) = retire(&tree, false, Some(7));
    assert_eq!(saves, 1);
    assert!(freed.is_empty());
    assert_eq!(acquired.blocks, plain_insert_blocks());
    let m = tree.lookup(&next_turn(), BS, 0, 0);
    assert_eq!(
        (m.matched_tokens, m.ssm_snapshot, m.ssm_snapshot_tokens),
        (48, Some(7), 48)
    );
}

#[test]
fn single_rank_frees_the_displaced_finish_leaf() {
    let tree = RadixTree::new();
    let (_, _, first) = retire(&tree, false, Some(5));
    assert!(first.is_empty());
    let (_, saves, freed) = retire(&tree, false, Some(7));
    assert_eq!((saves, freed), (1, vec![5]));
    assert_eq!(tree.lookup(&next_turn(), BS, 0, 0).ssm_snapshot, Some(7));
}

#[test]
fn single_rank_failed_save_inserts_plain() {
    let tree = RadixTree::new();
    let (acquired, saves, freed) = retire(&tree, false, None);
    assert_eq!(saves, 1);
    assert!(freed.is_empty());
    assert_eq!(acquired.blocks, plain_insert_blocks());
    let m = tree.lookup(&next_turn(), BS, 0, 0);
    assert_eq!((m.matched_tokens, m.ssm_snapshot), (48, None));
}
