// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Tests for `tail_exact::pass_takes_exact_gdn`, on the passes a 32772-token
//! prompt with 16-token blocks runs cold (two passes, split at 32752) and warm (one
//! replay from the snapshot at 32752).

use super::tail_exact::pass_takes_exact_gdn;
use crate::prefill_plan::{plan_chunk_len, tail_split_point};

const TOTAL: usize = 32772;
const BS: usize = 16;

#[test]
fn cold_tail_pass_matches_the_warm_replay_arm() {
    let split = tail_split_point(TOTAL, BS);
    assert_eq!(split, Some(32752));
    // 2026-10-14: The cold tail pass and the warm replay both start at the split.
    let cold_tail = pass_takes_exact_gdn(false, true, 32752, split);
    let warm_replay = pass_takes_exact_gdn(true, true, 32752, split);
    assert!(cold_tail && warm_replay);
}

#[test]
fn passes_below_the_split_keep_the_fla_arm() {
    let split = tail_split_point(TOTAL, BS);
    // 2026-10-14: Every cold chunk start the 4096-token scheduler produces before the
    // split, chunk 0 included.
    let mut offset = 0;
    while offset < split.unwrap() {
        assert!(
            !pass_takes_exact_gdn(false, true, offset, split),
            "offset {offset}"
        );
        offset += plan_chunk_len(offset, TOTAL, 4096, Some(BS), split);
    }
    assert_eq!(Some(offset), split);
}

#[test]
fn kill_switch_restores_the_fla_arm_on_the_cold_tail() {
    let split = tail_split_point(TOTAL, BS);
    assert!(!pass_takes_exact_gdn(false, false, 32752, split));
    // 2026-10-14: A restore still takes the exact arm with the switch off.
    assert!(pass_takes_exact_gdn(true, false, 32752, split));
}

#[test]
fn no_split_means_no_tail_pass() {
    assert!(!pass_takes_exact_gdn(false, true, 32752, None));
}
