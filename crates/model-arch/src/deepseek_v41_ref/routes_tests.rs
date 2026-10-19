// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The expert-union curve against hand-counted routes.
//!
//! Owner: model-arch, DeepSeek-V4.1 reference.
//! Invariants: none beyond the types.

use super::*;

/// 2026-10-19: One layer, four tokens, top-2 of 8: token t picks {t, t + 1}, so consecutive
/// tokens share one expert.
fn chain() -> Routes {
    Routes::new(1, 4, 2, 8, vec![0, 1, 1, 2, 2, 3, 3, 4]).unwrap()
}

#[test]
fn one_row_reads_topk_experts() {
    assert_eq!(chain().union_mean(1, 0).unwrap(), 2.0);
}

#[test]
fn consecutive_rows_count_a_shared_expert_once() {
    // 2026-10-19: windows of 2 at offsets 0..=2: {0,1,2}, {1,2,3}, {2,3,4}.
    assert_eq!(chain().union_mean(1, 1).unwrap(), 3.0);
    // 2026-10-19: one window of all 4 tokens: {0..=4}.
    assert_eq!(chain().union_mean(1, 3).unwrap(), 5.0);
}

#[test]
fn sequences_are_spaced_and_every_offset_is_averaged() {
    // 2026-10-19: c = 2, one row each, stride 2: offsets 0 -> tokens {0, 2} -> {0,1,2,3} and
    // 1 -> tokens {1, 3} -> {1,2,3,4}; both 4.
    assert_eq!(chain().union_mean(2, 0).unwrap(), 4.0);
    // 2026-10-19: identical routes in every row collapse to topk whatever c and k are.
    let same = Routes::new(1, 4, 2, 8, vec![5, 6, 5, 6, 5, 6, 5, 6]).unwrap();
    assert_eq!(same.union_mean(2, 1).unwrap(), 2.0);
}

#[test]
fn the_mean_runs_over_layers() {
    // 2026-10-19: layer 0 repeats one pair (union 2), layer 1 is the chain (union 3 at k = 1).
    let mut ids = vec![5, 6, 5, 6, 5, 6, 5, 6];
    ids.extend([0, 1, 1, 2, 2, 3, 3, 4]);
    let r = Routes::new(2, 4, 2, 8, ids).unwrap();
    assert_eq!(r.union_mean(1, 1).unwrap(), 2.5);
}

#[test]
fn malformed_routes_and_windows_are_refused() {
    assert!(Routes::new(1, 4, 2, 8, vec![0; 7]).is_err(), "short");
    assert!(
        Routes::new(1, 4, 2, 8, vec![8; 8]).is_err(),
        "id past n_experts"
    );
    assert!(Routes::new(0, 4, 2, 8, vec![]).is_err(), "empty");
    assert!(
        chain().union_mean(3, 1).is_err(),
        "3 windows of 2 in 4 tokens"
    );
    assert!(chain().union_mean(0, 0).is_err(), "no sequences");
}

#[test]
fn uniform_union_matches_the_closed_form() {
    assert_eq!(uniform_union(384, 6, 1), 6.0);
    assert!((uniform_union(384, 6, 16) - 85.53).abs() < 0.01);
    assert!((uniform_union(384, 6, 64) - 243.84).abs() < 0.01);
    assert!(uniform_union(384, 6, 100_000) <= 384.0);
}
