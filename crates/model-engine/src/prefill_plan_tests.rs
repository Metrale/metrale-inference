// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: Tests for `prefill_plan`: the chunk planner, the tail split point, the
//! chunk-end snapshot rule, and a cold-then-warm replay through the radix prefix cache.
//!
//! Owner: model-engine prefill (SSM prefix cache).
//! Invariants: none beyond the types.

use super::*;
use metrale_cache::radix_tree::RadixTree;
use metrale_telemetry::prefix_cache::PrefixCache;

const BS: usize = 16;

/// 2026-09-27: The chunks a prompt runs in, as the scheduler plans them and
/// `prefill_chunk_dispatch` splits a last chunk: chunk 0 proposes `first` tokens (a
/// dense serve's idle `max_batch_tokens`), later chunks `cont`.
fn run_chunks(total: usize, first: usize, cont: usize) -> Vec<(usize, usize)> {
    let split = tail_split_point(total, BS);
    let mut out = Vec::new();
    let mut off = 0;
    while off < total {
        let cap = if off == 0 { first } else { cont };
        let len = plan_chunk_len(off, total, (total - off).min(cap), Some(BS), split, None);
        assert!(len >= 1, "total={total} off={off}: empty chunk");
        let end = off + len;
        match split {
            Some(cut) if end == total && off < cut && cut < end => {
                out.push((off, cut));
                out.push((cut, end));
            }
            _ => out.push((off, end)),
        }
        off = end;
    }
    out
}

/// 2026-09-27: Cold prefill then the byte-identical warm lookup through a real
/// `RadixTree`: every non-last chunk end that `is_checkpoint_chunk_end` accepts is
/// inserted as an intermediate snapshot, and the full prompt's complete blocks as the
/// last chunk's insert. Returns `(matched_tokens, snapshot_tokens)` of the warm lookup.
fn cold_then_warm(total: usize, first: usize, cont: usize, interval: usize) -> (usize, usize) {
    let tokens: Vec<u32> = (0..total as u32)
        .map(|i| 1000 + (i * 7919) % 150_000)
        .collect();
    let blocks: Vec<u32> = (0..total.div_ceil(BS) as u32).collect();
    let cache = RadixTree::new();
    let chunks = run_chunks(total, first, cont);
    let mut snap = 0;
    for &(_, end) in &chunks[..chunks.len() - 1] {
        if is_checkpoint_chunk_end(end, total, BS, interval) {
            let eb = end / BS;
            cache.insert(&tokens[..end], &blocks[..eb], &[], BS, end, 0);
            cache.insert_intermediate_snapshot(
                &tokens[..end],
                &blocks[..eb],
                &[],
                BS,
                snap,
                0,
                end,
                0,
            );
            snap += 1;
        }
    }
    cache.insert(&tokens, &blocks[..total / BS], &[], BS, 0, 0);
    let m = cache.lookup(&tokens, BS, 0, 0);
    assert!(
        m.ssm_snapshot.is_some(),
        "total={total}: warm lookup found no snapshot"
    );
    (m.matched_tokens, m.ssm_snapshot_tokens)
}

#[test]
fn warm_32k_dense_restores_at_the_tail_split_point() {
    // 2026-09-27: The high-ISL fixture renders to 32772 tokens; a dense serve runs chunk 0
    // at 8192 + max_batch_size 1 and later chunks at 8192, with interval 16. Before the
    // planner ended chunks on block boundaries the warm lookup restored at 24577.
    let (matched, snap) = cold_then_warm(32772, 8193, 8192, 16);
    assert_eq!(matched, 32768);
    assert_eq!(snap, tail_split_point(32772, BS).unwrap());
    assert_eq!(snap, 32752);
}

#[test]
fn warm_restores_within_two_blocks_of_the_match_for_every_length() {
    // 2026-09-27: Small chunks so that every length from 2 blocks to 40 chunks is covered,
    // with an unaligned chunk 0 (the `+ max_batch_size` case) and interval snapshots off,
    // so only the tail rule can place the snapshot.
    for (first, cont) in [(67, 64), (65, 64), (72, 64), (64, 64), (130, 96)] {
        for total in (2 * BS + 1)..=2600 {
            let (matched, snap) = cold_then_warm(total, first, cont, 0);
            let cut = tail_split_point(total, BS).unwrap();
            assert!(
                snap >= cut && snap <= matched,
                "total={total} first={first}: snap {snap}, cut {cut}, matched {matched}"
            );
        }
    }
}

#[test]
fn chunks_tile_the_prompt_and_end_on_block_boundaries() {
    for (first, cont) in [(8193, 8192), (8200, 8192), (8191, 8191), (100, 50)] {
        for total in [
            1, 3, 16, 17, 33, 8192, 8193, 8200, 16385, 32768, 32772, 40000,
        ] {
            let chunks = run_chunks(total, first, cont);
            assert_eq!(chunks[0].0, 0);
            assert_eq!(chunks.last().unwrap().1, total);
            for w in chunks.windows(2) {
                assert_eq!(w[0].1, w[1].0, "total={total}: gap or overlap");
            }
            for &(s, e) in &chunks[..chunks.len() - 1] {
                assert!(e > s);
                assert_eq!(
                    e % BS,
                    0,
                    "total={total} first={first}: non-last end {e} off-block"
                );
            }
        }
    }
}

#[test]
fn plan_keeps_a_last_chunk_and_never_grows_a_chunk() {
    assert_eq!(
        plan_chunk_len(0, 300, 300, Some(BS), tail_split_point(300, BS), None),
        300
    );
    assert_eq!(plan_chunk_len(8192, 8195, 3, Some(BS), None, None), 3);
    assert_eq!(
        plan_chunk_len(0, 32772, 8193, Some(BS), Some(32752), None),
        8192
    );
    assert_eq!(
        plan_chunk_len(24576, 32772, 8192, Some(BS), Some(32752), None),
        8176
    );
    assert_eq!(plan_chunk_len(0, 100, 0, Some(BS), None, None), 0);
    // 2026-09-27: A proposal shorter than one block keeps its length, rounded to 4.
    assert_eq!(plan_chunk_len(0, 100, 10, Some(BS), None, None), 8);
    assert_eq!(plan_chunk_len(0, 100, 3, Some(BS), None, None), 3);
    // 2026-09-27: Without a block size the old WY4 rounding is all that applies.
    assert_eq!(plan_chunk_len(0, 32772, 8193, None, None, None), 8192);
    assert_eq!(plan_chunk_len(8193, 32772, 8191, None, None, None), 8188);
}

#[test]
fn tail_split_point_matches_the_dispatch_formula() {
    assert_eq!(tail_split_point(16, BS), None);
    assert_eq!(tail_split_point(32, BS), None);
    assert_eq!(tail_split_point(33, BS), Some(16));
    assert_eq!(tail_split_point(48, BS), Some(16));
    assert_eq!(tail_split_point(49, BS), Some(32));
    assert_eq!(tail_split_point(32772, BS), Some(32752));
    assert_eq!(tail_split_point(32768, BS), Some(32736));
    assert_eq!(tail_split_point(100, 0), None);
}

#[test]
fn checkpoint_chunk_end_rule() {
    // 2026-09-27: Prompt-tail ends (the last boundary under the prompt end and one block
    // below it) save whatever the interval; other ends need an interval-multiple block.
    assert!(is_checkpoint_chunk_end(32768, 32772, BS, 0));
    assert!(is_checkpoint_chunk_end(32752, 32772, BS, 0));
    assert!(!is_checkpoint_chunk_end(32736, 32772, BS, 0));
    assert!(is_checkpoint_chunk_end(8192, 32772, BS, 16));
    assert!(!is_checkpoint_chunk_end(8176, 32772, BS, 16));
    assert!(!is_checkpoint_chunk_end(8, 32772, BS, 16));
    assert!(!is_checkpoint_chunk_end(8192, 32772, 0, 16));
}

/// 2026-10-12: A whole-pass-splitting model (GLM-5.3) splits at the tail boundary, where a
/// byte-identical warm repeat's match lands, so it replays only the last partial block; other
/// models keep the split one block lower. Both are `None` where the prompt has no boundary.
#[test]
fn a_whole_pass_model_splits_at_the_tail_boundary() {
    for total in [17usize, 33, 48, 49, 268, 1036, 4108, 32772] {
        let tb = metrale_gpu_runtime::ssm_tail_boundary(total, BS);
        assert_eq!(tail_split_for(total, BS, true), tb, "{total}");
        assert_eq!(
            tail_split_for(total, BS, false),
            tail_split_point(total, BS),
            "{total}"
        );
    }
    assert_eq!(tail_split_for(268, BS, true), Some(256));
    assert_eq!(tail_split_for(268, BS, false), Some(240));
    assert_eq!(tail_split_for(4108, BS, true), Some(4096));
    assert_eq!(tail_split_for(16, BS, true), None);
}

const G: usize = PREFILL_GRID;

/// 2026-10-14: The passes a grid model (whole-pass split, GLM-5.3) runs for the rows
/// `start..total` at prefill budget `budget`: chunks of at most `budget`, a last chunk split
/// at the tail split point.
fn grid_passes(start: usize, total: usize, budget: usize) -> Vec<(usize, usize)> {
    let split = tail_split_for(total, BS, true);
    let mut out = Vec::new();
    let mut off = start;
    while off < total {
        let proposed = (total - off).min(budget);
        let len = plan_chunk_len(off, total, proposed, Some(BS), split, Some(G));
        assert!(
            len >= 1 && len <= proposed,
            "total={total} off={off}: len {len}"
        );
        let end = off + len;
        match split {
            Some(cut) if end == total && off < cut && cut < end => {
                out.push((off, cut));
                out.push((cut, end));
            }
            _ => out.push((off, end)),
        }
        off = end;
    }
    out
}

/// 2026-10-14: The passes cut again at every grid line: the units a grid layer runs.
fn cells(passes: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for &(s, e) in passes {
        let mut a = s;
        while a < e {
            let b = ((a / G + 1) * G).min(e);
            out.push((a, b));
            a = b;
        }
    }
    out
}

#[test]
fn every_grid_budget_runs_the_same_cells() {
    // 2026-10-14: The gate's lengths plus grid-adjacent ones, cold (start 0) and from warm
    // restore points on and off the grid.
    for total in [
        1024usize, 8191, 8192, 8193, 8200, 16384, 20000, 24580, 32768, 32780, 40000, 65536,
    ] {
        for start in [0usize, 4096, 19984, 32768] {
            if start >= total {
                continue;
            }
            let base = cells(&grid_passes(start, total, G));
            for budget in [2 * G, 3 * G, 8 * G] {
                assert_eq!(
                    cells(&grid_passes(start, total, budget)),
                    base,
                    "total={total} start={start} budget={budget}"
                );
            }
            // 2026-10-14: At budget G a pass is one cell, and every pass but the tail split
            // and the last ends on a grid line.
            let passes = grid_passes(start, total, G);
            assert_eq!(cells(&passes), passes, "total={total} start={start}");
            let split = tail_split_for(total, BS, true);
            for &(_, e) in &passes[..passes.len() - 1] {
                assert!(
                    e % G == 0 || Some(e) == split,
                    "total={total} start={start}: pass end {e} off the grid"
                );
            }
        }
    }
}

#[test]
fn an_idle_chunk_0_ends_on_the_grid_line() {
    // 2026-10-14: The idle serve's `max_batch_tokens` proposal (8192 + max batch 16) and the
    // busy serve's 8192 give the same chunk 0; without a grid the idle chunk ran 8208 rows.
    let split = tail_split_for(32780, BS, true);
    assert_eq!(plan_chunk_len(0, 32780, 8208, Some(BS), split, Some(G)), G);
    assert_eq!(plan_chunk_len(0, 32780, G, Some(BS), split, Some(G)), G);
    assert_eq!(plan_chunk_len(0, 32780, 8208, Some(BS), split, None), 8208);
    // 2026-10-14: From an off-grid restore the chunk runs to the next grid line it reaches.
    assert_eq!(
        plan_chunk_len(19984, 65536, 2 * G, Some(BS), None, Some(G)),
        4 * G - 19984
    );
    // 2026-10-14: A proposal that reaches no grid line keeps the block rule.
    assert_eq!(
        plan_chunk_len(0, 32780, 4000, Some(BS), None, Some(G)),
        4000
    );
}

#[test]
fn a_budget_off_the_grid_is_refused() {
    assert_eq!(check_prefill_grid(G, G), Ok(()));
    assert_eq!(check_prefill_grid(2 * G, G), Ok(()));
    for bad in [0usize, 16, 4096, 8208, 16400, 3 * G / 2] {
        let err = check_prefill_grid(bad, G).unwrap_err();
        assert!(err.contains(&format!("{bad}")), "{bad}: {err}");
    }
    assert!(check_prefill_grid(G, 0).is_err());
}
