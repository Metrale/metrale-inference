// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The prefill grid cut: the units of a pass against the grid-chunked serve's, the
//! call plan, and the retired lever.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::*;

const G: usize = 8192;

/// 2026-10-14: The units of the rows `start..end`, as absolute `(from, to)` ranges.
fn ranges(start: usize, end: usize, rows: usize, grid: Option<usize>) -> Vec<(usize, usize)> {
    let mut at = start;
    grid_units(start, end - start, rows, grid)
        .unwrap()
        .into_iter()
        .map(|k| {
            at += k;
            (at - k, at)
        })
        .collect()
}

/// 2026-10-14: A prompt run as passes ending at `ends` gives the same units as the same prompt
/// run as one pass per grid cell, for every sub-chunk width: the bits of a grid-chunked serve.
#[test]
fn any_pass_split_on_the_grid_runs_the_grid_serves_units() {
    for rows in [1usize, 16, 2048, 8208, 16400, 40000] {
        for (start, total) in [
            (0usize, 32780usize),
            (19984, 40000),
            (32768, 32780),
            (0, 8200),
        ] {
            let mut want = Vec::new();
            let mut a = start;
            while a < total {
                let b = ((a / G + 1) * G).min(total);
                want.extend(ranges(a, b, rows, None));
                a = b;
            }
            // 2026-10-14: One pass, passes of 2G from the first grid line, and a tail split.
            let one = ranges(start, total, rows, Some(G));
            assert_eq!(one, want, "rows {rows} {start}..{total}");
            let mut cut = Vec::new();
            let mut a = start;
            for b in [2 * G, 4 * G, total - 12, total] {
                if b > a && b <= total {
                    cut.extend(ranges(a, b, rows, Some(G)));
                    a = b;
                }
            }
            let mut want_cut = Vec::new();
            for (s, e) in want {
                if s < total - 12 && total - 12 < e {
                    want_cut.push((s, total - 12));
                    want_cut.push((total - 12, e));
                } else {
                    want_cut.push((s, e));
                }
            }
            assert_eq!(cut, want_cut, "rows {rows} {start}..{total} split");
        }
    }
}

/// 2026-10-14: Without a grid the units are the old sub-chunks; a grid off the KDA slab is an
/// error.
#[test]
fn no_grid_keeps_the_sub_chunks_and_a_bad_grid_is_refused() {
    assert_eq!(grid_units(100, 40, 16, None).unwrap(), vec![16, 16, 8]);
    assert_eq!(grid_units(8190, 10, 16, None).unwrap(), vec![10]);
    assert_eq!(grid_units(8190, 10, 16, Some(G)).unwrap(), vec![2, 8]);
    assert_eq!(grid_units(0, 0, 16, Some(G)).unwrap(), Vec::<usize>::new());
    assert!(grid_units(0, 10, 16, Some(8208)).is_err());
    assert!(grid_units(0, 10, 16, Some(0)).is_err());
}

/// 2026-10-14: The grid lines strictly inside a sequence, relative to its first row.
#[test]
fn grid_lines_fall_on_absolute_multiples() {
    assert!(grid_lines(0, 100, None).is_empty());
    assert!(grid_lines(0, G, Some(G)).is_empty());
    assert_eq!(grid_lines(0, G + 1, Some(G)), vec![G]);
    assert_eq!(grid_lines(G, G, Some(G)), Vec::<usize>::new());
    assert_eq!(grid_lines(G - 2, 2 * G, Some(G)), vec![2, G + 2]);
    assert_eq!(
        grid_lines(19984, 40000 - 19984, Some(G)),
        vec![24576 - 19984, 32768 - 19984]
    );
}

fn plan(units: &[usize], cap: usize, share: impl Fn(usize) -> Option<u8>) -> Vec<Vec<usize>> {
    plan_calls::<u8>(units, cap, |n| Ok(share(n))).unwrap()
}

/// 2026-10-14: The 16384 serve's case: two cells share a call when the sub-chunk holds both;
/// a narrower sub-chunk keeps a call per cell.
#[test]
fn cells_share_a_call_only_within_the_cap() {
    let all = |_: usize| Some(0u8);
    assert_eq!(plan(&[G, G], 16400, all), vec![vec![G, G]]);
    assert_eq!(plan(&[G, 7424], 16400, all), vec![vec![G, 7424]]);
    assert_eq!(plan(&[G, G], 8208, all), vec![vec![G], vec![G]]);
    assert_eq!(plan(&[G, G, G], 16400, all), vec![vec![G, G], vec![G]]);
    assert_eq!(plan(&[100], 16400, all), vec![vec![100]]);
}

/// 2026-10-14: A unit whose decisions differ from the call's first, or that would change the
/// call total's, starts a call of its own; a `None` unit always runs alone.
#[test]
fn differing_decisions_split_the_call() {
    let sp = |n: usize| Some(u8::from(n >= 192));
    assert_eq!(plan(&[G, G, 100], 1 << 20, sp), vec![vec![G, G], vec![100]]);
    let total = |n: usize| Some(u8::from(n > 12000));
    assert_eq!(
        plan(&[G, G, G], 1 << 20, total),
        vec![vec![G], vec![G], vec![G]]
    );
    let refused = |n: usize| (n != 4096).then_some(0u8);
    assert_eq!(
        plan(&[G, G, G, 4096], 1 << 20, refused),
        vec![vec![G, G, G], vec![4096]]
    );
    assert_eq!(
        plan(&[4096, G], 1 << 20, refused),
        vec![vec![4096], vec![G]]
    );
}

/// 2026-10-14: For assorted units, caps and predicates the calls flattened are the units, and
/// no call passes the cap unless it is one unit.
#[test]
fn calls_keep_the_units_in_order_within_the_cap() {
    let preds: [&dyn Fn(usize) -> Option<u8>; 4] = [
        &|_| Some(0),
        &|n| Some(u8::from(n >= 192)),
        &|n| (n % 3 != 0).then_some(1),
        &|_| None,
    ];
    for start in [0usize, 4096, 19984] {
        for n in [1usize, 255, 8192, 16384, 40000] {
            for rows in [16usize, 8192, 16400, 1 << 20] {
                let units = grid_units(start, n, rows, Some(G)).unwrap();
                for pred in preds {
                    let calls = plan(&units, rows, pred);
                    let flat: Vec<usize> = calls.iter().flatten().copied().collect();
                    assert_eq!(flat, units, "start {start} n {n} rows {rows}");
                    for c in &calls {
                        assert!(c.len() == 1 || c.iter().sum::<usize>() <= rows);
                    }
                }
            }
        }
    }
}

/// 2026-10-14: A share error comes back unchanged; a zero unit is refused.
#[test]
fn errors_propagate() {
    let r = plan_calls::<u8>(&[G, G], 1 << 20, |_| Err(anyhow::anyhow!("boom")));
    assert_eq!(r.unwrap_err().to_string(), "boom");
    assert!(plan_calls::<u8>(&[10, 0], 100, |_| Ok(Some(0))).is_err());
}
