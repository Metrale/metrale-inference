// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Cell mode: the lever's parse, the cells and segments against the passes of a
//! serve chunked on the grid, and the load-time budget check.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::*;

const GRID: CellGrid = CellGrid {
    first: 8208,
    step: 8192,
};

/// 2026-10-15: Unset is off; two positive counts are taken; anything else is refused.
#[test]
fn parses_and_refuses() {
    assert_eq!(parse(None), Ok(None));
    assert_eq!(parse(Some("8208,8192")), Ok(Some(GRID)));
    assert_eq!(parse(Some("1,1")), Ok(Some(CellGrid { first: 1, step: 1 })));
    for bad in [
        "",
        "8208",
        "8208,",
        ",8192",
        "0,8192",
        "8208,0",
        "8208,8192,1",
        "-1,8192",
        "a,b",
        "8208 ,8192",
    ] {
        assert!(parse(Some(bad)).is_err(), "{bad:?} accepted");
    }
}

/// 2026-10-15: The pipeline lever: unset and `0` are off whatever else is set; `1` is on only
/// with the grid and SP-mHC, without the SP pipeline and the dump. Negative controls: each
/// single missing or conflicting lever flips an accepted `1` to its own error.
#[test]
fn pipeline_parses_and_refuses() {
    let ok = (true, true, false, false);
    let p = |v, (g, m, s, d): (bool, bool, bool, bool)| parse_pipeline(v, g, m, s, d);
    for others in [ok, (false, false, true, true)] {
        assert_eq!(p(None, others), Ok(false));
        assert_eq!(p(Some("0"), others), Ok(false));
    }
    assert_eq!(p(Some("1"), ok), Ok(true));
    for (bad, what) in [
        (
            (false, true, false, false),
            "needs METRALE_GLM_PREFILL_CELL_GRID",
        ),
        ((true, false, false, false), "needs METRALE_GLM_SP_MHC=1"),
        (
            (true, true, true, false),
            "METRALE_GLM_SP_PIPELINE=1 cannot both",
        ),
        ((true, true, false, true), "METRALE_GLM_LAYER_DUMP"),
    ] {
        let e = p(Some("1"), bad).unwrap_err();
        assert!(e.contains(what), "{bad:?}: {e}");
    }
    for v in ["", "2", "on", "true", " 1"] {
        assert!(p(Some(v), ok).is_err(), "{v:?} accepted");
    }
}

/// 2026-10-15: The agreement word: 0 unset; `first` high, `step` low, so grids that differ in
/// either (or swap them) never agree; a count past 32 bits is refused rather than truncated.
#[test]
fn agreement_words_separate_grids() {
    assert_eq!(agreement_value(None).unwrap(), 0);
    assert_eq!(agreement_value(Some(GRID)).unwrap(), (8208u64 << 32) | 8192);
    let swapped = CellGrid {
        first: 8192,
        step: 8208,
    };
    assert_ne!(
        agreement_value(Some(swapped)).unwrap(),
        agreement_value(Some(GRID)).unwrap()
    );
    for wide in [
        CellGrid {
            first: 1 << 32,
            step: 8192,
        },
        CellGrid {
            first: 1,
            step: 1 << 32,
        },
    ] {
        assert!(agreement_value(Some(wide)).is_err(), "{wide:?}");
    }
}

/// 2026-10-15: The idle cold path's chunks of a `total`-token prompt: the first `first` tokens
/// (the idle budget, `max_batch_tokens`), then `budget` each, the last shorter.
fn chunks(total: usize, first: usize, budget: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut t = 0;
    while t < total {
        let n = (if t == 0 { first } else { budget }).min(total - t);
        out.push((t, n));
        t += n;
    }
    out
}

/// 2026-10-16: The cell serve's chunks: chunk 0 proposed `first` rows, later ones `budget`; a
/// non-last end aligned down to a 16-row KV block, then floored to the grid, as
/// `prefill_plan::plan_chunk_len` does with the lever set. Idle (16400, 16384) is a fixed point
/// of the floor; loaded (16384, 16384) starts at the first point.
fn floored_chunks(total: usize, first: usize, budget: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut t = 0;
    while t < total {
        let want = t + if t == 0 { first } else { budget };
        let e = if want >= total {
            total
        } else {
            GRID.floor_end(t, want / 16 * 16)
        };
        out.push((t, e - t));
        t = e;
    }
    out
}

/// 2026-10-16: The scheduler's chunkings: idle and loaded.
const CHUNKINGS: [(usize, usize); 2] = [(16400, 16384), (16384, 16384)];

/// 2026-10-15: Prompt lengths around every boundary of the first four grid points, the c32k
/// bench prompt and a 512k-class prompt.
fn totals() -> Vec<usize> {
    let mut v = vec![1, 2, 100, 8191, 33570, 65536, 131_072, 524_000];
    for g in [8208usize, 16400, 24592, 32784] {
        v.extend([g - 1, g, g + 1, g + 2, g + 16]);
    }
    v
}

/// 2026-10-15: For every prompt length, the cells of the 16384-budget serve's passes, laid end
/// to end, are exactly the passes of the 8192-budget serve (idle first chunks 16400 and 8208).
/// 2026-10-16: idle and loaded, the chunk ends floored; the idle chunking is the floor's fixed
/// point. Negative control: the loaded chunking unfloored (the binary before the floor) is not
/// the 8192 serve's at the c32k prompt.
#[test]
fn cells_are_the_8192_serves_passes() {
    let cells = |ch: Vec<(usize, usize)>| {
        let mut got = Vec::new();
        for (s, n) in ch {
            for (cs, ce) in GRID.cells(s, n) {
                got.push((s + cs, ce - cs));
            }
        }
        got
    };
    for total in totals() {
        let want: Vec<(usize, usize)> = chunks(total, 8208, 8192);
        assert_eq!(
            floored_chunks(total, 16400, 16384),
            chunks(total, 16400, 16384)
        );
        for (first, budget) in CHUNKINGS {
            let got = cells(floored_chunks(total, first, budget));
            assert_eq!(got, want, "total {total} chunking {first},{budget}");
            assert!(got.iter().all(|&(_, n)| n <= 8208), "total {total}");
        }
    }
    assert_ne!(
        cells(chunks(33570, 16384, 16384)),
        chunks(33570, 8208, 8192)
    );
}

/// 2026-10-15: Negative control: the layer's own sub-chunking of the same passes at 8208 rows
/// (what the 16384-budget serve runs without the lever) leaves the grid from the second chunk
/// on, so the arm without the lever must differ.
#[test]
fn sub_chunking_without_the_grid_leaves_it() {
    let mut off = Vec::new();
    for (s, n) in chunks(33570, 16400, 16384) {
        let mut t = 0;
        for k in crate::glm5next_sp_pipeline::sub_chunks(n, 8208) {
            off.push((s + t, k));
            t += k;
        }
    }
    assert_ne!(off, chunks(33570, 8208, 8192));
    assert_eq!(off[2], (16400, 8208));
}

/// 2026-10-15: Grid points and the next point after a position.
#[test]
fn grid_points() {
    assert!(!GRID.is_point(0));
    assert!(GRID.is_point(8208) && GRID.is_point(16400) && GRID.is_point(8208 + 5 * 8192));
    assert!(!GRID.is_point(8192) && !GRID.is_point(16384));
    assert_eq!(GRID.next_after(0), 8208);
    assert_eq!(GRID.next_after(8207), 8208);
    assert_eq!(GRID.next_after(8208), 16400);
    assert_eq!(GRID.next_after(16399), 16400);
    assert_eq!(GRID.cells(0, 0), Vec::<(usize, usize)>::new());
    assert_eq!(GRID.cells(16400, 1), vec![(0, 1)]);
    assert_eq!(GRID.cells(16000, 401), vec![(0, 400), (400, 401)]);
}

type Abs = (usize, usize, Option<u8>, Vec<(usize, u8)>);

/// 2026-10-15: The 8192 serve's segments of a prompt in absolute rows: per pass the capture
/// points strictly inside it, merge (`merge_plan`) or split (one segment per point, a capture after each, as `capture_segments` cuts), as
/// `prefill_segments` plans them.
fn reference(total: usize, merge: bool, abs_points: &[(usize, u8)]) -> Vec<Abs> {
    let mut out = Vec::new();
    for (s, n) in chunks(total, 8208, 8192) {
        let local: Vec<(usize, u8)> = abs_points
            .iter()
            .filter(|&&(p, _)| p > s && p < s + n)
            .map(|&(p, w)| (p - s, w))
            .collect();
        if merge {
            for (a, b, cuts) in crate::glm5next_tail_merge::merge_plan(n, 8208, &local) {
                out.push((s + a, s + b, None, cuts));
            }
        } else {
            let mut a = 0;
            for &(p, w) in &local {
                out.push((s + a, s + p, Some(w), Vec::new()));
                a = p;
            }
            out.push((s + a, s + n, None, Vec::new()));
        }
    }
    out
}

/// 2026-10-15: The cell plan of the 16384-budget serve, in absolute rows, and the boundary
/// captures it adds (points on a pass end of the 8192 serve that fall inside a 16384 pass).
fn cell_plan(
    total: usize,
    merge: bool,
    abs_points: &[(usize, u8)],
    rank0: bool,
) -> (Vec<Abs>, Vec<usize>) {
    let mut out = Vec::new();
    let mut boundary = Vec::new();
    for (s, n) in chunks(total, 16400, 16384) {
        let local: Vec<(usize, u8)> = abs_points
            .iter()
            .filter(|&&(p, _)| p > s && p < s + n)
            .map(|&(p, w)| (p - s, w))
            .collect();
        let segs = plan(GRID, s, n, 8208, merge, &local, rank0);
        let mut covered = 0;
        for (i, g) in segs.iter().enumerate() {
            assert_eq!(g.start, covered, "total {total}");
            assert!(g.end > g.start);
            covered = g.end;
            assert_eq!(
                g.clamp_after,
                GRID.is_point(s + g.end) && g.end < n && !(rank0 && s + g.end == GRID.first)
            );
            assert_eq!(i + 1 == segs.len(), g.end == n);
            if g.after.is_some() && GRID.is_point(s + g.end) {
                boundary.push(s + g.end);
            }
            out.push((s + g.start, s + g.end, g.after, g.cuts.clone()));
        }
        assert_eq!(covered, n);
    }
    (out, boundary)
}

/// 2026-10-15: With capture points anywhere (inside a cell, on a grid point, one row either
/// side of one), in both modes, the cell plan is the 8192 serve's segments, except that a
/// point on a grid point, a pass end there, becomes a capture after the segment ending on it.
#[test]
fn segments_match_the_8192_serve() {
    let point_sets: Vec<Vec<(usize, u8)>> = vec![
        vec![],
        vec![(100, 1)],
        vec![(8000, 0), (8100, 1)],
        vec![(8208, 1)],
        vec![(8207, 1)],
        vec![(8209, 1)],
        vec![(12_000, 0), (16_400, 1)],
        vec![(16_399, 0), (16_401, 1)],
        vec![(24_592, 0), (30_000, 1)],
    ];
    for total in totals() {
        for merge in [false, true] {
            for pts in &point_sets {
                let pts: Vec<(usize, u8)> =
                    pts.iter().copied().filter(|&(p, _)| p < total).collect();
                let want = reference(total, merge, &pts);
                for rank0 in [false, true] {
                    let (got, boundary) = cell_plan(total, merge, &pts, rank0);
                    let want_boundary: Vec<usize> = pts
                        .iter()
                        .map(|p| p.0)
                        .filter(|&p| {
                            GRID.is_point(p)
                                && !chunks(total, 16400, 16384).iter().any(|c| c.0 == p)
                        })
                        .collect();
                    assert_eq!(
                        boundary, want_boundary,
                        "total {total} merge {merge} {pts:?}"
                    );
                    let stripped: Vec<Abs> = got
                        .into_iter()
                        .map(|(a, b, w, c)| (a, b, w.filter(|_| !boundary.contains(&b)), c))
                        .collect();
                    assert_eq!(stripped, want, "total {total} merge {merge} {pts:?}");
                }
            }
        }
    }
}

/// 2026-10-15: The load check: the c32k serve's sizing passes; a first point or step wider than
/// the layer's rows, an idle first chunk off the grid and a budget off the step are refused.
#[test]
fn budget_check() {
    assert_eq!(check_cell_budget(GRID, 8208, 16384, 16400), Ok(()));
    assert_eq!(check_cell_budget(GRID, 8208, 24576, 24592), Ok(()));
    assert_eq!(check_cell_budget(GRID, 16400, 16384, 16400), Ok(()));
    assert!(check_cell_budget(GRID, 8207, 16384, 16400).is_err());
    assert!(
        check_cell_budget(
            CellGrid {
                first: 8208,
                step: 8208
            },
            8192,
            16384,
            16400
        )
        .is_err()
    );
    assert!(check_cell_budget(GRID, 8208, 16384, 16384).is_err());
    assert!(check_cell_budget(GRID, 8208, 8192, 8208).is_err());
    assert!(check_cell_budget(GRID, 8208, 12288, 12304).is_err());
    assert!(check_cell_budget(GRID, 8208, 16000, 16400).is_err());
    assert!(check_cell_budget(GRID, 8208, 8192, 16400).is_err());
}

/// 2026-10-15: The passes from a prefix hit restoring `u` rows: the chunks of the idle cold
/// path from origin 0, the fully cached ones dropped and the one holding `u` started at it.
/// 2026-10-16: of any chunking `ch`.
fn passes_from(ch: Vec<(usize, usize)>, u: usize) -> Vec<(usize, usize)> {
    ch.into_iter()
        .filter(|&(s, n)| s + n > u)
        .map(|(s, n)| (s.max(u), s + n - s.max(u)))
        .collect()
}

/// 2026-10-15: `(abs start, abs end, decode path)` of every segment the 8192 serve runs: a pass
/// the model routes to the decode path whole (`takes_decode_path`), else its segments by the
/// layer's rule (`split_segment_takes_decode_path`, `split` = split mode with several).
fn reference_paths(
    total: usize,
    u: usize,
    merge: bool,
    pts: &[usize],
    td: bool,
    opr: bool,
) -> Vec<(usize, usize, bool)> {
    use metrale_model_layers::layer::{split_segment_takes_decode_path, takes_decode_path};
    let mut out = Vec::new();
    for (s, n) in passes_from(chunks(total, 8208, 8192), u) {
        if takes_decode_path(n, s, opr, true) {
            out.push((s, s + n, true));
            continue;
        }
        let local: Vec<(usize, u8)> = pts
            .iter()
            .filter(|&&p| p > s && p < s + n)
            .map(|&p| (p - s, 0))
            .collect();
        let segs: Vec<(usize, usize)> = if merge {
            crate::glm5next_tail_merge::merge_plan(n, 8208, &local)
                .into_iter()
                .map(|(a, b, _)| (s + a, s + b))
                .collect()
        } else {
            let mut ends: Vec<usize> = local.iter().map(|&(p, _)| s + p).collect();
            ends.push(s + n);
            let mut a = s;
            ends.into_iter()
                .map(|b| {
                    let seg = (a, b);
                    a = b;
                    seg
                })
                .collect()
        };
        let split = !merge && segs.len() > 1;
        for (a, b) in segs {
            out.push((
                a,
                b,
                split_segment_takes_decode_path(b - a, a, split, td, opr),
            ));
        }
    }
    out
}

/// 2026-10-16: `(abs start, abs end, decode path)` of every cell-plan segment of `passes`
/// (rank 0's plan), by [`seg_takes_decode`].
fn cell_paths(
    passes: Vec<(usize, usize)>,
    merge: bool,
    pts: &[usize],
    td: bool,
    opr: bool,
) -> Vec<(usize, usize, bool)> {
    let mut got = Vec::new();
    for (s, n) in passes {
        let local: Vec<(usize, u8)> = pts
            .iter()
            .filter(|&&p| p > s && p < s + n)
            .map(|&p| (p - s, 0))
            .collect();
        for g in plan(GRID, s, n, 8208, merge, &local, true) {
            let d = seg_takes_decode(&g, s + g.start, td, opr);
            got.push((s + g.start, s + g.end, d));
        }
    }
    got
}

/// 2026-10-15: From a cold start or a hit at any length (on, one row before or after a grid
/// point, off the grid), with capture points that leave one-row segments and cells, under every
/// combination of the one-row levers and both tail modes, each cell-plan segment runs the path
/// the 8192 serve runs the same rows on ([`seg_takes_decode`]). 2026-10-16: under the idle and
/// the loaded chunking, chunk ends floored.
#[test]
fn decode_path_matches_the_8192_serve() {
    let mut cases = 0usize;
    for total in [8209usize, 16401, 24593, 24594, 33570] {
        for u in [0usize, 4096, 8207, 8208, 12_000, 16_399, 16_401, 24_591] {
            if u >= total {
                continue;
            }
            for pts in [vec![], vec![total - 1], vec![8207, total - 1], vec![12_000]] {
                let pts: Vec<usize> = pts.into_iter().filter(|&p| p > u && p < total).collect();
                for merge in [false, true] {
                    for (td, opr) in [(false, false), (true, false), (false, true), (true, true)] {
                        let want = reference_paths(total, u, merge, &pts, td, opr);
                        for (first, budget) in CHUNKINGS {
                            let ch = floored_chunks(total, first, budget);
                            let got = cell_paths(passes_from(ch, u), merge, &pts, td, opr);
                            assert_eq!(
                                got, want,
                                "total {total} hit {u} {pts:?} merge {merge} td {td} opr {opr} \
                                 chunking {first},{budget}"
                            );
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(cases > 400);
}

/// 2026-10-15: Negative control: the layer rule alone (every cell routed as a segment of its
/// pass) misses the one-row cell the 8192 serve runs as its own decode pass.
#[test]
fn segment_rule_alone_misses_a_one_row_cell() {
    let segs = plan::<u8>(GRID, 0, 8209, 8208, false, &[], true);
    let last = segs.last().unwrap();
    assert_eq!((last.start, last.end, last.whole_cell), (8208, 8209, true));
    assert!(seg_takes_decode(last, 8208, true, false));
    let as_segment = Seg {
        whole_cell: false,
        ..last.clone()
    };
    assert!(!seg_takes_decode(&as_segment, 8208, true, false));
}
