// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Cell mode's clamp schedule and hit origin against the 8192-budget serve's, per
//! rank: the state of every KDA layer sees the same restores, computed rows and clamps in the
//! same order, cold and on a prefix hit at any position.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::*;

const GRID: CellGrid = CellGrid {
    first: 8208,
    step: 8192,
};
const CH: Chunking = Chunking {
    first: 16400,
    budget: 16384,
};

/// 2026-10-15: What happens to one layer's SSM state, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ev {
    Restore,
    Compute(usize, usize),
    Clamp,
}

/// 2026-10-15: Deliberate defects of the cell schedule, each of which a test must catch.
#[derive(Clone, Copy, Default)]
struct Fault {
    /// 2026-10-15: Added to [`restored_clamps`]'s count.
    k_delta: isize,
    /// 2026-10-15: Plan rank 0's cells as a worker's (no clamp skip at the first point).
    plan_as_worker: bool,
    /// 2026-10-15: Skip [`rank0_first_chunk_clamp`] (rank 0's chunk 0 never clamps).
    no_first_chunk_clamp: bool,
    /// 2026-10-15: Anchor the grid at the first computed row instead of position 0.
    anchor_at_start: bool,
}

/// 2026-10-15: The scheduler's chunks of a `total`-token prompt: `first`, then `budget` each.
fn chunks(total: usize, first: usize, budget: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut s = 0;
    while s < total {
        let e = (s + if s == 0 { first } else { budget }).min(total);
        out.push((s, e));
        s = e;
    }
    out
}

/// 2026-10-15: The 8192 serve: a restore at chunk 0 on a hit, each chunk computing its rows past
/// `u` as one pass, every chunk clamped except rank 0's chunk 0.
fn grid_serve(total: usize, u: usize, rank0: bool) -> Vec<Ev> {
    let mut ev = Vec::new();
    for (i, (s, e)) in chunks(total, GRID.first, GRID.step).into_iter().enumerate() {
        if i == 0 && u > 0 {
            ev.push(Ev::Restore);
        }
        if e > u {
            ev.push(Ev::Compute(s.max(u), e));
        }
        if !(rank0 && i == 0) {
            ev.push(Ev::Clamp);
        }
    }
    ev
}

/// 2026-10-15: The 16384 serve under the grid, as the model runs it: a fully cached chunk only
/// clamps; the first computed chunk after a restore first takes the restored clamps; a computed
/// chunk runs [`plan`]'s cells with their clamps; every chunk end clamps except rank 0's chunk 0,
/// which clamps per [`rank0_first_chunk_clamp`].
fn cell_serve(total: usize, u: usize, rank0: bool, f: Fault) -> Vec<Ev> {
    let mut ev = Vec::new();
    for (i, (s, e)) in chunks(total, CH.first, CH.budget).into_iter().enumerate() {
        if i == 0 && u > 0 {
            ev.push(Ev::Restore);
        }
        if e > u {
            let start = s.max(u);
            if u > 0 && start == u {
                let k = restored_clamps(GRID, CH, rank0, u).unwrap() as isize + f.k_delta;
                ev.extend(std::iter::repeat_n(Ev::Clamp, k.max(0) as usize));
            }
            let (origin, grid_start) = if f.anchor_at_start {
                (start, 0)
            } else {
                (0, start)
            };
            let segs = plan::<u8>(
                GRID,
                grid_start,
                e - start,
                8208,
                false,
                &[],
                rank0 && !f.plan_as_worker,
            );
            for g in segs {
                let base = origin + grid_start;
                ev.push(Ev::Compute(base + g.start, base + g.end));
                if g.clamp_after {
                    ev.push(Ev::Clamp);
                }
            }
        }
        let first_rule = !f.no_first_chunk_clamp && rank0_first_chunk_clamp(GRID, e);
        if !(rank0 && i == 0) || first_rule {
            ev.push(Ev::Clamp);
        }
    }
    ev
}

/// 2026-10-15: Prompt lengths around the grid's and the chunking's boundaries.
fn totals() -> Vec<usize> {
    let mut v = vec![100, 8000, 33_152, 65_536, 131_072];
    for g in [8208usize, 16400, 24592, 32784, 40976, 49168] {
        v.extend([g - 1, g, g + 1, g + 5]);
    }
    v
}

/// 2026-10-15: Restore positions: cold, off the grid inside every cell and chunk of a 49k
/// prompt, on a grid point that is not a chunk end, on a chunk end, one row either side.
fn hits() -> Vec<usize> {
    let mut v = vec![
        0, 16, 1000, 4096, 8192, 12_000, 20_000, 28_672, 33_136, 45_000,
    ];
    for g in [8208usize, 16400, 24592, 32784, 40976] {
        v.extend([g - 16, g, g + 16, g + 1]);
    }
    v
}

/// 2026-10-15: Every (total, hit, rank) case.
fn cases() -> Vec<(usize, usize, bool)> {
    let mut v = Vec::new();
    for total in totals() {
        for u in hits().into_iter().filter(|&u| u < total) {
            for rank0 in [false, true] {
                v.push((total, u, rank0));
            }
        }
    }
    v
}

/// 2026-10-15: Cold and on every hit, on both ranks, the cell serve's events are the 8192
/// serve's: the same rows computed from the same origin and the same clamps between them.
#[test]
fn schedule_matches_the_8192_serve() {
    for (total, u, rank0) in cases() {
        assert_eq!(
            cell_serve(total, u, rank0, Fault::default()),
            grid_serve(total, u, rank0),
            "total {total} hit {u} rank0 {rank0}"
        );
    }
}

/// 2026-10-15: The restored clamps against hand counts: rank 0 skips its unclamped chunk 0,
/// a worker does not; a hit below the first point takes none.
#[test]
fn restored_clamps_by_hand() {
    for (u, w, r) in [
        (1000, 0, 0),
        (8207, 0, 0),
        (8208, 1, 0),
        (16_399, 1, 0),
        (16_400, 1, 0),
        (24_592, 2, 1),
        (32_783, 2, 1),
        (32_784, 2, 1),
        (40_976, 3, 2),
    ] {
        assert_eq!(restored_clamps(GRID, CH, false, u), Ok(w), "worker {u}");
        assert_eq!(restored_clamps(GRID, CH, true, u), Ok(r), "rank0 {u}");
    }
    let off = Chunking {
        first: 8208,
        budget: 4096,
    };
    assert!(restored_clamps(GRID, off, false, 20_000).is_err());
    assert!(!rank0_first_chunk_clamp(GRID, 8208));
    assert!(rank0_first_chunk_clamp(GRID, 8209));
}

/// 2026-10-15: Negative controls: each defect changes some case. The off-by-one controls run on
/// cases whose count is at least 1 on each rank, so neither can pass by having nothing to count.
#[test]
fn each_schedule_defect_is_caught() {
    let differs = |f: Fault, pick: &dyn Fn(usize, usize, bool) -> bool| {
        cases()
            .into_iter()
            .filter(|&(t, u, r)| pick(t, u, r))
            .any(|(t, u, r)| cell_serve(t, u, r, f) != grid_serve(t, u, r))
    };
    let k_pos = |_: usize, u: usize, r: bool| restored_clamps(GRID, CH, r, u).unwrap() >= 1;
    for rank0 in [false, true] {
        let on_rank = |t: usize, u: usize, r: bool| r == rank0 && k_pos(t, u, r);
        assert!(cases().into_iter().any(|(t, u, r)| on_rank(t, u, r)));
        let more = Fault {
            k_delta: 1,
            ..Fault::default()
        };
        let fewer = Fault {
            k_delta: -1,
            ..Fault::default()
        };
        assert!(differs(more, &on_rank), "k + 1, rank0 {rank0}");
        assert!(differs(fewer, &on_rank), "k - 1, rank0 {rank0}");
    }
    let all = |_: usize, _: usize, _: bool| true;
    for f in [
        Fault {
            plan_as_worker: true,
            ..Fault::default()
        },
        Fault {
            no_first_chunk_clamp: true,
            ..Fault::default()
        },
        Fault {
            anchor_at_start: true,
            ..Fault::default()
        },
    ] {
        assert!(differs(f, &all));
    }
    let off_grid_hit = |_: usize, u: usize, _: bool| u > 0 && !GRID.is_point(u);
    let anchored = Fault {
        anchor_at_start: true,
        ..Fault::default()
    };
    assert!(differs(anchored, &off_grid_hit));
}

/// 2026-10-15: The load checks of the model and the block size.
#[test]
fn model_and_block_checks() {
    assert_eq!(
        check_cell_model("glm5_next", (16, 128, 128), 16, 128),
        Ok(())
    );
    assert_eq!(
        check_cell_model("glm5_next_text", (8, 128, 128), 8, 128),
        Ok(())
    );
    assert!(check_cell_model("qwen3_next", (16, 128, 128), 16, 128).is_err());
    assert!(check_cell_model("glm5_next", (16, 64, 128), 16, 128).is_err());
    assert!(check_cell_model("glm5_next", (32, 128, 128), 16, 128).is_err());
    assert_eq!(check_cell_blocks(GRID, Some(16)), Ok(()));
    assert!(check_cell_blocks(GRID, Some(64)).is_err());
    assert!(check_cell_blocks(GRID, Some(0)).is_err());
    assert!(check_cell_blocks(GRID, None).is_err());
}
