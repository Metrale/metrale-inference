// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Cell mode's clamp schedule and hit origin against the 8192-budget serve's, per
//! rank: the state of every KDA layer sees the same restores, computed rows and clamps in the
//! same order, cold and on a prefix hit at any position. 2026-10-16: under every chunking the
//! scheduler forms (idle, loaded, a mixed tick's shaved budget), its non-last chunk ends
//! floored to the grid.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::*;

const GRID: CellGrid = CellGrid {
    first: 8208,
    step: 8192,
};

/// 2026-10-16: KV block size of the GLM-5.3 serve; `plan_chunk_len` aligns a non-last chunk end
/// down to it before the grid floor.
const BS: usize = 16;

/// 2026-10-16: The scheduler's proposed chunk-0 and later lengths: idle (`max_batch_tokens`
/// 16400, then `--max-prefill-tokens` 16384), loaded (another request is active, so chunk 0 gets
/// 16384 too: `phase_start_prefills`), and a tick whose budget lost three rows.
const IDLE: (usize, usize) = (16400, 16384);
const LOADED: (usize, usize) = (16384, 16384);
const SHAVED: (usize, usize) = (16381, 16381);
const CHUNKINGS: [(usize, usize); 3] = [IDLE, LOADED, SHAVED];

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
    /// 2026-10-16: Leave non-last chunk ends unfloored (the binary before the floor).
    no_floor: bool,
    /// 2026-10-16: Added to every floored non-last chunk end (an off-by-one floor).
    floor_delta: isize,
    /// 2026-10-16: Count the idle chunking's cached ends instead of the walked ones (the
    /// restored-clamps rule before 2026-10-16).
    idle_walk: bool,
}

/// 2026-10-15: The 8192 serve's chunks of a `total`-token prompt: `first`, then `budget` each.
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

/// 2026-10-16: The cell serve's chunks: chunk 0 proposed `p.0` rows, later ones `p.1`; a
/// non-last end aligned down to a KV block, then floored to the grid ([`CellGrid::floor_end`]),
/// as `prefill_plan::plan_chunk_len` does with the lever set.
fn cell_chunks(total: usize, p: (usize, usize), f: Fault) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut s = 0;
    while s < total {
        let want = s + if s == 0 { p.0 } else { p.1 };
        let e = if want >= total {
            total
        } else {
            let a = want / BS * BS;
            let a = if a > s { a } else { want };
            if f.no_floor {
                a
            } else {
                (GRID.floor_end(s, a) as isize + f.floor_delta) as usize
            }
        };
        out.push((s, e));
        s = e;
    }
    out
}

/// 2026-10-16: The ends of the idle chunking's chunks at or below `u` (what the rule before
/// 2026-10-16 assumed a request had walked).
fn idle_walked(total: usize, u: usize) -> Vec<usize> {
    let ch = cell_chunks(total, IDLE, Fault::default());
    ch.into_iter().map(|c| c.1).filter(|&e| e <= u).collect()
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
/// clamps; the first computed chunk after a restore first takes the restored clamps of the
/// chunk ends walked so far; a computed chunk runs [`plan`]'s cells with their clamps; every
/// chunk end clamps except rank 0's chunk 0, which clamps per [`rank0_first_chunk_clamp`].
fn cell_serve(total: usize, u: usize, rank0: bool, p: (usize, usize), f: Fault) -> Vec<Ev> {
    let mut ev = Vec::new();
    let mut walked = Vec::new();
    for (i, (s, e)) in cell_chunks(total, p, f).into_iter().enumerate() {
        if i == 0 && u > 0 {
            ev.push(Ev::Restore);
        }
        if e > u {
            let start = s.max(u);
            if u > 0 && start == u {
                let w = if f.idle_walk {
                    idle_walked(total, u)
                } else {
                    walked.clone()
                };
                let k = restored_clamps(GRID, &w, rank0, u).unwrap() as isize + f.k_delta;
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
        walked.push(e);
    }
    ev
}

/// 2026-10-15: Prompt lengths around the grid's and the chunking's boundaries.
fn totals() -> Vec<usize> {
    let mut v = vec![100, 8000, 33_152, 33_487, 65_536, 131_072];
    for g in [8208usize, 16400, 24592, 32784, 40976, 49168] {
        v.extend([g - 1, g, g + 1, g + 5]);
    }
    v
}

/// 2026-10-15: Restore positions: cold, off the grid inside every cell and chunk of a 49k
/// prompt, on a grid point that is not a chunk end, on a chunk end, one row either side.
fn hits() -> Vec<usize> {
    let mut v = vec![
        0, 16, 1000, 4096, 8192, 12_000, 16_384, 20_000, 28_672, 32_768, 33_136, 45_000,
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

/// 2026-10-15: Cold and on every hit, on both ranks, under every chunking, the cell serve's
/// events are the 8192 serve's: the same rows computed from the same origin and the same
/// clamps between them.
#[test]
fn schedule_matches_the_8192_serve() {
    for p in CHUNKINGS {
        for (total, u, rank0) in cases() {
            assert_eq!(
                cell_serve(total, u, rank0, p, Fault::default()),
                grid_serve(total, u, rank0),
                "chunking {p:?} total {total} hit {u} rank0 {rank0}"
            );
        }
    }
}

/// 2026-10-16: The floor: the idle chunking is a fixed point (its ends are already grid
/// points), a loaded chunk 0 of 16384 rows ends at the first point and the later ones stay on
/// the grid, a budget shaved below 16384 rows moves one point per chunk, a span holding no grid
/// point keeps its end. Off-by-one: 16399 floors to 8208, the point itself is kept, one row
/// past it floors back to it.
#[test]
fn floor_by_hand() {
    let ends = |p| -> Vec<usize> {
        let ch = cell_chunks(41_000, p, Fault::default());
        ch.into_iter().map(|c| c.1).collect()
    };
    assert_eq!(ends(IDLE), [16_400, 32_784, 41_000]);
    let raw = |p| -> Vec<usize> {
        let nf = Fault {
            no_floor: true,
            ..Fault::default()
        };
        cell_chunks(41_000, p, nf)
            .into_iter()
            .map(|c| c.1)
            .collect()
    };
    assert_eq!(raw(IDLE), ends(IDLE));
    assert_eq!(ends(LOADED), [8208, 24_592, 40_976, 41_000]);
    assert_eq!(raw(LOADED), [16_384, 32_768, 41_000]);
    assert_eq!(ends(SHAVED), [8208, 16_400, 24_592, 32_784, 41_000]);
    for (offset, end, want) in [
        (0, 8207, 8207),
        (0, 8208, 8208),
        (0, 8209, 8208),
        (0, 16_399, 8208),
        (0, 16_400, 16_400),
        (8208, 16_399, 16_399),
        (8208, 16_400, 16_400),
        (8208, 24_591, 16_400),
        (16_400, 24_591, 24_591),
    ] {
        assert_eq!(GRID.floor_end(offset, end), want, "({offset}, {end})");
    }
}

/// 2026-10-15: The restored clamps against hand counts: rank 0 skips its unclamped chunk 0,
/// a worker does not; a hit below the first point takes none. 2026-10-16: walked idle ends
/// (16400, 32784) and loaded ones (8208, 24592), whose chunk 0 rank 0 does not clamp.
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
        let walked = idle_walked(41_000, u);
        assert_eq!(
            restored_clamps(GRID, &walked, false, u),
            Ok(w),
            "worker {u}"
        );
        assert_eq!(restored_clamps(GRID, &walked, true, u), Ok(r), "rank0 {u}");
    }
    for (walked, u, w, r) in [
        (&[8208usize][..], 8208, 0, 0),
        (&[8208], 20_000, 1, 1),
        (&[8208, 24_592], 24_592, 1, 1),
        (&[8208, 24_592], 33_472, 2, 2),
    ] {
        assert_eq!(
            restored_clamps(GRID, walked, false, u),
            Ok(w),
            "worker {walked:?} {u}"
        );
        assert_eq!(
            restored_clamps(GRID, walked, true, u),
            Ok(r),
            "rank0 {walked:?} {u}"
        );
    }
    assert!(restored_clamps(GRID, &[8208, 24_592], false, 20_000).is_err());
    assert!(restored_clamps(GRID, &[100, 200, 300], false, 300).is_err());
    assert!(!rank0_first_chunk_clamp(GRID, 8208));
    assert!(rank0_first_chunk_clamp(GRID, 8209));
}

/// 2026-10-15: Negative controls: each defect changes some case. The off-by-one controls run on
/// cases whose count is at least 1 on each rank, so neither can pass by having nothing to count.
/// 2026-10-16: the floor's controls: unfloored, the idle chunking still matches (a fixed point)
/// and the loaded one does not (the binary before the floor: CELLR's loaded run differed); an
/// end one block or one row off the floor fails under both; counting the idle chunking's ends on
/// a loaded hit fails.
#[test]
fn each_schedule_defect_is_caught() {
    let differs = |p: (usize, usize), f: Fault, pick: &dyn Fn(usize, usize, bool) -> bool| {
        cases()
            .into_iter()
            .filter(|&(t, u, r)| pick(t, u, r))
            .any(|(t, u, r)| cell_serve(t, u, r, p, f) != grid_serve(t, u, r))
    };
    let k_pos =
        |t: usize, u: usize, r: bool| restored_clamps(GRID, &idle_walked(t, u), r, u).unwrap() >= 1;
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
        assert!(differs(IDLE, more, &on_rank), "k + 1, rank0 {rank0}");
        assert!(differs(IDLE, fewer, &on_rank), "k - 1, rank0 {rank0}");
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
        assert!(differs(IDLE, f, &all));
    }
    let off_grid_hit = |_: usize, u: usize, _: bool| u > 0 && !GRID.is_point(u);
    let anchored = Fault {
        anchor_at_start: true,
        ..Fault::default()
    };
    assert!(differs(IDLE, anchored, &off_grid_hit));
    let no_floor = Fault {
        no_floor: true,
        ..Fault::default()
    };
    let cold = |_: usize, u: usize, _: bool| u == 0;
    assert!(
        !differs(IDLE, no_floor, &all),
        "the idle chunking is a fixed point"
    );
    assert!(
        differs(LOADED, no_floor, &cold),
        "unfloored loaded chunks leave the grid"
    );
    assert!(differs(SHAVED, no_floor, &cold));
    for d in [-16isize, -1, 1, 16] {
        let off = Fault {
            floor_delta: d,
            ..Fault::default()
        };
        assert!(differs(IDLE, off, &cold), "floor {d:+}, idle");
        assert!(differs(LOADED, off, &cold), "floor {d:+}, loaded");
    }
    let idle_walk = Fault {
        idle_walk: true,
        ..Fault::default()
    };
    let hit = |_: usize, u: usize, _: bool| u > 0;
    assert!(
        !differs(IDLE, idle_walk, &all),
        "the idle walk is the idle chunking's"
    );
    for rank0 in [false, true] {
        let on_rank = |t: usize, u: usize, r: bool| r == rank0 && hit(t, u, r);
        assert!(
            differs(LOADED, idle_walk, &on_rank),
            "idle walk, loaded, rank0 {rank0}"
        );
    }
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

/// 2026-10-16: The load refusal of the levers that route around the floored chunk path: each
/// one alone is refused and named, the tail checkpoint only without mid-chunk capture, all four
/// together are all named, and none set (or mid-chunk capture alone) loads.
#[test]
fn cell_paths_refuse_the_off_path_levers() {
    let none = CellPathLevers {
        ssm_tail_ckpt: false,
        ssm_tail_midchunk: true,
        ep_prefill_batch: false,
        prefill_codispatch: false,
        prefill_varlen: false,
    };
    assert_eq!(check_cell_paths(none), Ok(()));
    let midchunk_ckpt = CellPathLevers {
        ssm_tail_ckpt: true,
        ..none
    };
    assert_eq!(check_cell_paths(midchunk_ckpt), Ok(()));
    let no_midchunk = CellPathLevers {
        ssm_tail_midchunk: false,
        ..none
    };
    assert_eq!(check_cell_paths(no_midchunk), Ok(()));
    let names = [
        "METRALE_SSM_TAIL_CKPT=1",
        "METRALE_EP_PREFILL_BATCH=1",
        "METRALE_PREFILL_CODISPATCH",
        "METRALE_PREFILL_VARLEN",
    ];
    let alone = [
        CellPathLevers {
            ssm_tail_ckpt: true,
            ssm_tail_midchunk: false,
            ..none
        },
        CellPathLevers {
            ep_prefill_batch: true,
            ..none
        },
        CellPathLevers {
            prefill_codispatch: true,
            ..none
        },
        CellPathLevers {
            prefill_varlen: true,
            ..none
        },
    ];
    for (i, levers) in alone.iter().enumerate() {
        let err = check_cell_paths(*levers).expect_err(names[i]);
        for (j, name) in names.iter().enumerate() {
            assert_eq!(err.contains(name), i == j, "{levers:?} names {name}: {err}");
        }
    }
    let all = CellPathLevers {
        ssm_tail_ckpt: true,
        ssm_tail_midchunk: false,
        ep_prefill_batch: true,
        prefill_codispatch: true,
        prefill_varlen: true,
    };
    let err = check_cell_paths(all).expect_err("all four");
    assert!(names.iter().all(|n| err.contains(n)), "{err}");
}
