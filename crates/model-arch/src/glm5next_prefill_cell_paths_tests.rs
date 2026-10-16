// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: The grid's path admission: the load refusal of the levers that route around the
//! floored chunk path, and the chunks a batched prefill pass may carry under the grid, checked
//! against the cell walk's own hooks (`cells`, `rank0_first_chunk_clamp`, `restored_clamps`).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::super::{rank0_first_chunk_clamp, restored_clamps};
use super::*;

const GRID: CellGrid = CellGrid {
    first: 8208,
    step: 8192,
};

/// 2026-10-16: The load refusal of the levers that route around the floored chunk path: each
/// one alone is refused and named with the refusal text the boot gate greps, the tail
/// checkpoint only without mid-chunk capture, all three together are all named, and none set
/// (or mid-chunk capture alone) loads.
#[test]
fn cell_paths_refuse_the_off_path_levers() {
    let none = CellPathLevers {
        ssm_tail_ckpt: false,
        ssm_tail_midchunk: true,
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
        assert!(
            err.starts_with("METRALE_GLM_PREFILL_CELL_GRID: ")
                && err.ends_with(" route prompts around the grid-floored chunk path"),
            "{err}"
        );
        for (j, name) in names.iter().enumerate() {
            assert_eq!(err.contains(name), i == j, "{levers:?} names {name}: {err}");
        }
    }
    let all = CellPathLevers {
        ssm_tail_ckpt: true,
        ssm_tail_midchunk: false,
        prefill_codispatch: true,
        prefill_varlen: true,
    };
    let err = check_cell_paths(all).expect_err("all three");
    assert!(names.iter().all(|n| err.contains(n)), "{err}");
}

/// 2026-10-16: What the cell walk would do to a chunk the batched pass skips: more than one
/// cell, rank 0's first-chunk clamp, a nonzero restored-clamp count on either rank at any
/// restore point a computing chunk can have (`u < end`), or a later chunk reading its record
/// (the chunk is not the whole prompt).
fn walk_touches(chunk_start: usize, chunk_len: usize, prompt_len: usize) -> bool {
    let end = chunk_start + chunk_len;
    let restores = (chunk_start..end).any(|u| {
        [true, false]
            .into_iter()
            .any(|rank0| restored_clamps(GRID, &[], rank0, u) != Ok(0))
    });
    GRID.cells(chunk_start, chunk_len).len() > 1
        || (chunk_start == 0 && rank0_first_chunk_clamp(GRID, chunk_len))
        || restores
        || end < prompt_len
}

/// 2026-10-16: The batched pass carries exactly the chunks the cell walk leaves alone among the
/// whole-prompt chunks from 0, up to the first point inclusive and not one row past it; a
/// non-last chunk 0 (even one floored to the first point, which rank 0 does not clamp but the
/// batched pass does), a later chunk inside one cell, and any chunk crossing the first point
/// stay on the single-sequence path.
#[test]
fn batched_pass_carries_whole_prompts_inside_the_first_cell() {
    for len in [1, 2, 17, 4096, 8207, 8208] {
        assert!(batched_pass_is_inert(GRID, 0, len, len), "{len}");
        assert!(!walk_touches(0, len, len), "{len}");
    }
    // Firing negative control: one row past the first point, the walk cuts a second cell and
    // rank 0 clamps after chunk 0, and the predicate refuses.
    assert!(walk_touches(0, 8209, 8209));
    assert!(!batched_pass_is_inert(GRID, 0, 8209, 8209));
    let refused = [
        (0, 8208, 20_000),
        (0, 4096, 4097),
        (8208, 8192, 16_400),
        (8208, 100, 8308),
        (0, 16_400, 33_487),
    ];
    for (s, l, p) in refused {
        assert!(!batched_pass_is_inert(GRID, s, l, p), "{s}+{l} of {p}");
    }
    // A restore at or past the first point takes clamps on a worker rank: no inert chunk can
    // reach it.
    assert_eq!(restored_clamps(GRID, &[], false, 8208), Ok(1));
}
