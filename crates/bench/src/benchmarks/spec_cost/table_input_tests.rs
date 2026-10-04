// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Tests for reading a spec-cost record back: records are written with the real
//! writer (`cell::record`), so a key renamed on one side fails here.
//!
//! Owner: bench, spec-cost.
//! Invariants: none beyond the types.

use super::super::cell::{CellVerdict, KEY_K, StepCost, record};
use super::*;

fn cost(wall_ms: f64, draft_ms: f64, joules: Option<(f64, f64)>) -> StepCost {
    StepCost {
        steps: 100.0,
        wall_ms,
        step_ms: wall_ms - 0.5,
        verify_ms: wall_ms - 0.5 - draft_ms,
        draft_ms,
        verify_j: joules.map(|j| j.0),
        draft_j: joules.map(|j| j.1),
        smi_j: None,
        tok_per_step: 2.0,
    }
}

fn written(k: f64, rows: &[(usize, CellVerdict)]) -> BTreeMap<String, f64> {
    let mut m = BTreeMap::new();
    m.insert(KEY_K.to_string(), k);
    for (n, v) in rows {
        record(*n, v, &mut m);
    }
    m
}

#[test]
fn reads_wall_time_split_into_draft_and_the_rest_in_numeric_width_order() {
    // 2026-10-04: Widths 16 and 2 sort "n16" before "n2" as strings; the read order is numeric.
    let m = written(
        2.0,
        &[
            (16, CellVerdict::Measured(cost(40.0, 3.0, Some((1.5, 0.1))))),
            (2, CellVerdict::Measured(cost(30.0, 2.5, Some((1.0, 0.09))))),
        ],
    );
    let cells = read(&m).unwrap();
    assert_eq!(
        cells,
        vec![
            MeasuredCell {
                n: 2,
                k: 2,
                verify_ms: 27.5,
                verify_j: 1.0,
                draft_ms: 2.5,
                draft_j: 0.09
            },
            MeasuredCell {
                n: 16,
                k: 2,
                verify_ms: 37.0,
                verify_j: 1.5,
                draft_ms: 3.0,
                draft_j: 0.1
            },
        ]
    );
}

#[test]
fn a_vacuous_width_refuses_the_record() {
    let m = written(
        1.0,
        &[
            (1, CellVerdict::Measured(cost(20.0, 2.0, Some((0.7, 0.07))))),
            (4, CellVerdict::Vacuous("ended early".into())),
        ],
    );
    let err = read(&m).unwrap_err();
    assert!(format!("{err:#}").contains("width 4 is vacuous"), "{err:#}");
}

#[test]
fn a_width_without_joules_refuses_the_record() {
    let m = written(1.0, &[(1, CellVerdict::Measured(cost(20.0, 2.0, None)))]);
    let err = read(&m).unwrap_err();
    assert!(format!("{err:#}").contains("has no verify_j"), "{err:#}");
}

#[test]
fn k_must_be_present_and_a_whole_count() {
    let mut m = written(
        1.0,
        &[(1, CellVerdict::Measured(cost(20.0, 2.0, Some((0.7, 0.07)))))],
    );
    m.insert(KEY_K.to_string(), 1.5);
    assert!(format!("{:#}", read(&m).unwrap_err()).contains("not a draft count"));
    m.remove(KEY_K);
    assert!(format!("{:#}", read(&m).unwrap_err()).contains("no `k`"));
}

#[test]
fn a_record_without_widths_is_refused() {
    let err = read(&written(0.0, &[])).unwrap_err();
    assert!(format!("{err:#}").contains("no width"), "{err:#}");
}

#[test]
fn a_draft_share_above_the_step_is_refused() {
    let m = written(
        1.0,
        &[(1, CellVerdict::Measured(cost(2.0, 3.0, Some((0.7, 0.07)))))],
    );
    let err = read(&m).unwrap_err();
    assert!(
        format!("{err:#}").contains("drafts 3 ms of a 2 ms step"),
        "{err:#}"
    );
}
