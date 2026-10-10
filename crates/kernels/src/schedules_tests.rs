// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `schedules::lookup_in` over a hand-built table: the default/opt-in split and the
//! inclusive row ranges.
//!
//! Owner: kernels crate tests.
//! Invariants: none beyond the types.

use super::{Cell, Enabled, Numerics, Schedule, lookup_in};

const fn entry(rows_lo: u32, rows_hi: u32, kernel: &'static str, enabled: Enabled) -> Schedule {
    Schedule {
        op: "linear",
        weight: "nvfp4/g16",
        activation: "bf16",
        k: 5120,
        n: 17408,
        rows_lo,
        rows_hi,
        kernel,
        family: "w4a16_tc",
        default: Some("w4a16_gemv::w4a16_gemv_sw"),
        numerics: match enabled {
            Enabled::Default => Numerics::BitIdentical,
            Enabled::OptIn => Numerics::Differs,
        },
        enabled,
    }
}

static TABLE: &[Schedule] = &[
    entry(1, 4, "w4a16_gemv_tc::tc4", Enabled::Default),
    entry(5, 8, "w4a16_gemv_tc::tc8", Enabled::OptIn),
];

const CELL: Cell<'static> = Cell {
    op: "linear",
    weight: "nvfp4/g16",
    activation: "bf16",
    k: 5120,
    n: 17408,
};

fn kernel(rows: u32, opt_in: bool) -> Option<&'static str> {
    lookup_in(TABLE, &CELL, rows, opt_in).map(|s| s.kernel)
}

#[test]
fn a_default_entry_answers_with_or_without_the_opt_in() {
    assert_eq!(kernel(2, false), Some("w4a16_gemv_tc::tc4"));
    assert_eq!(kernel(2, true), Some("w4a16_gemv_tc::tc4"));
}

#[test]
fn an_opt_in_entry_answers_only_when_opted_in() {
    assert_eq!(kernel(6, false), None);
    assert_eq!(kernel(6, true), Some("w4a16_gemv_tc::tc8"));
}

#[test]
fn row_ranges_are_inclusive_at_both_ends_and_closed_outside() {
    assert_eq!(kernel(1, false), Some("w4a16_gemv_tc::tc4"));
    assert_eq!(kernel(4, false), Some("w4a16_gemv_tc::tc4"));
    assert_eq!(kernel(5, true), Some("w4a16_gemv_tc::tc8"));
    assert_eq!(kernel(8, true), Some("w4a16_gemv_tc::tc8"));
    assert_eq!(kernel(0, true), None);
    assert_eq!(kernel(9, true), None);
}

#[test]
fn every_key_field_must_match() {
    let misses = [
        Cell {
            op: "lm_head",
            ..CELL
        },
        Cell {
            weight: "fp8",
            ..CELL
        },
        Cell {
            activation: "fp8",
            ..CELL
        },
        Cell { k: 4096, ..CELL },
        Cell { n: 4096, ..CELL },
    ];
    for c in misses {
        assert_eq!(lookup_in(TABLE, &c, 2, true), None, "{c:?}");
    }
}
