// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Range membership: every code lies in its own range and in exactly one range of
//! `CodeRange::ALL`; the ranges are ascending and disjoint; the bounds match the owner-confirmed
//! table in the crate docs.

use metrale_error::{CodeRange, ErrorCode};

#[test]
fn every_code_is_in_its_own_range() {
    for &c in ErrorCode::ALL {
        assert!(
            c.range().contains(c.as_u16()),
            "{} ({:#06x}) is outside its range {:?}",
            c.name(),
            c.as_u16(),
            c.range().bounds(),
        );
    }
}

#[test]
fn every_code_is_in_exactly_one_range() {
    for &c in ErrorCode::ALL {
        let owners = CodeRange::ALL
            .iter()
            .filter(|r| r.contains(c.as_u16()))
            .count();
        assert_eq!(
            owners,
            1,
            "{} ({:#06x}) is contained in {} ranges, expected exactly 1",
            c.name(),
            c.as_u16(),
            owners,
        );
    }
}

#[test]
fn ranges_are_ascending_and_disjoint() {
    let bounds: Vec<(u16, u16)> = CodeRange::ALL.iter().map(|r| r.bounds()).collect();
    for (i, &(lo, hi)) in bounds.iter().enumerate() {
        assert!(lo <= hi, "range {i} is empty: ({lo:#06x}, {hi:#06x})");
        if i > 0 {
            let (_, prev_hi) = bounds[i - 1];
            assert!(
                prev_hi < lo,
                "range {i} ({lo:#06x}..{hi:#06x}) overlaps or follows {prev_hi:#06x} unsorted",
            );
        }
    }
}

#[test]
fn range_bounds_match_the_owner_confirmed_table() {
    let bounds: Vec<(u16, u16)> = CodeRange::ALL.iter().map(|r| r.bounds()).collect();
    assert_eq!(
        bounds,
        vec![
            (0x0000, 0x00FF),
            (0x0100, 0x01FF),
            (0x0200, 0x02FF),
            (0x0300, 0x03FF),
            (0x0400, 0x04FF),
            (0x0500, 0x05FF),
            (0x0600, 0x06FF)
        ],
    );
}
