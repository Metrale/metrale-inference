// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The registry is a bijection between `ErrorCode` and its `u16` discriminant:
//! round-tripping, a full scan of the `u16` space, and no duplicate codes or names.

use std::collections::HashSet;

use metrale_error::ErrorCode;

#[test]
fn from_u16_round_trips_every_code() {
    for &c in ErrorCode::ALL {
        assert_eq!(
            ErrorCode::from_u16(c.as_u16()),
            Some(c),
            "code {}",
            c.as_u16()
        );
    }
}

#[test]
fn u16_space_hits_exactly_the_registered_codes() {
    let mut hits = 0;
    for n in 0..=u16::MAX {
        match ErrorCode::from_u16(n) {
            Some(c) => {
                hits += 1;
                assert_eq!(c.as_u16(), n, "from_u16({n}) returned a different code");
            }
            None => assert!(
                ErrorCode::ALL.iter().all(|c| c.as_u16() != n),
                "code {n} is registered but from_u16 returned None",
            ),
        }
    }
    assert_eq!(hits, ErrorCode::ALL.len());
}

#[test]
fn codes_and_names_are_unique() {
    let codes: HashSet<u16> = ErrorCode::ALL.iter().map(|c| c.as_u16()).collect();
    assert_eq!(codes.len(), ErrorCode::ALL.len(), "duplicate codes in ALL");
    let names: HashSet<&str> = ErrorCode::ALL.iter().map(|c| c.name()).collect();
    assert_eq!(names.len(), ErrorCode::ALL.len(), "duplicate names in ALL");
}

#[test]
fn all_is_strictly_ascending_by_code() {
    let codes: Vec<u16> = ErrorCode::ALL.iter().map(|c| c.as_u16()).collect();
    for pair in codes.windows(2) {
        assert!(pair[0] < pair[1], "ALL not ascending at {}", pair[0]);
    }
}
