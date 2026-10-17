// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-17: `codes.lock` and `ErrorCode` agree line for line: a renumbered, renamed or removed
//! code fails here, and a new code fails until its line is added to the lock.

use metrale_error::ErrorCode;

const LOCK: &str = include_str!("../codes.lock");

fn lock_entries() -> Vec<(u16, &'static str)> {
    LOCK.lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let (hex, name) = l
                .split_once(' ')
                .unwrap_or_else(|| panic!("codes.lock line {l:?} is not `0xNNNN Name`"));
            let digits = hex
                .strip_prefix("0x")
                .filter(|d| d.len() == 4)
                .unwrap_or_else(|| panic!("codes.lock code {hex:?} is not 0xNNNN"));
            let code = u16::from_str_radix(digits, 16)
                .unwrap_or_else(|_| panic!("codes.lock code {hex:?} is not hex"));
            assert_eq!(
                format!("0x{code:04x}"),
                hex,
                "codes.lock writes codes as lowercase 0xnnnn"
            );
            assert!(
                !name.contains(' '),
                "codes.lock line {l:?} has more than two fields"
            );
            (code, name)
        })
        .collect()
}

#[test]
fn every_locked_code_keeps_its_number_and_name() {
    for (code, name) in lock_entries() {
        let found = ErrorCode::from_u16(code);
        assert_eq!(
            found.map(ErrorCode::name),
            Some(name),
            "codes.lock records 0x{code:04x} {name}, but ErrorCode has {found:?} there. Codes are \
             append-only: never renumber, rename or remove one. Restore it and add a new code instead."
        );
    }
}

#[test]
fn every_code_is_locked() {
    let locked = lock_entries();
    for c in ErrorCode::ALL {
        assert!(
            locked.contains(&(c.as_u16(), c.name())),
            "ErrorCode::{} (0x{:04x}) is not in crates/error/codes.lock. Append the line `0x{:04x} {}` \
             to codes.lock, keeping the lock in ascending code order.",
            c.name(),
            c.as_u16(),
            c.as_u16(),
            c.name()
        );
    }
}

#[test]
fn lock_is_ascending_without_duplicates() {
    let locked = lock_entries();
    for w in locked.windows(2) {
        assert!(
            w[0].0 < w[1].0,
            "codes.lock is not strictly ascending at {:?} -> {:?}",
            w[0],
            w[1]
        );
    }
    assert_eq!(
        locked.len(),
        ErrorCode::ALL.len(),
        "codes.lock and ErrorCode::ALL differ in length"
    );
}
