// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The swap diff's verdict: a differing record or step fails it, and so does a
//! control that saw nothing.
//!
//! Owner: server CLI (FEATURES workstream).
//! Invariants: none beyond the types.

use super::{SwapComparison, failures};

fn cmp(variant: &str, record: bool, before: bool, after: bool) -> SwapComparison {
    SwapComparison {
        variant: variant.into(),
        record_equal: record,
        record_bytes: 64,
        steps_before_equal: before,
        steps_after_equal: after,
        first_mismatch: None,
    }
}

#[test]
fn a_pass_needs_every_part_equal_and_every_control_different() {
    let ok = cmp("circuit", true, true, true);
    let control = cmp("circuit, record changed", true, true, false);
    assert!(failures(std::slice::from_ref(&ok), std::slice::from_ref(&control)).is_empty());
    for bad in [
        cmp("circuit", false, true, true),
        cmp("circuit", true, false, true),
        cmp("circuit", true, true, false),
    ] {
        assert_eq!(failures(&[bad], std::slice::from_ref(&control)).len(), 1);
    }
    let blind = cmp("circuit, record changed", true, true, true);
    let r = failures(&[ok], &[blind]);
    assert_eq!(r.len(), 1);
    assert!(r[0].contains("saw no difference"), "{r:?}");
}
