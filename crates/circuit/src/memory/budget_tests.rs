// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The class's driver terms are read, never defaulted; the inverse queries return
//! the largest fitting value.
//!
//! Owner: metrale-circuit (memory).
//! Invariants: none beyond the types.

use super::*;

const TERMS: &str =
    "[memory]\nutil_ceiling = 0.85\ndriver_fixed_bytes = 1073741824\ndriver_budget_per_mille = 21\nunified = true\n";

#[test]
fn driver_terms_are_read_from_the_class_and_sized_like_the_reserve() {
    let t = DriverTerms::parse("gb10", TERMS).unwrap();
    // 2026-10-02: #72's dense default-tier boot: 103.4 GB budget -> "driver fixed 1024 +
    // driver bookkeeping 2224" MiB.
    let budget = util_budget(130_663_886_848, 0.85);
    assert_eq!(t.bytes(budget, 0) >> 20, 1024 + 2224);
    assert_eq!(t.bytes(budget, 5) - t.bytes(budget, 0), 5);
}

#[test]
fn a_missing_table_or_key_is_refused() {
    let no_table = DriverTerms::parse("gb10", "[hardware]\nname = \"gb10\"\n");
    assert!(
        matches!(&no_table, Err(BudgetError::Terms { detail, .. }) if detail.contains("no [memory]")),
        "{no_table:?}"
    );
    let missing = TERMS.replace("unified = true\n", "");
    assert!(DriverTerms::parse("gb10", &missing).is_err());
    let unknown = format!("{TERMS}surprise = 1\n");
    assert!(DriverTerms::parse("gb10", &unknown).is_err());
    let over = TERMS.replace("= 21", "= 1001");
    assert!(DriverTerms::parse("gb10", &over).is_err());
    let ceiling = TERMS.replace("util_ceiling = 0.85", "util_ceiling = 1.5");
    assert!(DriverTerms::parse("gb10", &ceiling).is_err());
}

#[test]
fn largest_fitting_bisects_to_the_boundary() {
    for limit in [1u64, 2, 7, 64, 99, 100] {
        let mut calls = 0;
        let got = largest_fitting(1, 100, &mut |x| {
            calls += 1;
            Ok(x <= limit)
        })
        .unwrap();
        assert_eq!(got, Some(limit));
        assert!(calls <= 10, "{calls} evaluations for limit {limit}");
    }
    assert_eq!(largest_fitting(5, 100, &mut |x| Ok(x < 5)).unwrap(), None);
    assert!(largest_fitting(9, 8, &mut |_| Ok(true)).is_err());
}
