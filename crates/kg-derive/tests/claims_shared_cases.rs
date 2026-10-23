// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The claim grammar's shared cases (queries part §5 `claims_grammar_shared_cases`):
//! every line of `ledger_gate_cases.tsv` parses to its expected canonical form or refusal word.
//! `tools/ledger.py selftest <this file>` runs the same lines through the Python parser.

use metrale_kg_derive::claims::{canonical, parse};

const CASES: &str = include_str!("ledger_gate_cases.tsv");

#[test]
fn every_shared_case_parses_to_its_expected_form() {
    let mut n = 0;
    for line in CASES.lines().filter(|l| !l.starts_with('#')) {
        let (cell, want) = line
            .rsplit_once('\t')
            .expect("a case line is `cell<TAB>expected`");
        let got = parse(cell).map_or_else(|r| format!("ERR:{r}"), |c| canonical(&c));
        assert_eq!(got, want, "cell `{cell}`");
        n += 1;
    }
    assert!(n >= 30, "the case file lost lines: {n}");
}
