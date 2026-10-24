// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Golden parity with `tools/ledger.py`: each fixture's `.check`, `.tainted` and
//! `.deps` file is ledger.py's own stdout on that fixture (captured 2026-10-23 with the 16-column
//! ledger.py: `gate`, `resolves`). The closures run over the *bundle* (`queries::ledger_links`), not the table, so
//! this is the parity of `kg_impact` itself. The live-file half of the P0 gate is
//! metrale-kg-mcp's `ledger_py_parity_on_the_real_ledger` (runs ledger.py through the host).

use metrale_kg_derive::bundle::Builder;
use metrale_kg_derive::ledger::{self, Table, render_check, render_deps, render_tainted};
use metrale_kg_derive::{ledger_adapter, queries};
use std::collections::BTreeSet;

macro_rules! fixture {
    ($name:literal) => {
        (
            include_str!(concat!("fixtures/ledger/", $name, ".tsv")),
            include_str!(concat!("fixtures/ledger/", $name, ".check")),
            include_str!(concat!("fixtures/ledger/", $name, ".tainted")),
        )
    };
}

fn bundle_of(text: &str) -> metrale_kg_derive::Bundle {
    let mut b = Builder::default();
    ledger_adapter::derive(&mut b, "c", "L.tsv", text, &BTreeSet::new());
    b.seal(Vec::new())
}

#[test]
fn check_output_is_byte_identical_to_ledger_py_on_every_fixture() {
    for (text, check, _) in [
        fixture!("valid"),
        fixture!("cycle"),
        fixture!("unknown"),
        fixture!("fields"),
        fixture!("dupes"),
        fixture!("header"),
        fixture!("gatebad"),
        fixture!("resolvebad"),
        fixture!("resolves"),
    ] {
        assert_eq!(render_check(&Table::parse(text)), check, "fixture:\n{text}");
    }
}

#[test]
fn tainted_over_the_bundle_is_byte_identical_to_ledger_py() {
    let (text, _, tainted) = fixture!("valid");
    let b = bundle_of(text);
    queries::ledger_valid(&b, "L.tsv").unwrap();
    let got = render_tainted(&ledger::tainted(&queries::ledger_links(&b, "c")));
    assert_eq!(got, tainted);
    assert!(got.contains("a\tretracted by z\n") && got.contains("b\tdepends on a\n"));
}

#[test]
fn deps_over_the_bundle_is_byte_identical_to_ledger_py_for_every_id() {
    let (text, _, _) = fixture!("valid");
    let want = include_str!("fixtures/ledger/valid.deps");
    let b = bundle_of(text);
    let links = queries::ledger_links(&b, "c");
    let got: String = ["a", "b", "c", "d", "e", "f", "g", "h", "z", "nope"]
        .iter()
        .map(|id| render_deps(&ledger::deps(&links, id)))
        .collect();
    assert_eq!(got, want);
}

#[test]
fn an_invalid_ledger_refuses_the_closures_like_ledger_py() {
    for (text, _, tainted) in [
        fixture!("cycle"),
        fixture!("unknown"),
        fixture!("fields"),
        fixture!("dupes"),
        fixture!("header"),
    ] {
        assert_eq!(
            tainted, "",
            "ledger.py prints nothing on stdout for an invalid ledger"
        );
        let b = bundle_of(text);
        let e = queries::ledger_valid(&b, "L.tsv").unwrap_err();
        assert_eq!(e.code(), metrale_error::ErrorCode::KgBadLedgerRow);
        assert!(e.to_string().ends_with(ledger::INVALID), "{e}");
        assert!(!b.fatal().is_empty());
    }
}

#[test]
fn a_flipped_depends_on_changes_the_closures_and_they_still_match_the_table() {
    let (text, _, _) = fixture!("valid");
    let flipped = text.replace("\tc,a\t-\t-\t-\n", "\ta,c\t-\t-\t-\n");
    assert_ne!(flipped, text, "the sabotage edit must apply");
    let (b0, b1) = (bundle_of(text), bundle_of(&flipped));
    let (l0, l1) = (
        queries::ledger_links(&b0, "c"),
        queries::ledger_links(&b1, "c"),
    );
    assert_ne!(ledger::deps(&l0, "d"), ledger::deps(&l1, "d"));
    assert_eq!(ledger::deps(&l1, "d"), ["d", "c", "b", "a"]);
    assert_eq!(l1, Table::parse(&flipped).links());
}

#[test]
fn blocked_is_byte_identical_to_ledger_py_and_a_resolved_row_leaves_q1() {
    for (name, text, want) in [
        (
            "resolves",
            include_str!("fixtures/ledger/resolves.tsv"),
            include_str!("fixtures/ledger/resolves.blocked"),
        ),
        (
            "valid",
            include_str!("fixtures/ledger/valid.tsv"),
            include_str!("fixtures/ledger/valid.blocked"),
        ),
    ] {
        assert_eq!(
            ledger::render_blocked(&ledger::blocked(&Table::parse(text))),
            want,
            "{name}"
        );
    }
    let b = bundle_of(include_str!("fixtures/ledger/resolves.tsv"));
    let q1: Vec<String> = queries::blocked_by_gate(&b)
        .into_iter()
        .map(|r| r.0.to_string())
        .collect();
    assert_eq!(q1, ["ledger:b"]);
    let a = metrale_kg_derive::id::KgId::parse("ledger:a").unwrap();
    let by: Vec<String> = b
        .out(&a, metrale_kg_derive::id::Rel::ResolvedBy)
        .map(|e| e.to.to_string())
        .collect();
    assert_eq!(by, ["ledger:c"]);
    assert!(
        !b.errors.iter().any(|f| f.msg.contains("blocker-of-a")),
        "a resolved row gets no synthetic blocker"
    );
}
