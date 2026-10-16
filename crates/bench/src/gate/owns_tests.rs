// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: `owns`: a BENCH.toml list is read whole or refused whole, and an entry owns the
//! paths under it per component.
//!
//! Owner: bench gate tests.
//! Invariants: none beyond the types.

use std::collections::BTreeSet;

use super::{Owners, from_bench_toml, owners_of};

#[test]
fn a_list_beside_the_benchmarks_parses_and_its_absence_owns_nothing() {
    let text = "owns = [\"crates/config/src/parsers/glm5_next\", \"crates/x.rs\"]\n\
                [[benchmarks]]\nquant = \"nvfp4\"\n";
    assert_eq!(
        from_bench_toml(text),
        Ok(vec![
            "crates/config/src/parsers/glm5_next".to_string(),
            "crates/x.rs".to_string()
        ])
    );
    assert_eq!(from_bench_toml("[[benchmarks]]\n"), Ok(vec![]));
}

#[test]
fn every_entry_that_could_over_or_mis_match_refuses_the_whole_list() {
    for bad in [
        "",
        "/crates/a",
        "crates/a/",
        "crates//a",
        "./crates/a",
        "crates/../kernels/gb10",
        "crates/*",
        "crates/a?",
        "crates/{a,b}",
        "crates\\a",
        "kernels/gb10/common/a.cu",
        "kernels",
    ] {
        let text = format!("owns = [\"crates/ok\", {bad:?}]\n");
        assert!(from_bench_toml(&text).is_err(), "{bad:?} must be refused");
    }
    assert!(
        from_bench_toml("owns = \"crates/a\"\n").is_err(),
        "not a list"
    );
    // 2026-10-16: Control: a path that merely contains `kernels` past its first component is fine.
    assert!(from_bench_toml("owns = [\"crates/kernels/src/a.rs\"]\n").is_ok());
}

#[test]
fn an_entry_owns_the_paths_under_it_per_component() {
    let owners = Owners::from([
        ("a".to_string(), BTreeSet::from(["crates/x".to_string()])),
        (
            "b".to_string(),
            BTreeSet::from(["crates/x/y.rs".to_string(), "docs/b".to_string()]),
        ),
    ]);
    assert_eq!(owners_of("crates/x/z.rs", &owners), BTreeSet::from(["a"]));
    assert_eq!(
        owners_of("crates/x/y.rs", &owners),
        BTreeSet::from(["a", "b"])
    );
    assert_eq!(owners_of("crates/x", &owners), BTreeSet::from(["a"]));
    assert!(owners_of("crates/xy/z.rs", &owners).is_empty());
    assert!(owners_of("crates", &owners).is_empty());
}
