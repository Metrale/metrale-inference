// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The spec parser accepts the full form, round-trips through its canonical text,
//! and refuses a missing key, an unknown key and every value this build does not implement.
//!
//! Owner: metrale-ml-utils.
//! Invariants: none beyond the types.

use super::*;

pub(crate) const UNIFORM: &str = r#"
schema = 1
seed = 7
[layers]
per_signature = 1
[experts]
keep = "all"
[vocab]
keep = "all"
[mtp]
keep = true
[vision]
keep = true
[capacity]
kv = "free"
[routing]
mode = "uniform"
[values]
mode = "init"
[speculative]
accept = "natural"
"#;

#[test]
fn the_full_form_parses_and_round_trips_through_canonical() {
    let s = MockSpec::parse(UNIFORM).expect("spec");
    assert_eq!(s.seed, 7);
    assert_eq!(s.per_signature, PerSignature::All(1));
    assert_eq!(s.routing, RoutingMode::Uniform);
    let again = MockSpec::parse(&s.canonical()).expect("canonical parses");
    assert_eq!(again, s);
    assert_eq!(again.canonical(), s.canonical());
}

#[test]
fn a_list_and_a_histogram_round_trip() {
    let text = UNIFORM
        .replace("per_signature = 1", "per_signature = [2, 1]")
        .replace(
            "mode = \"uniform\"",
            "mode = \"histogram\"\nhistogram = \"p \\\"q\\\".json\"",
        );
    let s = MockSpec::parse(&text).expect("spec");
    assert_eq!(s.per_signature, PerSignature::Each(vec![2, 1]));
    assert_eq!(
        s.routing,
        RoutingMode::Histogram {
            path: "p \"q\".json".into()
        }
    );
    assert_eq!(MockSpec::parse(&s.canonical()).unwrap(), s);
    assert_eq!(s.counts(2).unwrap(), vec![2, 1]);
    assert!(s.counts(3).is_err());
}

#[test]
fn missing_and_unknown_keys_are_refused() {
    let missing = UNIFORM.replace("[vocab]\nkeep = \"all\"\n", "");
    assert!(MockSpec::parse(&missing).is_err());
    let unknown = UNIFORM.replace("seed = 7", "seed = 7\nlayers_total = 3");
    assert!(MockSpec::parse(&unknown).is_err());
}

#[test]
fn later_milestone_values_are_refused_with_the_reason() {
    let cases = [
        (
            "keep = \"all\"\n[vocab]",
            "keep = \"32\"\n[vocab]",
            "experts.keep",
        ),
        (
            "[vocab]\nkeep = \"all\"",
            "[vocab]\nkeep = \"1024\"",
            "vocab.keep",
        ),
        ("[mtp]\nkeep = true", "[mtp]\nkeep = false", "mtp.keep"),
        (
            "[vision]\nkeep = true",
            "[vision]\nkeep = false",
            "vision.keep",
        ),
        ("kv = \"free\"", "kv = \"full-model\"", "capacity.kv"),
        (
            "[values]\nmode = \"init\"",
            "[values]\nmode = \"stats\"",
            "values.mode",
        ),
        (
            "accept = \"natural\"",
            "accept = \"0.7\"",
            "speculative.accept",
        ),
    ];
    for (from, to, key) in cases {
        let text = UNIFORM.replacen(from, to, 1);
        assert_ne!(text, UNIFORM, "{key}: the replacement must apply");
        let err = MockSpec::parse(&text).unwrap_err().to_string();
        assert!(err.contains(key) && err.contains("later"), "{key}: {err}");
    }
}

#[test]
fn routing_and_counts_are_validated() {
    let zero = UNIFORM.replace("per_signature = 1", "per_signature = 0");
    assert!(MockSpec::parse(&zero).is_err());
    let empty = UNIFORM.replace("per_signature = 1", "per_signature = []");
    assert!(MockSpec::parse(&empty).is_err());
    let no_path = UNIFORM.replace("mode = \"uniform\"", "mode = \"histogram\"");
    assert!(MockSpec::parse(&no_path).is_err());
    let stray = UNIFORM.replace(
        "mode = \"uniform\"",
        "mode = \"uniform\"\nhistogram = \"x\"",
    );
    assert!(MockSpec::parse(&stray).is_err());
    let other = UNIFORM.replace("mode = \"uniform\"", "mode = \"skewed\"");
    assert!(MockSpec::parse(&other).is_err());
}
