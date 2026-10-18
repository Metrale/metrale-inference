// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `--hermetic`'s `expand` and its `CLOSED_KEYS` table.
//! 2026-10-17: Split from `hermetic_tests.rs`: the table lives in metrale-bench, so these run
//! only in a build with the `bench` feature.
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use super::{CLOSED_KEYS, expand, mtp_gate_force, prefix_caching_enabled};

fn map(pairs: &[(&str, &str)]) -> std::collections::BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

/// 2026-09-26: `hermetic=true` expands into every `CLOSED_KEYS` entry and keeps its own
/// key. The gate expands before the recipe renders (`bench_serve_plan.rs`), so a recipe
/// that turns the prefix cache on still renders a command line `validate_serve_args`
/// accepts; `recipe_tests.rs` checks that on the fixture recipes.
#[test]
fn hermetic_expands_into_the_keys_it_closes() {
    let out = expand(map(&[("hermetic", "true")]));
    assert_eq!(
        out.get("enable_prefix_caching").map(String::as_str),
        Some("false")
    );
    assert_eq!(out.get("mtp_gate").map(String::as_str), Some("force"));
    assert_eq!(
        out.get("hermetic").map(String::as_str),
        Some("true"),
        "and the regime keeps its name"
    );
}

/// 2026-09-26: Without `hermetic=true` the map comes back unchanged.
#[test]
fn nothing_expands_without_hermetic() {
    assert_eq!(expand(map(&[])), map(&[]));
    assert_eq!(
        expand(map(&[("ssm_cache_slots", "256")])),
        map(&[("ssm_cache_slots", "256")])
    );
    // 2026-09-26: `is_requested` accepts only the exact value `true`.
    assert_eq!(
        expand(map(&[("hermetic", "false")])),
        map(&[("hermetic", "false")])
    );
}

/// 2026-09-26: A key the caller already set keeps its value, so `validate_serve_args`
/// still sees the contradiction and refuses it. Recipe defaults are not in this map.
#[test]
fn expansion_never_overwrites_a_value_someone_named() {
    let out = expand(map(&[
        ("hermetic", "true"),
        ("enable_prefix_caching", "true"),
    ]));
    assert_eq!(
        out.get("enable_prefix_caching").map(String::as_str),
        Some("true"),
        "an explicit opposite must survive to be refused, not be quietly fixed"
    );
    let out = expand(map(&[("hermetic", "true"), ("mtp_gate", "auto")]));
    assert_eq!(out.get("mtp_gate").map(String::as_str), Some("auto"));
}

/// 2026-09-26: Every `CLOSED_KEYS` entry resolves, under `--hermetic`, to the value the
/// table gives, and a key with no resolver check here fails the test.
#[test]
fn hermetic_closures_match_the_resolvers() {
    for (key, value) in CLOSED_KEYS {
        match *key {
            "enable_prefix_caching" => {
                let want: bool = value.parse().expect("a bool");
                // 2026-09-26: The request is the opposite of the expected value,
                // so a resolver that ignored `hermetic` would fail.
                assert_eq!(
                    prefix_caching_enabled(!want, true),
                    want,
                    "CLOSED_KEYS says {key}={value}, the resolver disagrees"
                );
            }
            "mtp_gate" => {
                assert_eq!(
                    mtp_gate_force(Some("auto"), true),
                    Some(*value == "force"),
                    "CLOSED_KEYS says {key}={value}, the resolver disagrees"
                );
            }
            other => panic!(
                "CLOSED_KEYS gained `{other}` with no resolver check — add one here, or \
                 the disclosure and the enforcement can disagree about it"
            ),
        }
    }
}
