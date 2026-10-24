// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `MoeLevers::from_settings` against the environment readers it replaced, frozen
//! here (the closures and `old_*`, unchanged but for taking the raw value as a parameter).
//!
//! Owner: model-layers (MoE).
//! Invariants: the probes that read differently are exactly the listed behaviour changes: a `1`
//! or `0` switch now also reads the other switch spellings and refuses a value the switch rule
//! cannot read; the presence levers, the atomic C=4 predicate and the routed row bound read as
//! before.

use super::*;
use crate::lever_equivalence::{Old, Outcome, check, defaults, settings_from};

/// 2026-10-17: `multi_seq/ffn.rs::grouped_routed_decode_min`.
fn old_grouped_routed_decode_min(v: Option<&str>) -> usize {
    v.and_then(|v| v.parse().ok()).unwrap_or(2)
}

/// 2026-10-17: `moe_grouped_decode_for` of the two old readers.
fn old_grouped_decode_for(n: usize, kill: Option<&str>, force: Option<&str>) -> bool {
    crate::layers::moe_grouped_decode_decide(n, kill.is_none(), force == Some("1"))
}

/// 2026-10-17: The levers of `pairs` and nothing else; `None` when the settings refuse them.
fn new_of(pairs: &[(&str, &str)]) -> Option<MoeLevers> {
    settings_from(pairs)
        .ok()
        .map(|s| MoeLevers::from_settings(&s))
}

type Field = (&'static str, fn(&MoeLevers) -> bool);

/// 2026-10-17: The `var_os(..).is_none()` kill switches: off at any value, `0` and the empty
/// string included.
#[test]
fn the_presence_kill_switches_read_as_before() {
    let cases: [Field; 4] = [
        ("METRALE_NO_MOE_GROUPED_DECODE", |l| l.grouped_decode),
        ("METRALE_NO_MOE_FP8_TC", |l| l.fp8_tc),
        ("METRALE_NO_FP8_MOE_GROUPED_DECODE", |l| {
            l.fp8_grouped_decode
        }),
        ("METRALE_NO_MOE_NVFP4_TC", |l| l.nvfp4_tc),
    ];
    for (env, field) in cases {
        let changes = check(
            env,
            Old::Presence,
            |p| p.is_none(),
            |s| field(&MoeLevers::from_settings(s)),
        );
        assert!(changes.is_empty(), "{changes:?}");
        for probe in ["", " ", "no"] {
            assert_eq!(
                new_of(&[(env, probe)]).map(|l| field(&l)),
                Some(false),
                "{probe:?}"
            );
        }
    }
}

/// 2026-10-17: The `== "1"` switches (`dump::enabled`, the two `MOE_BATCHED_DECODE` sites,
/// `grouped_routed_decode_enabled`, `moe_grouped_decode_forced`) and the `!= "0"` pairwise arm.
#[test]
fn the_switches_read_as_before_except_the_switch_spellings() {
    let opt_ins: [Field; 4] = [
        ("METRALE_DUMP_EXPERT_IDS", |l| l.dump_expert_ids),
        ("METRALE_MOE_BATCHED_DECODE", |l| l.batched_decode),
        ("METRALE_MOE_GROUPED_ROUTED_DECODE", |l| {
            l.grouped_routed_decode
        }),
        ("METRALE_MOE_GROUPED_DECODE", |l| l.grouped_decode_forced),
    ];
    for (env, field) in opt_ins {
        let changes = check(
            env,
            Old::OptIn,
            |p| p == Some("1"),
            |s| field(&MoeLevers::from_settings(s)),
        );
        assert_eq!(
            changes,
            [
                (env, Some("true"), Outcome::Changed),
                (env, Some("garbage"), Outcome::Rejected)
            ],
            "{env}"
        );
    }
    let env = "METRALE_MOE_PAIRWISE_DECODE";
    let changes = check(
        env,
        Old::OptOut,
        |p| p != Some("0"),
        |s| MoeLevers::from_settings(s).pairwise_decode,
    );
    assert_eq!(
        changes,
        [
            (env, Some("off"), Outcome::Changed),
            (env, Some("garbage"), Outcome::Rejected)
        ]
    );
}

/// 2026-10-17: `trait_decode_multi_seq.rs`'s atomic C=4 arm: exactly `1`.
#[test]
fn the_atomic_c4_predicate_is_kept() {
    let env = "METRALE_MOE_ATOMIC_C4_DECODE";
    let old = |p: Option<&str>| p == Some("1");
    let changes = check(env, Old::Verbatim, old, |s| {
        MoeLevers::from_settings(s).atomic_c4_decode
    });
    assert!(changes.is_empty(), "{changes:?}");
    for probe in ["", " 1", "01", "TRUE", "yes"] {
        assert_eq!(
            new_of(&[(env, probe)]).map(|l| l.atomic_c4_decode),
            Some(old(Some(probe)))
        );
    }
}

/// 2026-10-17: The routed arm's row bound keeps its parse, including the values outside the
/// probes: zero, signed, padded and empty.
#[test]
fn the_routed_row_bound_keeps_its_parse() {
    let env = "METRALE_MOE_GROUPED_ROUTED_DECODE_MIN";
    let changes = check(env, Old::Verbatim, old_grouped_routed_decode_min, |s| {
        MoeLevers::from_settings(s).grouped_routed_decode_min
    });
    assert!(changes.is_empty(), "{changes:?}");
    for probe in ["", " 4", "+4", "-4", "0", "4", "18446744073709551616"] {
        assert_eq!(
            new_of(&[(env, probe)]).map(|l| l.grouped_routed_decode_min),
            Some(old_grouped_routed_decode_min(Some(probe))),
            "{probe:?}"
        );
    }
}

/// 2026-10-17: `grouped_decode_for` is the old `moe_grouped_decode_for` at every width and
/// every pair of the kill switch and the force whose spellings both readers share.
#[test]
fn grouped_decode_for_is_the_old_decision() {
    const KILL: &str = "METRALE_NO_MOE_GROUPED_DECODE";
    const FORCE: &str = "METRALE_MOE_GROUPED_DECODE";
    for kill in [None, Some(""), Some("0"), Some("1")] {
        for force in [None, Some("0"), Some("1")] {
            let mut pairs = Vec::new();
            pairs.extend(kill.map(|v| (KILL, v)));
            pairs.extend(force.map(|v| (FORCE, v)));
            let levers = new_of(&pairs).expect("both values resolve");
            for n in [1, 4, 15, 16, 64] {
                assert_eq!(
                    levers.grouped_decode_for(n),
                    old_grouped_decode_for(n, kill, force),
                    "n={n} {kill:?} {force:?}"
                );
            }
        }
    }
}

/// 2026-10-17: Every lever at the value its reader read unset as.
#[test]
fn the_declared_defaults_are_the_old_unset_values() {
    let levers = MoeLevers::from_settings(&defaults());
    assert_eq!(levers, MoeLevers::declared());
    assert_eq!(
        levers.grouped_routed_decode_min,
        old_grouped_routed_decode_min(None)
    );
}
