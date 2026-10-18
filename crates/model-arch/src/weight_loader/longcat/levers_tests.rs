// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The LongCat loader levers against the environment reads of `ffn.rs` and
//! `ngram.rs` they replaced, frozen here.
//!
//! Owner: model-arch weight loader (LongCat).
//! Invariants: the probes that read differently are exactly the listed behaviour changes: a
//! strict-`1` switch now also arms at `true`, a value the switch rule cannot read is refused at
//! startup, and a cache size the integer rule cannot read is refused instead of falling back.

use super::ngram::cache_slots;
use metrale_model_layers::lever_equivalence::{Old, Outcome, check, settings_from};
use metrale_settings::InferenceServerSettings;

/// 2026-10-17: The `ngram.rs` slot read, unchanged but for its input.
fn old_slots(v: Option<&str>) -> usize {
    v.and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(65536)
}

type Row = (&'static str, fn(&InferenceServerSettings) -> bool);

const OPT_IN: [Row; 2] = [
    ("METRALE_LONGCAT_BF16_FFN", |s| s.kernels.longcat_bf16_ffn),
    ("METRALE_LONGCAT_FP8_EXPERTS", |s| {
        s.kernels.longcat_fp8_experts
    }),
];

const NGRAM_DISABLE: &str = "METRALE_NGRAM_DISABLE";
const SLOTS: &str = "METRALE_NGRAM_CACHE_SLOTS";

#[test]
fn the_switches_read_as_before_except_true() {
    for (env, new) in OPT_IN {
        assert_eq!(
            check(env, Old::OptIn, |p| p == Some("1"), new),
            [
                (env, Some("true"), Outcome::Changed),
                (env, Some("garbage"), Outcome::Rejected)
            ],
            "{env}"
        );
    }
}

#[test]
fn the_ngram_kill_switch_still_arms_at_any_value() {
    assert_eq!(
        check(
            NGRAM_DISABLE,
            Old::Presence,
            |p| p.is_some(),
            |s| s.speculative.ngram_disable
        ),
        []
    );
}

#[test]
fn the_cache_size_refuses_what_the_integer_rule_cannot_read() {
    assert_eq!(
        check(SLOTS, Old::Number, old_slots, cache_slots),
        [
            (SLOTS, Some("true"), Outcome::Rejected),
            (SLOTS, Some("off"), Outcome::Rejected),
            (SLOTS, Some("garbage"), Outcome::Rejected)
        ]
    );
}

/// 2026-10-17: Zero and negative sizes still read as the default, a padded size is still
/// trimmed, and the empty value, which read as the default, is now refused.
#[test]
fn the_cache_size_reads_as_before_at_edge_values() {
    for v in ["0", "-1", "4096", " 4096 "] {
        let s = settings_from(&[(SLOTS, v)]).expect("an integer resolves");
        assert_eq!(cache_slots(&s), old_slots(Some(v)), "{v:?}");
    }
    assert!(settings_from(&[(SLOTS, "")]).is_err());
}
