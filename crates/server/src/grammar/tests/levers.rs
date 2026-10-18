// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: [`GrammarLevers::from_settings`] against the per-call env readers it replaced
//! (`compile_tools`, `mask_cache::attach_mask_cache`, `prewarm::overlap_enabled`). Each lever
//! is resolved over `MapEnv` as `met serve` resolves a load; no process environment is read.
//!
//! Owner: server (grammar) tests.
//! Invariants: every outcome that differs from the old reader is listed in [`CHANGES`].

use super::super::GrammarLevers;
use crate::lever_equivalence::{Old, Out, Row, assert_old_readers_hold, settings_from};

fn resolved(env: &[(&str, &str)]) -> anyhow::Result<GrammarLevers> {
    Ok(GrammarLevers::from_settings(&settings_from(env)?))
}

#[test]
fn unset_is_what_an_empty_environment_resolves_to() {
    assert_eq!(
        resolved(&[]).expect("defaults resolve"),
        GrammarLevers::UNSET
    );
}

#[test]
fn the_cache_dir_is_kept_verbatim() {
    let at = |v| resolved(&[("METRALE_GRAMMAR_CACHE_DIR", v)]).map(|l| l.mask_cache_dir);
    assert_eq!(
        at("/srv/masks").expect("a path").as_deref(),
        Some("/srv/masks")
    );
    // 2026-10-17: A blank directory still reaches `snapshot_path`, which ignores it as before.
    assert_eq!(at(" ").expect("blank").as_deref(), Some(" "));
}

fn rows() -> Vec<Row<GrammarLevers>> {
    vec![
        ("METRALE_GRAMMAR_ALLOW_EMPTY_VALUE", Old::OptIn, |l| {
            Out::B(l.allow_empty_value)
        }),
        ("METRALE_GRAMMAR_FORCE_CLOSE", Old::OptIn, |l| {
            Out::B(l.force_close)
        }),
        ("METRALE_GRAMMAR_VALUE_HARDEN", Old::OptIn, |l| {
            Out::B(l.value_harden)
        }),
        ("METRALE_TOOL_SHORT_TRIGGER", Old::OptIn, |l| {
            Out::B(l.short_trigger)
        }),
        ("METRALE_GRAMMAR_ASYNC_PREWARM", Old::OnUnlessOffWord, |l| {
            Out::B(l.async_prewarm)
        }),
        ("METRALE_GRAMMAR_CACHE", Old::OnUnlessOffWord, |l| {
            Out::B(l.mask_cache)
        }),
    ]
}

/// 2026-10-17: The four `== "1"` switches now use the settings crate's switch grammar: switch
/// words parse in any case, trimmed, and anything else stops the load. The two kill switches
/// keep their readers and do not change.
const CHANGES: &[(&str, &str, Out)] = &[
    ("opt_in", "true", Out::B(true)),
    ("opt_in", "TRUE", Out::B(true)),
    ("opt_in", "yes", Out::B(true)),
    ("opt_in", "", Out::Err),
    ("opt_in", " 1 ", Out::B(true)),
    ("opt_in", "garbage", Out::Err),
    ("opt_in", "7", Out::Err),
    ("opt_in", "-1", Out::Err),
];

#[test]
fn resolved_levers_match_the_old_readers_except_the_recorded_changes() {
    assert_old_readers_hold(&rows(), resolved, CHANGES);
}
