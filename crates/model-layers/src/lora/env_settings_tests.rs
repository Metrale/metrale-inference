// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The LoRA readers on the resolved settings against the readers they replaced
//! (each copied here from `lora/env.rs` before the move).
//!
//! Owner: model-layers (lora).
//! Invariants: the probes that read differently are exactly the listed behaviour changes.

use super::{ExpertLoraPolicy, lora_peer, no_batch_verify, prefill_bgmv_forced};
use crate::lever_equivalence::{Old, Outcome, check, settings_from};

fn old_truthy(p: Option<&str>) -> bool {
    p.is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

fn old_expert_rank(p: Option<&str>) -> usize {
    p.and_then(|v| v.parse().ok())
        .filter(|&r: &usize| r > 0)
        .unwrap_or(16)
}

#[test]
fn lora_experts_reads_as_before_except_the_switch_spellings() {
    let changes = check("METRALE_LORA_EXPERTS", Old::Truthy, old_truthy, |s| {
        ExpertLoraPolicy::from_settings(s).enabled
    });
    assert_eq!(
        changes,
        [("METRALE_LORA_EXPERTS", Some("garbage"), Outcome::Rejected)]
    );
}

#[test]
fn lora_expert_rank_reads_as_before() {
    let changes = check(
        "METRALE_LORA_EXPERT_RANK",
        Old::Verbatim,
        old_expert_rank,
        |s| ExpertLoraPolicy::from_settings(s).max_rank,
    );
    assert_eq!(changes, []);
    // 2026-10-17: A rank other than the probes' 1 and the default 16 also passes through.
    for raw in ["32", "-4", " 8", ""] {
        let s = settings_from(&[("METRALE_LORA_EXPERT_RANK", raw)]).expect("a text row");
        assert_eq!(
            ExpertLoraPolicy::from_settings(&s).max_rank,
            old_expert_rank(Some(raw)),
            "{raw:?}"
        );
    }
}

#[test]
fn prefill_bgmv_reads_as_before_except_the_switch_spellings() {
    let changes = check(
        "METRALE_LORA_PREFILL_BGMV",
        Old::OptIn,
        |p| p == Some("1"),
        prefill_bgmv_forced,
    );
    assert_eq!(
        changes,
        [
            ("METRALE_LORA_PREFILL_BGMV", Some("true"), Outcome::Changed),
            (
                "METRALE_LORA_PREFILL_BGMV",
                Some("garbage"),
                Outcome::Rejected
            ),
        ]
    );
}

#[test]
fn no_batch_verify_reads_as_before_except_the_switch_spellings() {
    let changes = check(
        "METRALE_LORA_NO_BATCH_VERIFY",
        Old::OptIn,
        |p| p == Some("1"),
        no_batch_verify,
    );
    assert_eq!(
        changes,
        [
            (
                "METRALE_LORA_NO_BATCH_VERIFY",
                Some("true"),
                Outcome::Changed
            ),
            (
                "METRALE_LORA_NO_BATCH_VERIFY",
                Some("garbage"),
                Outcome::Rejected
            ),
        ]
    );
}

#[test]
fn lora_peer_reads_as_before() {
    let old = |p: Option<&str>| p.filter(|s| !s.is_empty()).map(str::to_owned);
    let changes = check("METRALE_LORA_PEER", Old::Verbatim, old, |s| {
        lora_peer(s).map(str::to_owned)
    });
    assert_eq!(changes, []);
    // 2026-10-17: Set but empty is no peer, as before.
    let s = settings_from(&[("METRALE_LORA_PEER", "")]).expect("a text row");
    assert_eq!(lora_peer(&s), None);
}
