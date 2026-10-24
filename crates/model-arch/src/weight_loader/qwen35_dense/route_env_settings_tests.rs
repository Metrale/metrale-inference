// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: [`RouteEnv::from_settings`] against the readers `RouteEnv::from_env` used, for
//! each of the three levers it reads itself (the dispatch half is `GemmDispatch::from_settings`,
//! tested in model-layers), and for `METRALE_MEM_PROFILE`, which the dense loader reads from the
//! same settings.
//!
//! Owner: model-arch (qwen35_dense).
//! Invariants: the probes that read differently are exactly the listed behaviour changes.

use super::RouteEnv;
use metrale_model_layers::lever_equivalence::{Old, Outcome, check};

#[test]
fn attn_w4a4_moves_from_presence_to_a_switch() {
    let changes = check(
        "METRALE_ATTN_W4A4",
        Old::Presence,
        |p| p.is_some(),
        |s| RouteEnv::from_settings(s).attn_w4a4,
    );
    assert_eq!(
        changes,
        [
            ("METRALE_ATTN_W4A4", Some("0"), Outcome::Changed),
            ("METRALE_ATTN_W4A4", Some("off"), Outcome::Changed),
            ("METRALE_ATTN_W4A4", Some("garbage"), Outcome::Rejected),
        ]
    );
}

#[test]
fn attn_prefill_q_t_moves_from_exact_one_to_a_switch() {
    let changes = check(
        "METRALE_ATTN_PREFILL_Q_T",
        Old::OptIn,
        |p| p == Some("1"),
        |s| RouteEnv::from_settings(s).attn_prefill_q_t,
    );
    assert_eq!(
        changes,
        [
            ("METRALE_ATTN_PREFILL_Q_T", Some("true"), Outcome::Changed),
            (
                "METRALE_ATTN_PREFILL_Q_T",
                Some("garbage"),
                Outcome::Rejected
            ),
        ]
    );
}

#[test]
fn keep_nvfp4_stays_presence() {
    let changes = check(
        "METRALE_DENSE_FP8_KEEP_NVFP4",
        Old::Presence,
        |p| p.is_some(),
        |s| RouteEnv::from_settings(s).keep_nvfp4,
    );
    assert!(changes.is_empty(), "{changes:?}");
}

#[test]
fn mem_profile_stays_presence() {
    let changes = check(
        "METRALE_MEM_PROFILE",
        Old::Presence,
        |p| p.is_some(),
        |s| s.telemetry.mem_profile,
    );
    assert!(changes.is_empty(), "{changes:?}");
}
