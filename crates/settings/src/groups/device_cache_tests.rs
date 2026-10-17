// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `CacheTuning` against the metrale-cache and gpu-runtime readers it replaced.
//!
//! Owner: settings.
//! Invariants: never reads or writes the process environment.

use super::CacheTuning;
use crate::groups::device::ProjectionError;
use crate::groups::equivalence::{VALUES, Want, check, check_values, eq_one, eq_one_to_switch};

fn cache(s: &crate::InferenceServerSettings) -> Result<CacheTuning, ProjectionError> {
    Ok(CacheTuning::from(s))
}

#[test]
fn prefix_subblock_accepts_true_and_rejects_garbage_instead_of_logging() {
    // 2026-10-17: Old `partial_tail::enabled_from`: an unrecognised value was logged and off.
    let old = |v: Option<&str>| v == Some("1");
    check(
        "METRALE_PREFIX_SUBBLOCK",
        old,
        |s| cache(s).map(|c| c.prefix_subblock),
        &eq_one_to_switch(),
    );
}

#[test]
fn kv_trace_moves_onto_a_switch() {
    check(
        "METRALE_KV_TRACE",
        eq_one,
        |s| cache(s).map(|c| c.kv_trace),
        &eq_one_to_switch(),
    );
}

#[test]
fn presence_levers_are_on_for_any_value() {
    type Field = fn(&CacheTuning) -> bool;
    let levers: [(&str, Field); 3] = [
        ("METRALE_SNAP_LOOKUP_DBG", |c| c.snap_lookup_dbg),
        ("METRALE_SSM_SNAP_STATS", |c| c.ssm_snap_stats),
        ("METRALE_SNAP_EVICT_LEGACY", |c| c.snap_evict_legacy),
    ];
    for (env, field) in levers {
        check(env, |v| v.is_some(), |s| cache(s).map(|c| field(&c)), &[]);
    }
}

#[test]
fn tail_lease_texts_keep_their_parses() {
    check_values(
        "METRALE_DISABLE_SSM_TAIL_PROTECT",
        &[VALUES.as_slice(), &[Some("on"), Some("ON"), Some("yes")]].concat(),
        |v| !matches!(v, Some("1" | "on" | "true")),
        |s| cache(s).map(|c| c.tail_lease),
        &[],
    );
    check_values(
        "METRALE_SSM_TAIL_LEASE_TTL",
        &[VALUES.as_slice(), &[Some("18"), Some("-1"), Some(" 18")]].concat(),
        |v| v.and_then(|v| v.parse::<u32>().ok()).unwrap_or(64),
        |s| cache(s).map(|c| c.tail_lease_ttl),
        &[],
    );
}

#[test]
fn snap_evict_alpha_clamps_and_rejects_what_the_old_parse_zeroed() {
    let old = |v: Option<&str>| {
        v.and_then(|v| v.parse::<f64>().ok())
            .map(|a| a.clamp(0.0, 8.0))
            .unwrap_or(0.0)
    };
    check_values(
        "METRALE_SNAP_EVICT_ALPHA",
        &[
            VALUES.as_slice(),
            &[
                Some("0.5"),
                Some("9"),
                Some("-2"),
                Some("inf"),
                Some(" 0.5"),
            ],
        ]
        .concat(),
        old,
        |s| cache(s).map(|c| c.snap_evict_alpha),
        &[
            (Some("true"), Want::Err),
            (Some("off"), Want::Err),
            (Some("garbage"), Want::Err),
            (Some("inf"), Want::Err),
            (Some(" 0.5"), Want::New(0.5)),
        ],
    );
}

#[test]
fn hermetic_is_on_only_for_one() {
    check(
        "METRALE_HERMETIC",
        eq_one,
        |s| cache(s).map(|c| c.hermetic),
        &[],
    );
}
