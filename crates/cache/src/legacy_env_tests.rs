// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: [`legacy_env_cache_tuning_from`] against the `CacheTuning` projection that replaces
//! it, over every value in `VALUES` (plus edges): equal except for the listed deviations, which
//! are the behaviour change the 2c-device-wire commit makes when it deletes the legacy readers.
//!
//! Owner: cache.
//! Invariants: never reads or writes the process environment.

use metrale_settings::InferenceServerSettings;
use metrale_settings::groups::device::ProjectionError;
use metrale_settings::groups::device_cache::CacheTuning;
use metrale_settings::groups::equivalence::{
    VALUES, Want, check, check_values, eq_one_to_switch, var_in, var_one,
};

use super::legacy_env_cache_tuning_from;

fn projected(s: &InferenceServerSettings) -> Result<CacheTuning, ProjectionError> {
    Ok(CacheTuning::from(s))
}

/// 2026-10-17: The legacy reader's view of one field with only `env` set to the value.
fn legacy<T>(env: &'static str, field: impl Fn(&CacheTuning) -> T) -> impl Fn(Option<&str>) -> T {
    move |v| field(&legacy_env_cache_tuning_from(var_one(env, v)))
}

/// 2026-10-17: [`VALUES`] plus `extra`.
fn with(extra: &[Option<&'static str>]) -> Vec<Option<&'static str>> {
    VALUES
        .iter()
        .copied()
        .chain(extra.iter().copied())
        .collect()
}

#[test]
fn an_empty_environment_reads_as_the_declared_tuning() {
    assert_eq!(
        legacy_env_cache_tuning_from(var_in(&[])),
        CacheTuning::declared().expect("declared defaults resolve")
    );
}

#[test]
fn the_eq_one_switches_accept_true_and_reject_garbage() {
    // 2026-10-17: `METRALE_PREFIX_SUBBLOCK` logged an unrecognised value and left it off;
    // `METRALE_KV_TRACE` read anything but `1` as off.
    type Field = fn(&CacheTuning) -> bool;
    let switches: [(&str, Field); 2] = [
        ("METRALE_PREFIX_SUBBLOCK", |c| c.prefix_subblock),
        ("METRALE_KV_TRACE", |c| c.kv_trace),
    ];
    for (env, field) in switches {
        check(
            env,
            legacy(env, field),
            |s| projected(s).map(|c| field(&c)),
            &eq_one_to_switch(),
        );
    }
}

#[test]
fn presence_levers_are_on_for_any_value_including_zero_and_empty() {
    type Field = fn(&CacheTuning) -> bool;
    let levers: [(&str, Field); 3] = [
        ("METRALE_SNAP_LOOKUP_DBG", |c| c.snap_lookup_dbg),
        ("METRALE_SSM_SNAP_STATS", |c| c.ssm_snap_stats),
        ("METRALE_SNAP_EVICT_LEGACY", |c| c.snap_evict_legacy),
    ];
    for (env, field) in levers {
        check_values(
            env,
            &with(&[Some(""), Some("0")]),
            legacy(env, field),
            |s| projected(s).map(|c| field(&c)),
            &[],
        );
    }
}

#[test]
fn tail_lease_texts_keep_their_parses() {
    const PROTECT: &str = "METRALE_DISABLE_SSM_TAIL_PROTECT";
    check_values(
        PROTECT,
        &with(&[Some("on"), Some("ON"), Some("yes")]),
        legacy(PROTECT, |c| c.tail_lease),
        |s| projected(s).map(|c| c.tail_lease),
        &[],
    );
    const TTL: &str = "METRALE_SSM_TAIL_LEASE_TTL";
    check_values(
        TTL,
        &with(&[Some("18"), Some("-1"), Some(" 18"), Some("4294967296")]),
        legacy(TTL, |c| c.tail_lease_ttl),
        |s| projected(s).map(|c| c.tail_lease_ttl),
        &[],
    );
}

#[test]
fn snap_evict_alpha_clamps_and_rejects_what_the_legacy_parse_zeroed() {
    const ALPHA: &str = "METRALE_SNAP_EVICT_ALPHA";
    check_values(
        ALPHA,
        &with(&[
            Some("0.5"),
            Some("9"),
            Some("-2"),
            Some("inf"),
            Some(" 0.5"),
        ]),
        legacy(ALPHA, |c| c.snap_evict_alpha),
        |s| projected(s).map(|c| c.snap_evict_alpha),
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
    const HERMETIC: &str = "METRALE_HERMETIC";
    check(
        HERMETIC,
        legacy(HERMETIC, |c| c.hermetic),
        |s| projected(s).map(|c| c.hermetic),
        &[],
    );
}
