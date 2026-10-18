// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Each `legacy_env_*` reader against the settings projection that replaces it, over
//! every value in `VALUES` (plus numeric edges): equal except for the listed deviations, which are
//! the behaviour change the 2c-device-wire commit makes when it deletes the legacy readers.
//!
//! Owner: gpu-runtime.
//! Invariants: never reads or writes the process environment.

use metrale_settings::InferenceServerSettings;
use metrale_settings::groups::device::{DeviceSettings, ProjectionError};
use metrale_settings::groups::device_cache::CacheTuning;
use metrale_settings::groups::equivalence::{
    VALUES, Want, check_fallible, check_values, eq_one_to_switch, var_in, var_one,
};

use super::{
    legacy_env_arena_levers_from, legacy_env_device_settings_from, legacy_env_hermetic_from,
    legacy_env_ssm_tail_ckpt_from, legacy_env_ssm_tail_midchunk_from,
};
use crate::buffers::ArenaLevers;

fn device(s: &InferenceServerSettings) -> Result<DeviceSettings, ProjectionError> {
    DeviceSettings::try_from(s)
}

/// 2026-10-17: The legacy reader's view of one field with only `env` set to the value.
fn legacy<T>(
    env: &'static str,
    field: impl Fn(&DeviceSettings) -> T,
) -> impl Fn(Option<&str>) -> Result<T, ()> {
    move |v| {
        legacy_env_device_settings_from(var_one(env, v))
            .map(|d| field(&d))
            .map_err(|_| ())
    }
}

/// 2026-10-17: [`VALUES`] plus `extra`.
fn with(extra: &[Option<&'static str>]) -> Vec<Option<&'static str>> {
    VALUES
        .iter()
        .copied()
        .chain(extra.iter().copied())
        .collect()
}

/// 2026-10-17: An Int lever whose old reader parsed an unsigned type and defaulted on failure:
/// the non-numbers and the negative value now fail.
fn int_rejects<T>() -> Vec<(Option<&'static str>, Want<T>)> {
    ["true", "off", "garbage", "-1"]
        .map(|v| (Some(v), Want::Err))
        .into()
}

#[test]
fn an_empty_environment_reads_as_the_declared_settings() {
    assert_eq!(
        legacy_env_device_settings_from(var_in(&[])),
        Ok(DeviceSettings::declared().expect("declared defaults project"))
    );
}

#[test]
fn eq_one_switches_match_except_true_and_garbage() {
    type Field = fn(&DeviceSettings) -> bool;
    let switches: [(&str, Field); 7] = [
        ("METRALE_DEBUG_SYNC_KERNELS", |d| d.debug_sync_kernels),
        ("METRALE_MAIN_STREAM_HIGH_PRIORITY", |d| {
            d.stream_priority.high
        }),
        ("METRALE_NCCL_HOST_HOIST", |d| d.nccl_host_hoist),
        ("METRALE_D2H_SPIN_SYNC", |d| d.d2h_spin_sync),
        ("METRALE_GGUF_NATIVE_Q2", |d| d.gguf_native_q2),
        ("METRALE_GGUF_NATIVE_Q2_MMQ", |d| d.gguf_native_q2_mmq),
        ("METRALE_FP8_ROWWISE", |d| d.fp8_rowwise),
    ];
    for (env, field) in switches {
        check_fallible(
            env,
            &VALUES,
            legacy(env, field),
            |s| device(s).map(|d| field(&d)),
            &eq_one_to_switch(),
        );
    }
}

#[test]
fn under_greatest_accepts_the_switch_spellings_the_legacy_parse_rejected() {
    const ENV: &str = "METRALE_MAIN_STREAM_UNDER_GREATEST";
    let field = |d: &DeviceSettings| d.stream_priority.under_greatest;
    check_fallible(
        ENV,
        &VALUES,
        legacy(ENV, field),
        |s| device(s).map(|d| field(&d)),
        &[
            (Some("true"), Want::New(true)),
            (Some("off"), Want::New(false)),
        ],
    );
}

#[test]
fn redzone_bytes_round_up_and_reject_what_the_legacy_parse_zeroed() {
    const ENV: &str = "METRALE_REDZONE";
    let field = |d: &DeviceSettings| d.redzone.bytes;
    check_fallible(
        ENV,
        &with(&[Some("17"), Some("-1"), Some(" 48")]),
        legacy(ENV, field),
        |s| device(s).map(|d| field(&d)),
        &[
            (Some("true"), Want::Err),
            (Some("off"), Want::Err),
            (Some("garbage"), Want::Err),
            (Some("-1"), Want::Err),
            (Some(" 48"), Want::New(48)),
        ],
    );
}

#[test]
fn redzone_fill_index_and_trace_reject_what_the_legacy_parses_defaulted() {
    const FILL: &str = "METRALE_REDZONE_FILL";
    let fill = |d: &DeviceSettings| d.redzone.fill;
    check_fallible(
        FILL,
        &with(&[Some("300"), Some("-1"), Some("7")]),
        legacy(FILL, fill),
        |s| device(s).map(|d| fill(&d)),
        &[
            (Some("true"), Want::Err),
            (Some("off"), Want::Err),
            (Some("garbage"), Want::Err),
            (Some("300"), Want::Err),
            (Some("-1"), Want::Err),
        ],
    );
    const MIN: &str = "METRALE_REDZONE_MIN_IDX";
    let min = |d: &DeviceSettings| d.redzone.min_idx;
    check_fallible(
        MIN,
        &with(&[Some("-1"), Some("7")]),
        legacy(MIN, min),
        |s| device(s).map(|d| min(&d)),
        &int_rejects(),
    );
    const TRACE: &str = "METRALE_REDZONE_TRACE_IDX";
    let trace = |d: &DeviceSettings| d.redzone.trace_idx;
    check_fallible(
        TRACE,
        &with(&[Some("-1"), Some("7")]),
        legacy(TRACE, trace),
        |s| device(s).map(|d| trace(&d)),
        &int_rejects(),
    );
}

#[test]
fn cutlass_workspace_is_mebibytes_and_rejects_what_the_legacy_parse_defaulted() {
    const ENV: &str = "METRALE_CUTLASS_WORKSPACE_MB";
    let field = |d: &DeviceSettings| d.cutlass_workspace_bytes;
    check_fallible(
        ENV,
        &with(&[Some("64"), Some("-1")]),
        legacy(ENV, field),
        |s| device(s).map(|d| field(&d)),
        &int_rejects(),
    );
}

#[test]
fn d2h_trace_and_the_splitk_texts_have_no_deviation() {
    const TRACE: &str = "METRALE_D2H_TRACE";
    let trace = |d: &DeviceSettings| d.d2h_trace;
    check_fallible(
        TRACE,
        &with(&[Some("12345")]),
        legacy(TRACE, trace),
        |s| device(s).map(|d| trace(&d)),
        &[],
    );
    type Field = fn(&DeviceSettings) -> Option<String>;
    let texts: [(&str, Field); 2] = [
        ("METRALE_ATTN_DECODE_SPLITK", |d| {
            d.attn_decode_splitk.clone()
        }),
        ("METRALE_ATTN_DECODE_GQA_PACK", |d| {
            d.attn_decode_gqa_pack.clone()
        }),
    ];
    for (env, field) in texts {
        check_fallible(
            env,
            &with(&[Some("4"), Some("auto")]),
            legacy(env, field),
            |s| device(s).map(|d| field(&d)),
            &[],
        );
    }
}

#[test]
fn arena_levers_match_their_projection_except_the_switch_spellings() {
    let declared = ArenaLevers::from(&DeviceSettings::declared().expect("declared"));
    type Set = fn(&mut ArenaLevers);
    let switches: [(&str, Set); 3] = [
        ("METRALE_GGUF_NATIVE_Q2", |a| a.gguf_native_q2 = true),
        ("METRALE_GGUF_NATIVE_Q2_MMQ", |a| {
            a.gguf_native_q2_mmq = true
        }),
        ("METRALE_FP8_ROWWISE", |a| a.fp8_rowwise = true),
    ];
    for (env, on) in switches {
        let mut enabled = declared;
        on(&mut enabled);
        check_values(
            env,
            &VALUES,
            |v| legacy_env_arena_levers_from(var_one(env, v)),
            |s| device(s).map(|d| ArenaLevers::from(&d)),
            &[
                (Some("true"), Want::New(enabled)),
                (Some("garbage"), Want::Err),
            ],
        );
    }
    check_values(
        "METRALE_ATTN_DECODE_SPLITK",
        &with(&[Some("4"), Some("auto"), Some("legacy")]),
        |v| legacy_env_arena_levers_from(var_one("METRALE_ATTN_DECODE_SPLITK", v)),
        |s| device(s).map(|d| ArenaLevers::from(&d)),
        &[],
    );
}

#[test]
fn the_process_switches_match_their_settings_rows() {
    const CKPT: &str = "METRALE_SSM_TAIL_CKPT";
    check_values(
        CKPT,
        &VALUES,
        |v| legacy_env_ssm_tail_ckpt_from(var_one(CKPT, v)),
        |s| Ok(s.scheduler.ssm_tail_ckpt),
        &eq_one_to_switch(),
    );
    const HERMETIC: &str = "METRALE_HERMETIC";
    check_values(
        HERMETIC,
        &VALUES,
        |v| legacy_env_hermetic_from(var_one(HERMETIC, v)),
        |s| Ok(CacheTuning::from(s).hermetic),
        &[],
    );
    const MIDCHUNK: &str = "METRALE_SSM_TAIL_MIDCHUNK";
    check_values(
        MIDCHUNK,
        &with(&[Some(""), Some(" 0")]),
        |v| legacy_env_ssm_tail_midchunk_from(var_one(MIDCHUNK, v)),
        |s| Ok(s.scheduler.ssm_tail_midchunk.as_deref() != Some("0")),
        &[],
    );
}
