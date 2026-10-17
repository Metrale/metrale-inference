// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `DeviceSettings` against the gpu-runtime and kernels readers it replaced. Each
//! closure is the deleted reader's parse, copied verbatim apart from the variable lookup.
//!
//! Owner: settings.
//! Invariants: never reads or writes the process environment.

use super::{DeviceSettings, ProjectionError};
use crate::groups::equivalence::{
    VALUES, Want, check, check_fallible, check_values, eq_one, eq_one_to_switch, resolved,
};

fn device(s: &crate::InferenceServerSettings) -> Result<DeviceSettings, ProjectionError> {
    DeviceSettings::try_from(s)
}

/// 2026-10-17: [`VALUES`] plus the numeric edges the Int levers need.
fn numeric(extra: &[Option<&'static str>]) -> Vec<Option<&'static str>> {
    VALUES
        .iter()
        .copied()
        .chain(extra.iter().copied())
        .collect()
}

#[test]
fn eq_one_switches_match_the_old_readers_except_true_and_garbage() {
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
        check(
            env,
            eq_one,
            |s| device(s).map(|d| field(&d)),
            &eq_one_to_switch(),
        );
    }
}

#[test]
fn under_greatest_accepts_the_switch_spellings_the_old_parse_rejected() {
    // 2026-10-17: Old `main_stream_under_greatest` parse; its on-without-high check stays in
    // gpu-runtime, where the stream is created.
    let old = |v: Option<&str>| match v {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err(()),
    };
    check_fallible(
        "METRALE_MAIN_STREAM_UNDER_GREATEST",
        &VALUES,
        old,
        |s| device(s).map(|d| d.stream_priority.under_greatest),
        &[
            (Some("true"), Want::New(true)),
            (Some("off"), Want::New(false)),
        ],
    );
}

#[test]
fn redzone_bytes_round_up_and_reject_what_the_old_parse_zeroed() {
    let old = |v: Option<&str>| {
        v.and_then(|v| v.parse::<usize>().ok())
            .map(|n| if n == 0 { 0 } else { n.next_multiple_of(16) })
            .unwrap_or(0)
    };
    check_values(
        "METRALE_REDZONE",
        &numeric(&[Some("17"), Some("-1"), Some(" 48")]),
        old,
        |s| device(s).map(|d| d.redzone.bytes),
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
fn a_negative_redzone_is_a_projection_error_naming_the_lever() {
    let s = resolved(&[("METRALE_REDZONE", "-1")]).expect("-1 is a valid Int");
    match device(&s) {
        Err(ProjectionError::OutOfRange { env, .. }) => assert_eq!(env, "METRALE_REDZONE"),
        other => panic!("expected OutOfRange, got {other:?}"),
    }
}

#[test]
fn redzone_fill_index_and_trace_match_the_old_parses_on_valid_values() {
    let fill_old = |v: Option<&str>| v.and_then(|v| v.parse::<u8>().ok()).unwrap_or(0xEE);
    check_values(
        "METRALE_REDZONE_FILL",
        &numeric(&[Some("300"), Some("-1")]),
        fill_old,
        |s| device(s).map(|d| d.redzone.fill),
        &[
            (Some("true"), Want::Err),
            (Some("off"), Want::Err),
            (Some("garbage"), Want::Err),
            (Some("300"), Want::Err),
            (Some("-1"), Want::Err),
        ],
    );
    let usize_old = |v: Option<&str>| v.and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
    check_values(
        "METRALE_REDZONE_MIN_IDX",
        &numeric(&[Some("-1"), Some("7")]),
        usize_old,
        |s| device(s).map(|d| d.redzone.min_idx),
        &int_rejects(),
    );
    let trace_old = |v: Option<&str>| v.and_then(|v| v.parse::<usize>().ok());
    check_values(
        "METRALE_REDZONE_TRACE_IDX",
        &numeric(&[Some("-1"), Some("7")]),
        trace_old,
        |s| device(s).map(|d| d.redzone.trace_idx),
        &int_rejects(),
    );
}

/// 2026-10-17: An Int lever where the old reader parsed an unsigned type and defaulted on failure:
/// the non-numbers and the negative value now fail.
fn int_rejects<T>() -> Vec<(Option<&'static str>, Want<T>)> {
    ["true", "off", "garbage", "-1"]
        .map(|v| (Some(v), Want::Err))
        .into()
}

#[test]
fn cutlass_workspace_is_mebibytes_and_rejects_what_the_old_parse_defaulted() {
    let old =
        |v: Option<&str>| v.and_then(|v| v.parse::<usize>().ok()).unwrap_or(512) * 1024 * 1024;
    check_values(
        "METRALE_CUTLASS_WORKSPACE_MB",
        &numeric(&[Some("64"), Some("-1")]),
        old,
        |s| device(s).map(|d| d.cutlass_workspace_bytes),
        &int_rejects(),
    );
}

#[test]
fn d2h_trace_keeps_the_old_lenient_parse() {
    let old = |v: Option<&str>| v.map(|v| v.parse::<u64>().unwrap_or(0));
    check_values(
        "METRALE_D2H_TRACE",
        &numeric(&[Some("12345")]),
        old,
        |s| device(s).map(|d| d.d2h_trace),
        &[],
    );
}

#[test]
fn splitk_texts_pass_through_raw_for_the_kernels_resolvers() {
    for env in ["METRALE_ATTN_DECODE_SPLITK", "METRALE_ATTN_DECODE_GQA_PACK"] {
        check(
            env,
            |v| v.map(String::from),
            |s| {
                device(s).map(|d| match env {
                    "METRALE_ATTN_DECODE_SPLITK" => d.attn_decode_splitk,
                    _ => d.attn_decode_gqa_pack,
                })
            },
            &[],
        );
    }
}

#[test]
fn declared_is_the_projection_of_an_empty_environment() {
    let empty = resolved(&[]).expect("declared defaults resolve");
    assert_eq!(
        DeviceSettings::declared().expect("declared defaults project"),
        device(&empty).expect("empty environment projects")
    );
}
