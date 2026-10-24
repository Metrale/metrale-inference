// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `NemotronLevers::from_settings` against the environment reads of `nemotron.rs`,
//! `nemotron/ssm_layer.rs` and `nemotron_moe/prefill_weights.rs` it replaced, frozen here.
//!
//! Owner: model-arch weight loader (Nemotron-H).
//! Invariants: the probes that read differently are exactly the listed behaviour changes: the
//! two BF16 opt-outs now also turn off at `off`, and a value the switch rule cannot read is
//! refused at startup.

use super::*;
use metrale_model_layers::lever_equivalence::{Old, Outcome, check, settings_from};

/// 2026-10-17: The reads `from_settings` replaced, unchanged but for their paths.
fn from_values(value: impl Fn(&str) -> Option<String>) -> NemotronLevers {
    let mode = value("METRALE_NEMOTRON_NATIVE_FP8_SSM").unwrap_or_else(|| "1".to_string());
    NemotronLevers {
        bf16_attn: value("METRALE_NEMOTRON_BF16_ATTN").as_deref() != Some("0"),
        native_fp8_ssm_decode: matches!(mode.as_str(), "1" | "both" | "decode"),
        native_fp8_ssm_prefill: matches!(mode.as_str(), "1" | "both"),
        native_bf16_ssm: value("METRALE_NEMOTRON_NATIVE_BF16_SSM").as_deref() != Some("0"),
        no_ssm_fp8_prefill: value("METRALE_NO_SSM_FP8_PREFILL").is_some(),
        no_ssm_prefill_t: value("METRALE_NO_SSM_PREFILL_T").is_some(),
        shared_fp8_prefill: value("METRALE_SHARED_FP8_PREFILL").is_some(),
    }
}

const OPT_OUT: [&str; 2] = [
    "METRALE_NEMOTRON_BF16_ATTN",
    "METRALE_NEMOTRON_NATIVE_BF16_SSM",
];
const PRESENCE: [&str; 3] = [
    "METRALE_NO_SSM_FP8_PREFILL",
    "METRALE_NO_SSM_PREFILL_T",
    "METRALE_SHARED_FP8_PREFILL",
];
const FP8_SSM: &str = "METRALE_NEMOTRON_NATIVE_FP8_SSM";

fn old_at(env: &str, probe: Option<&str>) -> NemotronLevers {
    from_values(|var| (var == env).then(|| probe.map(str::to_string)).flatten())
}

fn changes(env: &'static str, old: Old) -> Vec<(&'static str, Option<&'static str>, Outcome)> {
    check(env, old, |p| old_at(env, p), NemotronLevers::from_settings)
}

#[test]
fn every_lever_reads_as_before_except_the_opt_out_spellings() {
    for env in OPT_OUT {
        assert_eq!(
            changes(env, Old::OptOut),
            [
                (env, Some("off"), Outcome::Changed),
                (env, Some("garbage"), Outcome::Rejected)
            ],
            "{env}"
        );
    }
    for env in PRESENCE {
        assert_eq!(changes(env, Old::Presence), [], "{env}");
    }
    assert_eq!(changes(FP8_SSM, Old::Verbatim), []);
}

/// 2026-10-17: The FP8 SSM mode at each value it names, the empty value and a padded one.
#[test]
fn the_fp8_ssm_mode_reads_as_before_at_every_named_value() {
    for v in ["", "1", "both", "decode", "0", " both", "BOTH", "prefill"] {
        let s = settings_from(&[(FP8_SSM, v)]).expect("a text lever takes any value");
        assert_eq!(
            NemotronLevers::from_settings(&s),
            old_at(FP8_SSM, Some(v)),
            "{v:?}"
        );
    }
}

/// 2026-10-17: The lists cover every variable the old reads read.
#[test]
fn the_probe_lists_cover_every_lever_the_old_reads_read() {
    let read = core::cell::RefCell::new(Vec::new());
    let _ = from_values(|var| {
        read.borrow_mut().push(var.to_string());
        None
    });
    let mut read = read.into_inner();
    read.sort();
    let mut listed: Vec<&str> = OPT_OUT.iter().chain(&PRESENCE).copied().collect();
    listed.push(FP8_SSM);
    listed.sort_unstable();
    assert_eq!(read, listed);
}
