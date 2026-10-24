// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `Qwen35LoadLevers::from_settings` against the environment reads of
//! `load_layers` and `selectors.rs` it replaced, frozen here as `from_values`.
//!
//! Owner: model-arch weight loader (Qwen3.5).
//! Invariants: the probes that read differently are exactly the listed behaviour changes: a
//! strict-`1` switch now also arms at `true`, the low-memory opt-out now also turns off at
//! `off`, a value the switch rule cannot read is refused at startup, and
//! `METRALE_HOLO_MOE_GROUPED_CUTLASS=true` now arms (`shared_levers_tests`).

use super::*;
use metrale_model_layers::lever_equivalence::{Old, Outcome, PROBES, check, settings_from};

/// 2026-10-17: The reads `from_settings` replaced, unchanged but for their paths: `value` is a
/// variable's text, `None` when unset.
fn from_values(value: impl Fn(&str) -> Option<String>) -> Qwen35LoadLevers {
    let is_one = |k: &str| value(k).as_deref() == Some("1");
    Qwen35LoadLevers {
        low_memory_moe: value("METRALE_HOLO_LOW_MEMORY_MOE").as_deref() != Some("0"),
        fast_moe_layers: value("METRALE_HOLO_FAST_MOE_LAYERS")
            .unwrap_or_else(|| "0-99999".to_string()),
        native_fp8_ssm: is_one("METRALE_HOLO_NATIVE_FP8_SSM"),
        native_fp8_attn: is_one("METRALE_HOLO_NATIVE_FP8_ATTN"),
        force_nvfp4_all: is_one("METRALE_FORCE_NVFP4_ALL"),
        force_nvfp4_moe: is_one("METRALE_FORCE_NVFP4_MOE"),
        fp4_proj_decode: is_one("METRALE_HOLO_FP4_PROJ_DECODE"),
        grouped_cutlass: is_one("METRALE_HOLO_MOE_GROUPED_CUTLASS"),
        fp8_dequant_moe_to_bf16: is_one("METRALE_FP8_DEQUANT_MOE_TO_BF16"),
        fp8_dequant_attn_to_bf16: is_one("METRALE_FP8_DEQUANT_ATTN_TO_BF16"),
        fp8_dequant_layers: value("METRALE_FP8_DEQUANT_LAYERS"),
        unified_moe_layout: value("METRALE_UNIFIED_MOE_LAYOUT")
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true")),
    }
}

const OPT_IN: [&str; 3] = [
    "METRALE_HOLO_NATIVE_FP8_SSM",
    "METRALE_FORCE_NVFP4_ALL",
    "METRALE_FORCE_NVFP4_MOE",
];

const VERBATIM: [&str; 7] = [
    "METRALE_HOLO_FAST_MOE_LAYERS",
    "METRALE_HOLO_NATIVE_FP8_ATTN",
    "METRALE_HOLO_FP4_PROJ_DECODE",
    "METRALE_FP8_DEQUANT_MOE_TO_BF16",
    "METRALE_FP8_DEQUANT_ATTN_TO_BF16",
    "METRALE_FP8_DEQUANT_LAYERS",
    "METRALE_UNIFIED_MOE_LAYOUT",
];

const LOW_MEMORY: &str = "METRALE_HOLO_LOW_MEMORY_MOE";
const GROUPED: &str = "METRALE_HOLO_MOE_GROUPED_CUTLASS";

fn old_at(env: &str, probe: Option<&str>) -> Qwen35LoadLevers {
    from_values(|var| (var == env).then(|| probe.map(str::to_string)).flatten())
}

fn changes(env: &'static str, old: Old) -> Vec<(&'static str, Option<&'static str>, Outcome)> {
    check(
        env,
        old,
        |probe| old_at(env, probe),
        Qwen35LoadLevers::from_settings,
    )
}

#[test]
fn every_lever_reads_as_before_except_the_switch_spellings() {
    for env in OPT_IN {
        assert_eq!(
            changes(env, Old::OptIn),
            [
                (env, Some("true"), Outcome::Changed),
                (env, Some("garbage"), Outcome::Rejected)
            ],
            "{env}"
        );
    }
    assert_eq!(
        changes(LOW_MEMORY, Old::OptOut),
        [
            (LOW_MEMORY, Some("off"), Outcome::Changed),
            (LOW_MEMORY, Some("garbage"), Outcome::Rejected)
        ]
    );
    for env in VERBATIM {
        assert_eq!(changes(env, Old::Verbatim), [], "{env}");
    }
}

/// 2026-10-17: The grouped-CUTLASS lever is the shared predicate, which also arms at `true`;
/// every other probe, and the empty value, reads as the strict-`1` reader did.
#[test]
fn grouped_cutlass_differs_only_at_true() {
    for probe in PROBES.into_iter().chain([Some(""), Some("TRUE")]) {
        let pairs: Vec<(&str, &str)> = probe.map(|v| (GROUPED, v)).into_iter().collect();
        let new = Qwen35LoadLevers::from_settings(&settings_from(&pairs).expect("text lever"));
        assert_eq!(
            new == old_at(GROUPED, probe),
            probe != Some("true"),
            "{probe:?}"
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
    read.dedup();
    let mut listed: Vec<String> = OPT_IN
        .iter()
        .chain(&VERBATIM)
        .chain(&[LOW_MEMORY, GROUPED])
        .map(|v| (*v).to_string())
        .collect();
    listed.sort();
    assert_eq!(read, listed);
}

/// 2026-10-17: The text levers at values where a predicate could part from the old one.
#[test]
fn text_levers_read_as_before_at_edge_values() {
    for env in VERBATIM {
        for v in ["", "1", " 1", "TRUE", "true", "0", "3,5-7", " 9 - 2 ,x"] {
            let s = settings_from(&[(env, v)]).expect("a text lever takes any value");
            assert_eq!(
                Qwen35LoadLevers::from_settings(&s),
                old_at(env, Some(v)),
                "{env}={v:?}"
            );
        }
    }
}

/// 2026-10-17: `linear_attn_arms.rs` reads `METRALE_HOLO_FP8_SSM_DECODE` from the settings
/// switch where it read `== "1"`.
#[test]
fn the_fp8_ssm_decode_overlay_reads_as_before_except_the_switch_spellings() {
    const ENV: &str = "METRALE_HOLO_FP8_SSM_DECODE";
    assert_eq!(
        check(
            ENV,
            Old::OptIn,
            |probe| probe == Some("1"),
            |s| s.kernels.holo_fp8_ssm_decode
        ),
        [
            (ENV, Some("true"), Outcome::Changed),
            (ENV, Some("garbage"), Outcome::Rejected)
        ]
    );
}

/// 2026-10-17: The dequant layer list now parses through `parse_layer_ranges`; the parser it
/// had, frozen here, selects the same layers on every list.
#[test]
fn dequant_layer_selection_reads_as_before() {
    fn old(spec: Option<&str>, layer: usize) -> bool {
        let Some(s) = spec else { return true };
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for part in s.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if let Some((a, b)) = part.split_once('-') {
                if let (Ok(a), Ok(b)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                    ranges.push((a.min(b), a.max(b)));
                }
            } else if let Ok(a) = part.parse::<usize>() {
                ranges.push((a, a));
            }
        }
        ranges.iter().any(|&(a, b)| layer >= a && layer <= b)
    }
    let specs = [
        None,
        Some(""),
        Some("3"),
        Some("31-39"),
        Some("39-31"),
        Some(" 1 , 5 - 7 ,x,,9-"),
        Some("-1"),
        Some("2-x"),
    ];
    for spec in specs {
        for layer in 0..48 {
            assert_eq!(
                super::super::selectors::layer_dequant_selected(spec, layer),
                old(spec, layer),
                "{spec:?} layer {layer}"
            );
        }
    }
}
