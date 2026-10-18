// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The shared loader levers against the environment readers they replaced, frozen
//! here. Every family's old reader is probed; the only probe that reads differently is
//! `METRALE_HOLO_MOE_GROUPED_CUTLASS=true` for the qwen35 and qwen4_exp readers (exactly `1`),
//! which now arms as laguna's reader did.

use super::*;
use metrale_model_layers::lever_equivalence::{Old, PROBES, check, settings_from};

/// 2026-10-17: The probes plus the empty value (which every text lever reads as set), case and padding variants, and the `full` value laguna's old test refused.
fn probes() -> impl Iterator<Item = Option<&'static str>> {
    PROBES
        .into_iter()
        .chain([Some(""), Some(" OFF "), Some("TRUE"), Some("full")])
}

fn new_at<T>(env: &str, probe: Option<&str>, f: impl Fn(&InferenceServerSettings) -> T) -> T {
    let pairs: Vec<(&str, &str)> = probe.map(|v| (env, v)).into_iter().collect();
    f(&settings_from(&pairs).expect("a text lever accepts any value"))
}

/// 2026-10-17: `laguna/load_layers.rs::cutlass_grouped_moe_enabled`.
fn old_grouped_laguna(v: Option<&str>) -> bool {
    matches!(v, Some("1") | Some("true"))
}

/// 2026-10-17: The qwen35 fast-MoE and `qwen4_exp/ffn.rs` readers.
fn old_grouped_qwen(v: Option<&str>) -> bool {
    v == Some("1")
}

/// 2026-10-17: `laguna::unified_moe_layout_enabled` and the `selectors.rs` copy.
fn old_unified(v: Option<&str>) -> bool {
    v.is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// 2026-10-17: `linear_attn_arms/nvfp4.rs` and `qwen35_dense/gdn_dequant.rs`.
fn old_gdn_bf16(v: Option<&str>) -> bool {
    matches!(v, Some("1"))
}

/// 2026-10-17: `longcat/mod.rs` and `mistral_loader/.../phase_assemble.rs`.
fn old_nvfp4_mla_disabled(v: Option<&str>) -> bool {
    v.map(|v| {
        let v = v.trim().to_ascii_lowercase();
        matches!(v.as_str(), "0" | "false" | "no" | "off")
    })
    .unwrap_or(false)
}

#[test]
fn shared_text_levers_read_as_before() {
    let mut changes = Vec::new();
    changes.extend(check(
        "METRALE_HOLO_MOE_GROUPED_CUTLASS",
        Old::Verbatim,
        old_grouped_laguna,
        grouped_cutlass,
    ));
    changes.extend(check(
        "METRALE_UNIFIED_MOE_LAYOUT",
        Old::Verbatim,
        old_unified,
        unified_moe_layout,
    ));
    changes.extend(check(
        "METRALE_GDN_BF16_WEIGHTS",
        Old::Verbatim,
        old_gdn_bf16,
        gdn_bf16_weights,
    ));
    changes.extend(check(
        "METRALE_NVFP4_MLA",
        Old::Verbatim,
        old_nvfp4_mla_disabled,
        nvfp4_mla_disabled,
    ));
    assert!(changes.is_empty(), "{changes:?}");
    for p in probes() {
        let at = |env, f: fn(&InferenceServerSettings) -> bool| new_at(env, p, f);
        assert_eq!(
            at("METRALE_HOLO_MOE_GROUPED_CUTLASS", grouped_cutlass),
            old_grouped_laguna(p),
            "{p:?}"
        );
        assert_eq!(
            at("METRALE_UNIFIED_MOE_LAYOUT", unified_moe_layout),
            old_unified(p),
            "{p:?}"
        );
        assert_eq!(
            at("METRALE_GDN_BF16_WEIGHTS", gdn_bf16_weights),
            old_gdn_bf16(p),
            "{p:?}"
        );
        assert_eq!(
            at("METRALE_NVFP4_MLA", nvfp4_mla_disabled),
            old_nvfp4_mla_disabled(p),
            "{p:?}"
        );
    }
}

/// 2026-10-17: The qwen readers took exactly `1`; the shared predicate also takes `true`, and
/// that is the only probe on which they part.
#[test]
fn grouped_cutlass_parts_from_the_qwen_reader_only_at_true() {
    let differing: Vec<_> = probes()
        .filter(|&p| {
            new_at("METRALE_HOLO_MOE_GROUPED_CUTLASS", p, grouped_cutlass) != old_grouped_qwen(p)
        })
        .collect();
    assert_eq!(differing, [Some("true")]);
}
