// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Tests of [`GemmDispatch`]: the resolution rules, the cuBLASLt scope grammar, and
//! the equivalence of [`GemmDispatch::from_settings`] with the environment reader it replaced.
//!
//! Owner: model-layers ops.
//! Invariants: settings come from a fixed map (`lever_equivalence::settings_from`), never the
//! process environment.

use super::*;
use crate::lever_equivalence::{Old, check, settings_from};

/// 2026-10-17: The reader `from_settings` replaced, frozen as it was: a switch is on only at
/// exactly `1`.
fn old_from_values(mut value: impl FnMut(&str) -> Option<String>) -> GemmDispatch {
    fn on(value: &mut impl FnMut(&str) -> Option<String>, var: &str) -> bool {
        value(var).as_deref() == Some("1")
    }
    let all_nvfp4 = on(&mut value, "METRALE_CUTLASS_NVFP4_GEMM");
    GemmDispatch {
        w4a16_variant: match value("METRALE_W4A16_VARIANT").as_deref() {
            Some("v1") => 1,
            Some("v2") => 2,
            Some("v3") => 3,
            _ => 0,
        },
        fp8_blockscaled_prefill: !on(&mut value, "METRALE_FP8_SINGLE_SCALE"),
        cublas: parse_cublas_scope(value("METRALE_CUBLAS_GEMM").as_deref()).0,
        cublas_fp8: on(&mut value, "METRALE_CUBLAS_FP8"),
        cutlass_gemm: on(&mut value, "METRALE_CUTLASS_GEMM"),
        cutlass_nvfp4_gemm: all_nvfp4,
        cutlass_nvfp4_qkvz: all_nvfp4 || on(&mut value, "METRALE_CUTLASS_NVFP4_QKVZ"),
        cutlass_nvfp4_attn_q: all_nvfp4 || on(&mut value, "METRALE_CUTLASS_NVFP4_ATTN_Q"),
        cutlass_nvfp4_attn_kv: all_nvfp4 || on(&mut value, "METRALE_CUTLASS_NVFP4_ATTN_KV"),
        cutlass_nvfp4_attn_o: all_nvfp4 || on(&mut value, "METRALE_CUTLASS_NVFP4_ATTN_O"),
        cutlass_nvfp4_ssm_out: on(&mut value, "METRALE_CUTLASS_NVFP4_SSM_OUT"),
    }
}

const CASES: [(&str, Old); 11] = [
    ("METRALE_W4A16_VARIANT", Old::Verbatim),
    ("METRALE_FP8_SINGLE_SCALE", Old::OptIn),
    ("METRALE_CUBLAS_GEMM", Old::Verbatim),
    ("METRALE_CUBLAS_FP8", Old::OptIn),
    ("METRALE_CUTLASS_GEMM", Old::OptIn),
    ("METRALE_CUTLASS_NVFP4_GEMM", Old::OptIn),
    ("METRALE_CUTLASS_NVFP4_QKVZ", Old::OptIn),
    ("METRALE_CUTLASS_NVFP4_ATTN_Q", Old::OptIn),
    ("METRALE_CUTLASS_NVFP4_ATTN_KV", Old::OptIn),
    ("METRALE_CUTLASS_NVFP4_ATTN_O", Old::OptIn),
    ("METRALE_CUTLASS_NVFP4_SSM_OUT", Old::OptIn),
];

#[test]
fn every_dispatch_lever_resolves_as_its_old_reader_or_as_a_listed_change() {
    for (env, kind) in CASES {
        let changes = check(
            env,
            kind,
            |probe| {
                old_from_values(|name| (name == env).then(|| probe.map(str::to_owned)).flatten())
            },
            GemmDispatch::from_settings,
        );
        eprintln!("{env}: {changes:?}");
    }
}

#[test]
fn the_cases_cover_every_row_this_resolver_owns() {
    for spec in metrale_settings::groups::by_name() {
        if spec.reader == "crates/model-layers/src/layers/ops/dispatch_config.rs" {
            assert!(
                CASES.iter().any(|(env, _)| *env == spec.env),
                "{} is read here but has no equivalence case",
                spec.env
            );
        }
    }
}

fn resolve(values: &[(&str, &str)]) -> GemmDispatch {
    GemmDispatch::from_settings(&settings_from(values).expect("the probe values resolve"))
}

#[test]
fn defaults_have_only_blockscaled_prefill_on() {
    let d = GemmDispatch::defaults();
    assert_eq!(
        resolve(&[]),
        d,
        "absent environment uses the public default"
    );
    assert_eq!(
        d,
        GemmDispatch {
            fp8_blockscaled_prefill: true,
            cublas: CublasScope::OFF,
            cublas_fp8: false,
            cutlass_gemm: false,
            cutlass_nvfp4_gemm: false,
            cutlass_nvfp4_qkvz: false,
            cutlass_nvfp4_attn_q: false,
            cutlass_nvfp4_attn_kv: false,
            cutlass_nvfp4_attn_o: false,
            cutlass_nvfp4_ssm_out: false,
            w4a16_variant: 0,
        }
    );
}

#[test]
fn the_umbrella_flag_implies_the_per_projection_ones() {
    let d = resolve(&[("METRALE_CUTLASS_NVFP4_GEMM", "1")]);
    assert!(d.cutlass_nvfp4_gemm);
    assert!(d.cutlass_nvfp4_qkvz);
    assert!(d.cutlass_nvfp4_attn_qkv("q_proj"));
    assert!(d.cutlass_nvfp4_attn_qkv("k_proj"));
    assert!(d.cutlass_nvfp4_attn_qkv("v_proj"));
    assert!(d.cutlass_nvfp4_attn_o);
    // 2026-09-25: SSM-out is not implied by the umbrella flag.
    assert!(!d.cutlass_nvfp4_ssm_out);
}

#[test]
fn per_projection_flags_are_independent() {
    let cases = [
        (
            "METRALE_CUTLASS_NVFP4_QKVZ",
            [true, false, false, false, false],
        ),
        (
            "METRALE_CUTLASS_NVFP4_ATTN_Q",
            [false, true, false, false, false],
        ),
        (
            "METRALE_CUTLASS_NVFP4_ATTN_KV",
            [false, false, true, false, false],
        ),
        (
            "METRALE_CUTLASS_NVFP4_ATTN_O",
            [false, false, false, true, false],
        ),
        (
            "METRALE_CUTLASS_NVFP4_SSM_OUT",
            [false, false, false, false, true],
        ),
    ];
    for (name, expected) in cases {
        let d = resolve(&[(name, "1")]);
        assert_eq!(
            [
                d.cutlass_nvfp4_qkvz,
                d.cutlass_nvfp4_attn_q,
                d.cutlass_nvfp4_attn_kv,
                d.cutlass_nvfp4_attn_o,
                d.cutlass_nvfp4_ssm_out,
            ],
            expected,
            "{name} must not enable a neighboring projection"
        );
    }
}

#[test]
fn non_nvfp4_flags_map_independently_and_single_scale_is_inverted() {
    let cases = [
        ("METRALE_CUBLAS_GEMM", [true, false, false]),
        ("METRALE_CUBLAS_FP8", [false, true, false]),
        ("METRALE_CUTLASS_GEMM", [false, false, true]),
    ];
    for (name, expected) in cases {
        let d = resolve(&[(name, "1")]);
        assert_eq!(
            [d.cublas.any(), d.cublas_fp8, d.cutlass_gemm],
            expected,
            "{name} must not enable a neighboring GEMM path"
        );
        assert!(d.fp8_blockscaled_prefill);
    }
    assert!(!resolve(&[("METRALE_FP8_SINGLE_SCALE", "1")]).fp8_blockscaled_prefill);
}

fn scope(raw: &str) -> CublasScope {
    resolve(&[("METRALE_CUBLAS_GEMM", raw)]).cublas
}

/// 2026-09-25: Each spelling in the [`CublasScope`] table maps to its
/// family set.
#[test]
fn the_scope_grammar_maps_each_spelling_to_its_family_set() {
    let f = |ffn, attn, ssm, head| CublasScope {
        ffn,
        attn,
        ssm,
        head,
    };
    let cases: [(&str, CublasScope); 13] = [
        ("all", CublasScope::ALL),
        ("1", CublasScope::ALL),
        ("true", CublasScope::ALL),
        ("ALL", CublasScope::ALL),
        ("off", CublasScope::OFF),
        ("0", CublasScope::OFF),
        ("false", CublasScope::OFF),
        ("", CublasScope::OFF),
        ("ffn", f(true, false, false, false)),
        ("attn", f(false, true, false, false)),
        ("ssm", f(false, false, true, false)),
        ("head", f(false, false, false, true)),
        ("ffn,attn", f(true, true, false, false)),
    ];
    for (raw, expected) in cases {
        assert_eq!(scope(raw), expected, "METRALE_CUBLAS_GEMM={raw:?}");
    }
    assert_eq!(scope(" ffn , ssm "), f(true, false, true, false));
    // 2026-09-25: A union, so `off` clears nothing; a clearing token would
    // make the result depend on token order.
    assert_eq!(scope("ffn,off"), f(true, false, false, false));
}

/// 2026-09-25: A typo must not widen the set; it is dropped and reported
/// as typed, lowercased.
#[test]
fn unknown_families_are_dropped_and_reported_never_widening_the_set() {
    assert_eq!(scope("junk"), CublasScope::OFF);
    assert_eq!(
        scope("ffn,junk"),
        CublasScope {
            ffn: true,
            ..CublasScope::OFF
        }
    );
    let (resolved, unknown) = parse_cublas_scope(Some("ffn, FNN ,bogus"));
    assert_eq!(
        resolved,
        CublasScope {
            ffn: true,
            ..CublasScope::OFF
        }
    );
    assert_eq!(
        unknown,
        vec!["fnn".to_owned(), "bogus".to_owned()],
        "the warning must name what the operator typed, lowercased"
    );
}

/// 2026-09-25: An absent variable and `off` both resolve to the empty set
/// with no unknown token to warn about.
#[test]
fn an_absent_variable_resolves_to_the_empty_set_silently() {
    assert_eq!(parse_cublas_scope(None), (CublasScope::OFF, Vec::new()));
    assert_eq!(
        parse_cublas_scope(Some("off")),
        (CublasScope::OFF, Vec::new())
    );
    assert!(!CublasScope::OFF.any());
    assert!(CublasScope::ALL.any());
}

#[test]
fn w4a16_variants_accept_only_documented_spellings() {
    for (value, expected) in [
        ("v1", 1),
        ("v2", 2),
        ("v3", 3),
        ("1", 0),
        ("V1", 0),
        ("unknown", 0),
    ] {
        assert_eq!(
            resolve(&[("METRALE_W4A16_VARIANT", value)]).w4a16_variant,
            expected,
            "value {value}"
        );
    }
}

#[test]
fn an_unknown_projection_label_falls_back_to_the_umbrella_flag() {
    assert!(!GemmDispatch::defaults().cutlass_nvfp4_attn_qkv("mystery"));
    let d = GemmDispatch {
        cutlass_nvfp4_gemm: true,
        ..GemmDispatch::defaults()
    };
    assert!(d.cutlass_nvfp4_attn_qkv("mystery"));
}
