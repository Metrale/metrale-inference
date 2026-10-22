// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The `--activation-quantization` grammar: accepted forms, the route at every
//! boundary, the canonical form, and each way a ladder can be malformed.

use super::*;

fn parse(s: &str) -> ActivationQuantization {
    ActivationQuantization::parse(s).unwrap_or_else(|e| panic!("{s:?}: {e:#}"))
}

fn refused(s: &str) -> String {
    format!(
        "{:#}",
        ActivationQuantization::parse(s).expect_err(&format!("{s:?} must be refused"))
    )
}

/// 2026-09-30: Each single format routes every family at every row count to itself.
#[test]
fn a_single_format_is_that_format_everywhere() {
    for f in ActQuantFormat::ALL {
        let a = parse(f.name());
        for fam in ProjFamily::ALL {
            for rows in [1, 2, 7, 64, 65, 128, 256, u32::MAX] {
                assert_eq!(
                    a.route(fam, rows),
                    f,
                    "{} {} rows {rows}",
                    f.name(),
                    fam.name()
                );
            }
            assert_eq!(a.is_invariant(fam), f != ActQuantFormat::Adaptive);
        }
        assert_eq!(a.is_adaptive(), f == ActQuantFormat::Adaptive);
        assert_eq!(a.to_string(), f.name());
    }
}

/// 2026-09-30: The owner's example ladder, checked at and around every boundary.
#[test]
fn a_ladder_routes_each_row_count_to_its_rung() {
    let a = parse("1=bf16;2-8=nvfp4;9-=fp8");
    let want = [
        (1, ActQuantFormat::Bf16),
        (2, ActQuantFormat::Nvfp4),
        (8, ActQuantFormat::Nvfp4),
        (9, ActQuantFormat::Fp8),
        (128, ActQuantFormat::Fp8),
    ];
    for (rows, f) in want {
        for fam in ProjFamily::ALL {
            assert_eq!(a.route(fam, rows), f, "rows {rows}");
        }
    }
    assert!(!a.is_invariant(ProjFamily::Gdn));
    assert!(!a.is_adaptive());
    assert_eq!(a.to_string(), "1=bf16;2-8=nvfp4;9-=fp8");
}

/// 2026-09-30: A family override replaces the base for that family only; the others keep it.
#[test]
fn a_family_override_touches_only_its_family() {
    let a = parse("adaptive,lm_head:bf16,ffn:1-4=adaptive;5-=nvfp4");
    assert_eq!(a.route(ProjFamily::LmHead, 1), ActQuantFormat::Bf16);
    assert_eq!(a.route(ProjFamily::Ffn, 4), ActQuantFormat::Adaptive);
    assert_eq!(a.route(ProjFamily::Ffn, 5), ActQuantFormat::Nvfp4);
    for fam in [ProjFamily::Gdn, ProjFamily::Attn, ProjFamily::Moe] {
        assert_eq!(a.route(fam, 200), ActQuantFormat::Adaptive);
    }
    assert!(a.is_invariant(ProjFamily::LmHead));
    assert!(!a.is_adaptive());
    // 2026-09-30: Families print in their fixed order, whatever the input order.
    assert_eq!(
        a.to_string(),
        "adaptive,ffn:1-4=adaptive;5-=nvfp4,lm_head:bf16"
    );
}

/// 2026-09-30: The canonical form parses back to an equal value, and equal ladders print the
/// same: adjacent rungs of one format merge, an override equal to the base is dropped.
#[test]
fn the_canonical_form_round_trips_and_is_unique() {
    for s in [
        "declared",
        "1=bf16;2-8=nvfp4;9-=fp8",
        "fp8,gdn:bf16,attn:1-16=bf16;17-=fp8",
        "1=bf16;2-=adaptive",
    ] {
        let a = parse(s);
        assert_eq!(parse(&a.to_string()), a, "{s}");
    }
    assert_eq!(parse("1=fp8;2-3=fp8;4-=fp8").to_string(), "fp8");
    assert_eq!(parse("1-=adaptive"), ActivationQuantization::adaptive());
    assert_eq!(parse("bf16,gdn:bf16").to_string(), "bf16");
    assert_eq!(parse(" 1 = bf16 ; 2- = fp8 ").to_string(), "1=bf16;2-=fp8");
}

/// 2026-09-30: The default is `declared`, uniform; recipes pin `adaptive`.
#[test]
fn the_default_is_declared_and_adaptive_is_the_named_ladder() {
    assert_eq!(
        ActivationQuantization::default(),
        ActivationQuantization::uniform(ActQuantFormat::Declared)
    );
    assert!(ActivationQuantization::adaptive().is_adaptive());
    assert_eq!(ActivationQuantization::adaptive().to_string(), "adaptive");
}

/// 2026-09-30: Every malformed ladder is refused, with a reason naming the fault.
#[test]
fn malformed_ladders_are_refused() {
    let cases = [
        ("", "empty ladder"),
        ("fp16", "unknown activation format"),
        ("2-=bf16", "must start at row 1"),
        ("1=bf16;3-=fp8", "must start at row 2"),
        ("1-4=bf16;3-=fp8", "must start at row 5"),
        ("1-4=bf16", "must be open-ended"),
        ("1-=bf16;5-=fp8", "open-ended but rungs follow"),
        ("0-=bf16", "counted from 1"),
        ("1-0=bf16;1-=fp8", "counted from 1"),
        ("3-2=bf16", "must start at row 1"),
        ("1=bf16;2-1=fp8;2-=fp8", "before it starts"),
        ("x-=bf16", "not a whole number"),
        ("1=bf16;fp8", "not RANGE=FORMAT"),
        ("bf16,mlp:fp8", "unknown projection family"),
        ("bf16,ffn", "not FAMILY:LADDER"),
        ("bf16,ffn:fp8,ffn:nvfp4", "overridden twice"),
        ("bf16,ffn:2-=fp8", "must start at row 1"),
    ];
    for (s, why) in cases {
        let e = refused(s);
        assert!(e.contains(why), "{s:?}: {e:?} does not say {why:?}");
        assert!(
            e.contains("--activation-quantization"),
            "{s:?}: {e:?} does not name the flag"
        );
    }
}
