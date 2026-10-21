// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `DsaLevers::from_settings` and `DsaOwnedLevers::from_settings` against the
//! environment readers they replaced, frozen here (`old_*`, unchanged but for returning `None`
//! where they returned an error).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: the probes that read differently are exactly the listed behaviour changes: a
//! strict `0`/`1` switch now also reads the other switch spellings (`true`, `off`) and a padded
//! `1`, an `== "1"` switch now also arms at `true` and refuses a value the switch rule cannot read,
//! and the serve-reference diagnostic now reads `0`/`off` as off and `true`/`on` as on.

use super::*;
use metrale_model_layers::lever_equivalence::{Old, Outcome, check, defaults, settings_from};

/// 2026-10-17: `attend::levers::parse_switch` (P_HI, PIPE, TC, TC_NANOR).
fn old_strict(v: Option<&str>) -> Option<bool> {
    match v {
        None | Some("0") => Some(false),
        Some("1") => Some(true),
        Some(_) => None,
    }
}

/// 2026-10-17: `attend::levers::parse_hb_splits`.
fn old_hb_splits(v: Option<&str>) -> Option<Option<usize>> {
    let Some(v) = v else { return Some(None) };
    match v.trim().parse::<usize>() {
        Ok(n) if (1..=HB_MAX_SPLITS).contains(&n) => Some(Some(n)),
        _ => None,
    }
}

/// 2026-10-17: `attend::check::serve_reference`.
fn old_serve_ref(v: Option<&str>) -> Option<bool> {
    match v {
        None => Some(false),
        Some("1") => Some(true),
        Some(_) => None,
    }
}

/// 2026-10-17: `sp_select::lever::parse_payload`.
fn old_payload(v: Option<&str>, table_on: bool) -> Option<Payload> {
    match v {
        None | Some("0") => Some(Payload::Tokens),
        Some("1") if table_on => Some(Payload::Selected),
        Some(_) => None,
    }
}

/// 2026-10-17: `pairs` and nothing else, read through `field`; `None` when the settings or the
/// levers refuse them.
fn new_with<T>(pairs: &[(&str, &str)], field: impl Fn(&DsaLevers) -> T) -> Option<T> {
    settings_from(pairs)
        .ok()
        .and_then(|s| DsaLevers::from_settings(&s).ok())
        .map(|l| field(&l))
}

/// 2026-10-17: `env` at `probe` and nothing else, read through `field`.
fn new_at<T>(env: &str, probe: Option<&str>, field: impl Fn(&DsaLevers) -> T) -> Option<T> {
    let pairs: Vec<(&str, &str)> = probe.map(|v| (env, v)).into_iter().collect();
    new_with(&pairs, field)
}

type Field<T> = (&'static str, fn(&DsaLevers) -> T);

#[test]
fn the_strict_prefill_switches_read_as_before_and_take_the_switch_spellings() {
    let cases: [Field<bool>; 4] = [
        ("METRALE_GLM_DSA_HB_PREFILL_P_HI", |l| l.prefill.p_hi),
        ("METRALE_GLM_DSA_HB_PREFILL_PIPE", |l| l.prefill.pipe),
        ("METRALE_GLM_DSA_HB_PREFILL_TC", |l| l.prefill.tc),
        ("METRALE_GLM_DSA_TC_NANOR", |l| l.prefill.tc_nanor),
    ];
    for (env, field) in cases {
        for probe in [
            None,
            Some("0"),
            Some("1"),
            Some("garbage"),
            Some(""),
            Some("2"),
        ] {
            assert_eq!(
                new_at(env, probe, field),
                old_strict(probe),
                "{env}={probe:?}"
            );
        }
        for (probe, now) in [("true", true), ("on", true), (" 1", true), ("off", false)] {
            assert_eq!(
                (old_strict(Some(probe)), new_at(env, Some(probe), field)),
                (None, Some(now)),
                "{env}={probe:?}"
            );
        }
    }
}

#[test]
fn the_opt_in_switches_read_as_before_except_the_switch_spellings() {
    let cases: [Field<bool>; 6] = [
        ("METRALE_GLM_DSA_HB_PREFILL_ONE_SPLIT", |l| {
            l.prefill.one_split
        }),
        ("METRALE_GLM_DSA_SCORES_TILED", |l| l.select.scores_tiled),
        ("METRALE_GLM_DSA_SCORES_MMA", |l| l.select.scores_mma),
        ("METRALE_GLM_DSA_SCORES_PIPE", |l| l.select.scores_pipe),
        ("METRALE_GLM_DSA_SELECT_FUSED", |l| l.select.select_fused),
        ("METRALE_GLM_DSA_TOPK_RADIX", |l| l.select.topk_radix),
    ];
    for (env, field) in cases {
        let changes = check(
            env,
            Old::OptIn,
            |probe| probe == Some("1"),
            |s| field(&DsaLevers::from_settings(s).expect("one switch resolves")),
        );
        assert_eq!(
            changes,
            [
                (env, Some("true"), Outcome::Changed),
                (env, Some("garbage"), Outcome::Rejected)
            ],
            "{env}"
        );
    }
}

#[test]
fn the_split_count_reads_as_before() {
    let env = "METRALE_GLM_DSA_DECODE_HB";
    let max = HB_MAX_SPLITS.to_string();
    let over = (HB_MAX_SPLITS + 1).to_string();
    for probe in [
        None,
        Some("1"),
        Some("8"),
        Some(" 8"),
        Some("8 "),
        Some(max.as_str()),
        Some(over.as_str()),
        Some("0"),
        Some("-1"),
        Some(""),
        Some("on"),
        Some("8x"),
    ] {
        assert_eq!(
            new_at(env, probe, |l| l.decode_hb),
            old_hb_splits(probe),
            "{probe:?}"
        );
    }
}

#[test]
fn the_serve_reference_takes_the_switch_spellings() {
    let env = "METRALE_GLM_DSA_DECODE_HB_SERVE_REF";
    for probe in [None, Some("1"), Some("garbage"), Some("2")] {
        assert_eq!(
            new_at(env, probe, |l| l.serve_ref),
            old_serve_ref(probe),
            "{probe:?}"
        );
    }
    for (probe, now) in [("0", false), ("off", false), ("true", true), ("on", true)] {
        assert_eq!(
            (
                old_serve_ref(Some(probe)),
                new_at(env, Some(probe), |l| l.serve_ref)
            ),
            (None, Some(now)),
            "{probe:?}"
        );
    }
}

const TABLE: &str = "4096:1024,8208:2052";

/// 2026-10-17: The table text reaches the one parser untouched, and both structs resolve it.
#[test]
fn the_sp_table_text_reads_as_before() {
    let env = lever::ENV;
    for probe in [
        None,
        Some("0"),
        Some(""),
        Some(TABLE),
        Some(" 4096:1024"),
        Some("garbage"),
    ] {
        let pairs: Vec<(&str, &str)> = probe.map(|v| (env, v)).into_iter().collect();
        let old = lever::parse(probe).ok();
        let owned = settings_from(&pairs)
            .ok()
            .and_then(|s| DsaOwnedLevers::from_settings(&s).ok())
            .map(|o| o.sp_table);
        assert_eq!(owned, old, "{probe:?}");
        assert_eq!(
            new_with(&pairs, |_| ()).is_some(),
            old.is_some(),
            "{probe:?}"
        );
    }
}

#[test]
fn the_sp_payload_reads_as_before_and_takes_the_switch_spellings() {
    let payload = |pairs: &[(&str, &str)]| {
        settings_from(pairs)
            .ok()
            .and_then(|s| DsaOwnedLevers::from_settings(&s).ok())
            .map(|o| o.sp_payload)
    };
    for table_on in [false, true] {
        let with = |sel: Option<&'static str>| {
            let mut pairs = Vec::new();
            if table_on {
                pairs.push((lever::ENV, TABLE));
            }
            if let Some(v) = sel {
                pairs.push((lever::ENV_SEL, v));
            }
            pairs
        };
        for probe in [None, Some("0"), Some("1"), Some("garbage"), Some("")] {
            assert_eq!(
                payload(&with(probe)),
                old_payload(probe, table_on),
                "{probe:?} {table_on}"
            );
        }
        let on = table_on.then_some(Payload::Selected);
        for (probe, now) in [("true", on), ("off", Some(Payload::Tokens))] {
            assert_eq!(
                (
                    old_payload(Some(probe), table_on),
                    payload(&with(Some(probe)))
                ),
                (None, now),
                "{probe:?} {table_on}"
            );
        }
    }
}

/// 2026-10-17: `lever::agreement_value` as it composed the two readers, and a value that tells
/// apart every reachable combination.
#[test]
fn the_sp_agreement_is_the_old_composition() {
    let old = |table: Option<&str>, sel: Option<&str>| {
        let t = lever::parse(table).unwrap();
        let p = old_payload(sel, t.is_some()).unwrap();
        lever::agreement_of(t.as_ref()) ^ u64::from(p == Payload::Selected)
    };
    let mut seen = Vec::new();
    for (table, sel) in [(None, None), (Some(TABLE), None), (Some(TABLE), Some("1"))] {
        let pairs: Vec<(&str, &str)> = [
            table.map(|v| (lever::ENV, v)),
            sel.map(|v| (lever::ENV_SEL, v)),
        ]
        .into_iter()
        .flatten()
        .collect();
        let new = new_with(&pairs, |l| l.sp_agreement).expect("resolves");
        assert_eq!(new, old(table, sel), "{table:?} {sel:?}");
        assert!(!seen.contains(&new), "{table:?} {sel:?} collides");
        seen.push(new);
    }
    assert_eq!(seen[0], 0, "off agrees as 0");
}

#[test]
fn the_check_dir_text_reads_as_before() {
    let changes = check(
        "METRALE_GLM_DSA_DECODE_HB_CHECK",
        Old::Verbatim,
        |probe| probe.map(str::to_string),
        |s| {
            DsaOwnedLevers::from_settings(s)
                .expect("any text resolves")
                .hb_check
        },
    );
    assert!(changes.is_empty(), "{changes:?}");
    assert_eq!(
        new_at("METRALE_GLM_DSA_DECODE_HB_CHECK", Some(" /x "), |_| ()),
        Some(())
    );
    let s = settings_from(&[("METRALE_GLM_DSA_DECODE_HB_CHECK", " /x ")]).unwrap();
    assert_eq!(
        DsaOwnedLevers::from_settings(&s)
            .unwrap()
            .hb_check
            .as_deref(),
        Some(" /x ")
    );
}

/// 2026-10-21: The default-on decode levers (the ceiling launch's radix top-k, the warp
/// scores) are on unset and `0`/`off` turn each off without touching the other.
#[test]
fn the_default_on_decode_levers_turn_off_one_at_a_time() {
    let cases: [Field<bool>; 2] = [
        ("METRALE_GLM_DSA_DECODE_TOPK_RADIX", |l| l.select.decode_topk_radix),
        ("METRALE_GLM_DSA_SCORES_WARP", |l| l.select.scores_warp),
    ];
    for (env, f) in cases {
        assert_eq!(new_at(env, None, f), Some(true), "{env}");
        assert_eq!(new_at(env, Some("0"), f), Some(false), "{env}");
        assert_eq!(new_at(env, Some("off"), f), Some(false), "{env}");
        assert_eq!(new_at(env, Some("1"), f), Some(true), "{env}");
        assert_eq!(new_at(env, Some("garbage"), f), None, "{env}");
        for (other, g) in cases {
            if other != env {
                assert_eq!(new_at(env, Some("0"), g), Some(true), "{env}=0 leaves {other}");
            }
        }
    }
}

/// 2026-10-17: Every lever off at the declared defaults, as each old reader read unset.
/// 2026-10-21: except `decode_topk_radix`, a new default-on lever with no old reader.
#[test]
fn the_declared_defaults_are_the_old_unset_values() {
    let s = defaults();
    assert_eq!(
        DsaLevers::from_settings(&s).expect("defaults resolve"),
        DsaLevers {
            decode_hb: None,
            prefill: DsaPrefillLevers {
                one_split: false,
                p_hi: false,
                pipe: false,
                tc: false,
                tc_nanor: false,
            },
            select: DsaSelectLevers {
                scores_tiled: false,
                scores_mma: false,
                scores_pipe: false,
                select_fused: false,
                topk_radix: false,
                decode_topk_radix: true,
                scores_warp: true,
            },
            serve_ref: false,
            sp_agreement: 0,
        }
    );
    assert_eq!(
        DsaOwnedLevers::from_settings(&s).expect("defaults resolve"),
        DsaOwnedLevers {
            sp_table: None,
            sp_payload: Payload::Tokens,
            hb_check: None,
        }
    );
}
