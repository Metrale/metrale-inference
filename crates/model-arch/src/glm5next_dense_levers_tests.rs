// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `DenseLevers::from_settings` and `DenseLevers::check_tiers` against the
//! environment readers they replaced, frozen here (`old_*`, unchanged but for taking the tier as
//! a parameter where they read it).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: the probes that read differently are exactly the listed behaviour changes: a row
//! count now also reads padded and reads `00`/`+0`/`-0` as off, a strict `0`/`1` switch now also
//! reads the other switch spellings, `METRALE_GLM_BATCHM_WIDE` now also switches off at `off` and
//! refuses a value the switch rule cannot read, and `METRALE_GLM_MTP_DRAFTER_TIERS` now also arms
//! at `true` and refuses such a value.

use super::*;
use metrale_model_layers::lever_equivalence::{Old, Outcome, check, defaults, settings_from};

/// 2026-10-17: `glm5next_w4a16_seg::parse_seg`.
fn old_seg(v: Option<&str>, tier_on: bool) -> Result<SegMode, String> {
    let mode = match v {
        None | Some("0") => return Ok(SegMode::Off),
        Some("fuse") => SegMode::Fuse,
        Some("split") => SegMode::Split,
        Some(other) => {
            return Err(format!(
                "METRALE_GLM_W4A16_SEG={other:?}: expected 0, fuse or split"
            ));
        }
    };
    if !tier_on {
        return Err(format!(
            "METRALE_GLM_W4A16_SEG={} needs --dense-quantization w4a16",
            v.unwrap_or_default()
        ));
    }
    Ok(mode)
}

/// 2026-10-17: `glm5next_w4a16_seg::parse_gemv_rows`.
fn old_gemv_rows(v: Option<&str>, tier_on: bool) -> Result<u32, String> {
    let r = match v {
        None | Some("0") => return Ok(0),
        Some(s) => match s.parse::<u32>() {
            Ok(r) if (2..=GEMV_ROWS_MAX).contains(&r) => r,
            _ => {
                return Err(format!(
                    "METRALE_GLM_W4A16_GEMV_ROWS={s:?}: expected 0 or 2..={GEMV_ROWS_MAX}"
                ));
            }
        },
    };
    if !tier_on {
        return Err(format!(
            "METRALE_GLM_W4A16_GEMV_ROWS={r} needs --dense-quantization w4a16"
        ));
    }
    Ok(r)
}

/// 2026-10-17: `glm5next_w4a16_seg::parse_gemv_seg`.
fn old_gemv_seg(v: Option<&str>, tier_on: bool) -> Result<bool, String> {
    match v {
        None | Some("0") => Ok(false),
        Some("1") if tier_on => Ok(true),
        Some("1") => Err("METRALE_GLM_W4A16_GEMV_SEG=1 needs --dense-quantization w4a16".into()),
        Some(other) => Err(format!(
            "METRALE_GLM_W4A16_GEMV_SEG={other:?}: expected 0 or 1"
        )),
    }
}

/// 2026-10-17: The three `parse_min_rows` (`glm5next_w4a16_tc`, `glm5next_w4a16_wide`,
/// `glm5next_fp8_wide`), identical but for the variable, the floor and the tier text.
fn old_min_rows(
    env: &str,
    v: Option<&str>,
    floor: usize,
    tier_on: bool,
    tier: &str,
) -> Result<Option<usize>, String> {
    let r = match v {
        None | Some("0") => return Ok(None),
        Some(s) => match s.parse::<usize>() {
            Ok(r) if r >= floor => r,
            _ => {
                return Err(format!(
                    "{env}={s:?}: expected 0 or a row count of at least {floor}"
                ));
            }
        },
    };
    if !tier_on {
        return Err(format!("{env}={r} needs {tier}"));
    }
    Ok(Some(r))
}

/// 2026-10-17: `glm5next_batch_inv::parse`.
fn old_batch_inv(v: Option<&str>) -> Result<bool, String> {
    match v {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!(
            "METRALE_GLM_BATCH_INVARIANT={other:?}: expected 0 or 1"
        )),
    }
}

const W4A16: &str = "--dense-quantization w4a16";
const FP8: &str = "--dense-quantization fp8 or w4a16, or --dsa-quantization w8a8";
const TC: &str = "METRALE_GLM_W4A16_TC_GEMM_MIN_ROWS";
const WIDE: &str = "METRALE_GLM_W4A16_DEQUANT_GEMM_MIN_ROWS";
const FP8_WIDE: &str = "METRALE_GLM_FP8_DEQUANT_GEMM_MIN_ROWS";

/// 2026-10-17: `pairs` and nothing else under the tiers `(w4a16_dense, fp8_registry)`, read
/// through `field`; `None` when the settings, the levers or the tier check refuse them.
fn new_tier<T>(
    pairs: &[(&str, &str)],
    tiers: (bool, bool),
    field: impl Fn(&DenseLevers) -> T,
) -> Option<T> {
    settings_from(pairs)
        .ok()
        .and_then(|s| DenseLevers::from_settings(&s).ok())
        .filter(|l| l.check_tiers(tiers.0, tiers.1).is_ok())
        .map(|l| field(&l))
}

/// 2026-10-17: `env` at `probe` and nothing else, as [`new_tier`].
fn new_at<T>(
    env: &str,
    probe: Option<&str>,
    tiers: (bool, bool),
    field: impl Fn(&DenseLevers) -> T,
) -> Option<T> {
    let pairs: Vec<(&str, &str)> = probe.map(|v| (env, v)).into_iter().collect();
    new_tier(&pairs, tiers, field)
}

#[test]
fn the_seg_text_and_its_tier_read_as_before() {
    let env = "METRALE_GLM_W4A16_SEG";
    for tier in [false, true] {
        for probe in [
            None,
            Some("0"),
            Some("fuse"),
            Some("split"),
            Some("1"),
            Some("on"),
            Some("Split"),
            Some(" fuse"),
            Some(""),
        ] {
            assert_eq!(
                new_at(env, probe, (tier, tier), |l| l.seg),
                old_seg(probe, tier).ok(),
                "{probe:?} tier {tier}"
            );
        }
    }
}

#[test]
fn the_gemv_rows_read_as_before_except_padded_and_signed_zero() {
    let env = "METRALE_GLM_W4A16_GEMV_ROWS";
    for tier in [false, true] {
        for probe in [
            None,
            Some("0"),
            Some("2"),
            Some("8"),
            Some("+4"),
            Some("1"),
            Some("9"),
            Some("-2"),
            Some("four"),
            Some(""),
        ] {
            assert_eq!(
                new_at(env, probe, (tier, tier), |l| l.gemv_rows),
                old_gemv_rows(probe, tier).ok(),
                "{probe:?} tier {tier}"
            );
        }
        for (probe, now) in [(" 4", 4), ("4 ", 4), ("00", 0), ("+0", 0), ("-0", 0)] {
            let want = (now == 0 || tier).then_some(now);
            assert_eq!(
                (
                    old_gemv_rows(Some(probe), tier).ok(),
                    new_at(env, Some(probe), (tier, tier), |l| l.gemv_rows)
                ),
                (None, want),
                "{probe:?} tier {tier}"
            );
        }
    }
}

type Switch = (&'static str, fn(&DenseLevers) -> bool);

/// 2026-10-17: `METRALE_GLM_W4A16_GEMV_SEG` (needs the W4A16 tier) and
/// `METRALE_GLM_BATCH_INVARIANT` (needs none).
#[test]
fn the_strict_switches_read_as_before_and_take_the_switch_spellings() {
    let cases: [(Switch, bool); 2] = [
        (("METRALE_GLM_W4A16_GEMV_SEG", |l| l.gemv_seg), true),
        (
            ("METRALE_GLM_BATCH_INVARIANT", |l| l.batch_invariant),
            false,
        ),
    ];
    for ((env, field), needs_tier) in cases {
        let old = |v: Option<&str>, tier: bool| {
            if needs_tier {
                old_gemv_seg(v, tier).ok()
            } else {
                old_batch_inv(v).ok()
            }
        };
        for tier in [false, true] {
            for probe in [
                None,
                Some("0"),
                Some("1"),
                Some("garbage"),
                Some(""),
                Some("2"),
            ] {
                assert_eq!(
                    new_at(env, probe, (tier, tier), field),
                    old(probe, tier),
                    "{env}={probe:?} tier {tier}"
                );
            }
            for (probe, now) in [("true", true), ("on", true), (" 1", true), ("off", false)] {
                let want = (!now || tier || !needs_tier).then_some(now);
                assert_eq!(
                    (
                        old(Some(probe), tier),
                        new_at(env, Some(probe), (tier, tier), field)
                    ),
                    (None, want),
                    "{env}={probe:?} tier {tier}"
                );
            }
        }
    }
}

/// 2026-10-17: Each floor lever, its floor, whether it needs the W4A16 tier (else the FP8
/// registry), its tier text and its field.
type Floor = (
    &'static str,
    usize,
    bool,
    &'static str,
    fn(&DenseLevers) -> Option<usize>,
);

const FLOORS: [Floor; 3] = [
    (
        TC,
        crate::glm5next_w4a16_tc::MIN_ROWS_FLOOR,
        true,
        W4A16,
        |l| l.tc_min_rows,
    ),
    (
        WIDE,
        crate::glm5next_w4a16_wide::MIN_ROWS_FLOOR,
        true,
        W4A16,
        |l| l.dequant_min_rows,
    ),
    (
        FP8_WIDE,
        crate::glm5next_fp8_wide::MIN_ROWS_FLOOR,
        false,
        FP8,
        |l| l.fp8_dequant_min_rows,
    ),
];

#[test]
fn the_row_floors_read_as_before_except_padded_and_signed_zero() {
    for (env, floor, w4, tier_text, field) in FLOORS {
        let (at, above, below) = (
            floor.to_string(),
            (floor + 1).to_string(),
            (floor - 1).to_string(),
        );
        let plus = format!("+{floor}");
        let (pad_l, pad_r) = (format!(" {floor}"), format!("{floor} "));
        for on in [false, true] {
            let tiers = (on && w4, on);
            for probe in [
                None,
                Some("0"),
                Some(at.as_str()),
                Some(above.as_str()),
                Some(below.as_str()),
                Some(plus.as_str()),
                Some("1"),
                Some("-3"),
                Some("x"),
                Some(""),
            ] {
                assert_eq!(
                    new_at(env, probe, tiers, field),
                    old_min_rows(env, probe, floor, on, tier_text).ok(),
                    "{env}={probe:?} tier {on}"
                );
            }
            for (probe, now) in [
                (pad_l.as_str(), Some(floor)),
                (pad_r.as_str(), Some(floor)),
                ("00", None),
                ("+0", None),
                ("-0", None),
            ] {
                let want = (now.is_none() || on).then_some(now);
                assert_eq!(
                    (
                        old_min_rows(env, Some(probe), floor, on, tier_text).ok(),
                        new_at(env, Some(probe), tiers, field)
                    ),
                    (None, want),
                    "{env}={probe:?} tier {on}"
                );
            }
        }
    }
}

/// 2026-10-17: `glm5next_dequant_persist::mode` as it composed the persist text with the three
/// floor readers; a load accepts a combination when every reader does.
#[test]
fn the_persist_composition_and_the_floors_it_needs_read_as_before() {
    let old = |pairs: &[(&'static str, &str)], on: bool| -> Option<Persist> {
        let get = |e: &str| pairs.iter().find(|(k, _)| *k == e).map(|(_, v)| *v);
        let floor = |i: usize| {
            let (e, floor, _, text, _) = FLOORS[i];
            old_min_rows(e, get(e), floor, on, text).map(|r| r.is_some())
        };
        let (tc, wide, fp8) = (floor(0).ok()?, floor(1).ok()?, floor(2).ok()?);
        let v = get(dequant_persist::ENV);
        if matches!(v, None | Some("0")) {
            return Some(Persist::Off);
        }
        dequant_persist::parse(v, wide, tc, fp8).ok()
    };
    for on in [false, true] {
        for persist in [
            None,
            Some("0"),
            Some("kda"),
            Some("w4a16"),
            Some("all"),
            Some("ALL"),
            Some(" kda"),
        ] {
            for bits in 0..8u8 {
                let mut pairs: Vec<(&'static str, &str)> = Vec::new();
                if let Some(v) = persist {
                    pairs.push((dequant_persist::ENV, v));
                }
                for (bit, env) in [(1, WIDE), (2, TC), (4, FP8_WIDE)] {
                    if bits & bit != 0 {
                        pairs.push((env, "17"));
                    }
                }
                assert_eq!(
                    new_tier(&pairs, (on, on), |l| l.persist),
                    old(&pairs, on),
                    "{pairs:?} tier {on}"
                );
            }
        }
    }
}

/// 2026-10-17: A lever set without its tier is refused with the text its reader refused it with.
#[test]
fn a_lever_without_its_tier_is_refused_with_the_old_text() {
    let refused = |env: &'static str, v: &str, tiers: (bool, bool)| {
        let s = settings_from(&[(env, v)]).expect("the value resolves");
        let l = DenseLevers::from_settings(&s).expect("the levers resolve");
        l.check_tiers(tiers.0, tiers.1)
            .expect_err("refused")
            .to_string()
    };
    for v in ["fuse", "split"] {
        let env = "METRALE_GLM_W4A16_SEG";
        assert_eq!(
            refused(env, v, (false, true)),
            old_seg(Some(v), false).unwrap_err()
        );
    }
    let env = "METRALE_GLM_W4A16_GEMV_ROWS";
    assert_eq!(
        refused(env, "4", (false, true)),
        old_gemv_rows(Some("4"), false).unwrap_err()
    );
    let env = "METRALE_GLM_W4A16_GEMV_SEG";
    assert_eq!(
        refused(env, "1", (false, true)),
        old_gemv_seg(Some("1"), false).unwrap_err()
    );
    for (env, floor, w4, text, _) in FLOORS {
        let v = (floor + 3).to_string();
        let tiers = (false, w4);
        assert_eq!(
            refused(env, &v, tiers),
            old_min_rows(env, Some(&v), floor, false, text).unwrap_err(),
            "{env}"
        );
    }
}

#[test]
fn batchm_wide_reads_as_before_except_the_switch_spellings() {
    let env = "METRALE_GLM_BATCHM_WIDE";
    let changes = check(
        env,
        Old::OptOut,
        |probe| probe != Some("0"),
        |s| {
            DenseLevers::from_settings(s)
                .expect("one switch resolves")
                .batchm_wide
        },
    );
    assert_eq!(
        changes,
        [
            (env, Some("off"), Outcome::Changed),
            (env, Some("garbage"), Outcome::Rejected)
        ]
    );
}

#[test]
fn the_drafter_tiers_read_as_before_except_the_switch_spellings() {
    let env = "METRALE_GLM_MTP_DRAFTER_TIERS";
    let changes = check(
        env,
        Old::OptIn,
        |probe| probe == Some("1"),
        |s| {
            DenseLevers::from_settings(s)
                .expect("one switch resolves")
                .mtp_drafter_tiers
        },
    );
    assert_eq!(
        changes,
        [
            (env, Some("true"), Outcome::Changed),
            (env, Some("garbage"), Outcome::Rejected)
        ]
    );
}

/// 2026-10-17: Every lever at the value its reader read unset as, and nothing refused on any
/// tier.
#[test]
fn the_declared_defaults_are_the_old_unset_values() {
    let l = DenseLevers::from_settings(&defaults()).expect("defaults resolve");
    assert_eq!(
        l,
        DenseLevers {
            seg: SegMode::Off,
            gemv_rows: 0,
            gemv_seg: false,
            gemv_pipe: true,
            swiglu_down: false,
            dense_seg: crate::glm5next_dense_seg::DenseSeg::Off,
            dense_seg_sms: None,
            tc_min_rows: None,
            dequant_min_rows: None,
            fp8_dequant_min_rows: None,
            persist: Persist::Off,
            batch_invariant: false,
            batchm_wide: true,
            batchm_few: true,
            mtp_drafter_tiers: false,
        }
    );
    for tiers in [(false, false), (false, true), (true, true)] {
        assert!(l.check_tiers(tiers.0, tiers.1).is_ok(), "{tiers:?}");
    }
}
