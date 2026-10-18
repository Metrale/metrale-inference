// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `MhcLevers::from_settings` against the environment readers it replaced, frozen
//! here (`old_*`, unchanged but for returning `None` where they returned an error).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: the probes that read differently are exactly the listed behaviour changes: a
//! strict `0`/`1` switch now also reads the other switch spellings (`true`, `off`), an `== "1"`
//! switch now also arms at `true` and refuses a value the switch rule cannot read, and an
//! integer is trimmed before it is parsed.

use super::*;
use metrale_model_layers::lever_equivalence::{Old, Outcome, check, defaults, settings_from};

/// 2026-10-17: The strict switches' reader (`glm5next_sp_mhc::parse` and its copies): unset or
/// `0` off, `1` on, anything else an error.
fn old_strict(v: Option<&str>) -> Option<bool> {
    match v {
        None | Some("0") => Some(false),
        Some("1") => Some(true),
        Some(_) => None,
    }
}

/// 2026-10-17: `glm5next_hc_post_sum_grid::parse`.
fn old_ctas(v: Option<&str>) -> Option<Option<u32>> {
    match v {
        None | Some("0") => Some(None),
        Some(s) => match s.parse::<u32>() {
            Ok(n) if (1..=MAX_CTAS).contains(&n) => Some(Some(n)),
            _ => None,
        },
    }
}

/// 2026-10-17: `mhc_slice::parse_slice`.
fn old_slice(v: Option<&str>) -> Option<Option<usize>> {
    match v {
        None | Some("0") => Some(None),
        Some(s) => s.parse::<usize>().ok().map(Some),
    }
}

/// 2026-10-17: The levers with `pairs` set and nothing else; `None` when the settings or the
/// levers refuse them.
fn new(pairs: &[(&str, &str)]) -> Option<MhcLevers> {
    settings_from(pairs)
        .ok()
        .and_then(|s| MhcLevers::from_settings(&s).ok())
}

/// 2026-10-17: `env` at `probe` (with `METRALE_GLM_SP_MHC=1` beside it unless `env` is that
/// lever, so the dependants resolve), read through `field`.
fn new_at<T>(env: &str, probe: Option<&str>, field: impl Fn(&MhcLevers) -> T) -> Option<T> {
    let mut pairs = Vec::new();
    if env != "METRALE_GLM_SP_MHC" {
        pairs.push(("METRALE_GLM_SP_MHC", "1"));
    }
    if let Some(v) = probe {
        pairs.push((env, v));
    }
    new(&pairs).map(|l| field(&l))
}

type Field<T> = (&'static str, fn(&MhcLevers) -> T);

const STRICT: [Field<bool>; 5] = [
    ("METRALE_GLM_SP_MHC", |l| l.sp_mhc),
    ("METRALE_GLM_SP_OVERLAP", |l| l.sp_overlap),
    ("METRALE_GLM_SP_PIPELINE", |l| l.sp_pipeline),
    ("METRALE_GLM_MHC_POST_PRE_FUSED", |l| l.post_pre_fused),
    ("METRALE_GLM_MHC_POST_VEC", |l| l.post_vec),
];

#[test]
fn the_strict_switches_read_as_before_and_take_the_switch_spellings() {
    // 2026-10-17: `METRALE_GLM_MHC_PRE_TILE` had the same reader, as `parse_pre_tile`.
    let tile: Field<bool> = ("METRALE_GLM_MHC_PRE_TILE", |l| l.pre_tile);
    for (env, field) in STRICT.into_iter().chain([tile]) {
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
        // 2026-10-17: The changes: the other switch spellings, and a trimmed `1`.
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
    let cases: [Field<bool>; 3] = [
        ("METRALE_GLM_MHC_PRE_TOK", |l| l.pre_tok),
        ("METRALE_GLM_MHC_PRE_PREFETCH", |l| l.pre_prefetch),
        ("METRALE_GLM_MHC_PRE_SPLITK", |l| l.pre_splitk),
    ];
    for (env, field) in cases {
        let changes = check(
            env,
            Old::OptIn,
            |probe| probe == Some("1"),
            |s| field(&MhcLevers::from_settings(s).expect("one switch resolves")),
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

/// 2026-10-17: The dependants refuse to run without SP-mHC, as the old readers did at first use.
#[test]
fn the_sp_dependants_need_sp_mhc() {
    for (env, _) in &STRICT[1..4] {
        let err = settings_from(&[(env, "1")])
            .map(|s| MhcLevers::from_settings(&s).unwrap_err().to_string())
            .expect("the switch resolves");
        assert!(err.contains("METRALE_GLM_SP_MHC=1"), "{env}: {err}");
        assert!(
            new(&[(env, "0")]).is_some(),
            "{env}=0 refused without SP-mHC"
        );
    }
}

#[test]
fn the_pipeline_refuses_the_layer_dump_and_a_malformed_dump_is_refused() {
    let sp = ("METRALE_GLM_SP_MHC", "1");
    let pipe = ("METRALE_GLM_SP_PIPELINE", "1");
    let dump = ("METRALE_GLM_LAYER_DUMP", "dumps");
    assert!(new(&[sp, pipe, dump]).is_none());
    assert!(new(&[sp, pipe]).is_some_and(|l| l.sp_pipeline));
    assert!(new(&[sp, dump]).is_some_and(|l| !l.sp_pipeline));
    // 2026-10-17: The old pipeline reader read the dump's config, so a row list without a
    // directory failed it with the pipeline off too.
    assert!(new(&[("METRALE_GLM_LAYER_DUMP_ROWS", "3")]).is_none());
}

const INT_PROBES: [Option<&str>; 9] = [
    None,
    Some("0"),
    Some("1"),
    Some("44"),
    Some("4096"),
    Some("4097"),
    Some("-1"),
    Some("garbage"),
    Some(""),
];

#[test]
fn the_post_sum_cap_reads_as_before_but_trims() {
    let env = "METRALE_GLM_MHC_POST_SUM_CTAS";
    for probe in INT_PROBES {
        assert_eq!(
            new_at(env, probe, |l| l.post_sum_ctas),
            old_ctas(probe),
            "{probe:?}"
        );
    }
    assert_eq!(
        (
            old_ctas(Some(" 8")),
            new_at(env, Some(" 8"), |l| l.post_sum_ctas)
        ),
        (None, Some(Some(8)))
    );
}

#[test]
fn the_pre_slice_reads_as_before_but_trims() {
    let env = "METRALE_GLM_MHC_PRE_SLICE";
    for probe in INT_PROBES {
        assert_eq!(
            new_at(env, probe, |l| l.pre_slice),
            old_slice(probe),
            "{probe:?}"
        );
    }
    assert_eq!(
        (
            old_slice(Some(" 8")),
            new_at(env, Some(" 8"), |l| l.pre_slice)
        ),
        (None, Some(Some(8)))
    );
}

/// 2026-10-17: Every lever off at the declared defaults, as each old reader read unset.
#[test]
fn the_declared_defaults_are_the_old_unset_values() {
    let l = MhcLevers::from_settings(&defaults()).expect("defaults resolve");
    assert_eq!(
        l,
        MhcLevers {
            sp_mhc: false,
            sp_overlap: false,
            sp_pipeline: false,
            post_pre_fused: false,
            post_vec: false,
            post_sum_ctas: None,
            pre_tok: false,
            pre_tile: false,
            pre_prefetch: false,
            pre_splitk: false,
            pre_slice: None,
        }
    );
}
