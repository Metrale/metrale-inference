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
    // 2026-10-22: It and `METRALE_GLM_SP_OVERLAP` are on by default now, so unset no longer reads
    // as the old reader's off: `the_promoted_levers_are_on_by_default_and_0_turns_them_off`.
    let promoted = |env: &str| env == "METRALE_GLM_SP_OVERLAP";
    for (env, field) in STRICT.into_iter().filter(|(env, _)| !promoted(env)) {
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
/// 2026-10-22: Except `pre_tile`, on by default since today.
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
            // 2026-10-22: Default on (`sp_overlap` is too, but off without SP-mHC).
            pre_tile: true,
            pre_prefetch: false,
            pre_c1: true,
            pre_splitk: false,
            pre_slice: None,
        }
    );
}

/// 2026-10-21: `METRALE_GLM_MHC_PRE_C1` is on unless switched off, reads the switch spellings, and
/// refuses a value the switch rule cannot read.
#[test]
fn the_decode_pre_norm_is_on_by_default_and_0_turns_it_off() {
    let at = |probe| new_at("METRALE_GLM_MHC_PRE_C1", probe, |l| l.pre_c1);
    assert_eq!(at(None), Some(true));
    assert_eq!(at(Some("1")), Some(true));
    assert_eq!(at(Some("0")), Some(false));
    assert_eq!(at(Some("off")), Some(false));
    assert_eq!(at(Some("garbage")), None);
}

/// 2026-10-22: `METRALE_GLM_MHC_PRE_TILE` and `METRALE_GLM_SP_OVERLAP` are on by default and `0`
/// turns each off; left at its default, the overlap is off without SP-mHC (no refusal), while a
/// set `1` without it is still refused.
#[test]
fn the_promoted_levers_are_on_by_default_and_0_turns_them_off() {
    let cases: [Field<bool>; 2] = [
        ("METRALE_GLM_MHC_PRE_TILE", |l| l.pre_tile),
        ("METRALE_GLM_SP_OVERLAP", |l| l.sp_overlap),
    ];
    for (env, field) in cases {
        assert_eq!(new_at(env, None, field), Some(true), "{env}");
        assert_eq!(new_at(env, Some("0"), field), Some(false), "{env}");
        assert_eq!(new_at(env, Some("off"), field), Some(false), "{env}");
        assert_eq!(new_at(env, Some("1"), field), Some(true), "{env}");
        assert_eq!(new_at(env, Some("garbage"), field), None, "{env}");
    }
    assert_eq!(new(&[]).map(|l| l.sp_overlap), Some(false));
    assert!(new(&[("METRALE_GLM_SP_OVERLAP", "1")]).is_none());
}

/// 2026-10-22: `METRALE_GLM_MHC_FUSED` is on by default, so `fused_runs` decides per site: the
/// fused kernels in the build and a shape and BF16 `hc_fn` they take run fused; a missing kernel,
/// an FP32 `hc_fn` or a shape they refuse keep the three launches instead of failing.
#[test]
fn the_fused_pre_norm_runs_only_where_the_kernels_take_the_site() {
    use crate::glm5next_mhc::{Glm5NextMhcKernels, Glm5NextMhcSiteWeights};
    use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};
    let gpu = metrale_gpu_runtime::gpu::mock::MockGpuBackend::new();
    let k = Glm5NextMhcKernels::resolve(&gpu, MhcLevers::declared()).unwrap();
    let site = |bf16| Glm5NextMhcSiteWeights {
        hc_fn: DevicePtr(0x100),
        hc_fn_bf16: bf16,
        hc_scale: DevicePtr(0x200),
        hc_base: DevicePtr(0x300),
        mix: DevicePtr(0x400),
        done: DevicePtr(0x480),
        mix_tokens: crate::glm5next_mhc::MHC_MIX_MAX_TOKENS,
    };
    assert!(fused_runs(&k, &site(true), 4096, 4));
    assert!(!fused_runs(&k, &site(false), 4096, 4));
    assert!(!fused_runs(&k, &site(true), 4096, 5));
    assert!(!fused_runs(&k, &site(true), 8194, 4));
    let mut absent = k;
    absent.hc_pre_norm = KernelHandle(0);
    assert!(!fused_runs(&absent, &site(true), 4096, 4));
    let mut absent_rows = k;
    absent_rows.hc_pre_norm_rows = KernelHandle(0);
    assert!(!fused_runs(&absent_rows, &site(true), 4096, 4));
    assert!(
        crate::glm5next_layer::GlmLevers::from_settings(&defaults())
            .unwrap()
            .mhc_fused,
        "default on"
    );
    assert!(
        !crate::glm5next_layer::GlmLevers::from_settings(
            &settings_from(&[("METRALE_GLM_MHC_FUSED", "0")]).unwrap()
        )
        .unwrap()
        .mhc_fused,
        "0 turns it off"
    );
}
