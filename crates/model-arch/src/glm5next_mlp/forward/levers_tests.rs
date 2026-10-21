// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `MlpLevers::from_settings` against the environment readers it replaced, frozen
//! here (`old_*`, unchanged but for returning `None` where they returned an error).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: the probes that read differently are exactly the listed behaviour changes: an
//! `== "1"` switch now also arms at `true`, an `== "0"` switch now also turns off at `off`, and
//! both refuse a value the switch rule cannot read; a strict `0`/`1` switch now also reads the
//! other switch spellings; a number is now trimmed, and a width that is not a number or is
//! negative, and an unknown tile, are now refused where they ran as the default.

use super::*;
use metrale_model_layers::lever_equivalence::{Old, Outcome, check, defaults, settings_from};

/// 2026-10-17: `parse_lever` (`w4a4_grouped`) and `parse_gu_lever` (`w4a4_grouped_gu`).
fn old_strict(v: Option<&str>) -> Option<bool> {
    match v {
        None | Some("0") => Some(false),
        Some("1") => Some(true),
        Some(_) => None,
    }
}

/// 2026-10-17: `row_batch_max` and `prefill_gemm_min_rows` before their clamp: unset or not a
/// number is `default`.
fn old_width(v: Option<&str>, default: usize) -> usize {
    v.and_then(|v| v.parse::<usize>().ok()).unwrap_or(default)
}

/// 2026-10-17: `parse_min_rows` (`w4a4_dense_grouped`), `parse_shared_fork` (`shared_fork`) and
/// `parse_nc` (`w4a4_grouped_nc`): unset or `0` off, else a value `ok` accepts.
fn old_count(v: Option<&str>, ok: impl Fn(usize) -> bool) -> Option<Option<usize>> {
    match v {
        None | Some("0") => Some(None),
        Some(s) => s.parse::<usize>().ok().filter(|r| ok(*r)).map(Some),
    }
}

/// 2026-10-17: `gemm_tile` (`forward_prefill_gemm::tile`): unset or an unknown name is the
/// default.
fn old_tile(v: Option<&str>) -> GemmTile {
    v.and_then(select_gemm_tile).unwrap_or(DEFAULT_GEMM_TILE)
}

/// 2026-10-17: `moe_w4a4_raster`: unset is the deleted `DEFAULT` (`0,4`); a set value goes
/// through the one text parser, whose predicate is unchanged.
fn old_raster(v: Option<&str>) -> Option<Option<RasterGroups>> {
    match v {
        None => Some(Some(RasterGroups { gu: 0, down: 4 })),
        Some(s) => parse_raster(s).ok(),
    }
}

/// 2026-10-17: `pairs` and nothing else, read through `field`; `None` when the settings or the
/// levers refuse them.
fn new_with<T>(pairs: &[(&str, &str)], field: impl Fn(&MlpLevers) -> T) -> Option<T> {
    settings_from(pairs)
        .ok()
        .and_then(|s| MlpLevers::from_settings(&s).ok())
        .map(|l| field(&l))
}

/// 2026-10-17: `env` at `probe` and nothing else, read through `field`.
fn new_at<T>(env: &str, probe: Option<&str>, field: impl Fn(&MlpLevers) -> T) -> Option<T> {
    let pairs: Vec<(&str, &str)> = probe.map(|v| (env, v)).into_iter().collect();
    new_with(&pairs, field)
}

type Field<T> = (&'static str, fn(&MlpLevers) -> T);

#[test]
fn the_opt_in_switches_read_as_before_except_the_switch_spellings() {
    let cases: [Field<bool>; 2] = [
        ("METRALE_NO_GLM_MOE_ROW_BATCH", |l| l.row_batch_off),
        ("METRALE_GLM_MOE_HOST_DISPATCH", |l| l.host_dispatch),
    ];
    for (env, field) in cases {
        let changes = check(
            env,
            Old::OptIn,
            |probe| probe == Some("1"),
            |s| field(&MlpLevers::from_settings(s).expect("one switch resolves")),
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
fn the_opt_out_switches_read_as_before_except_the_switch_spellings() {
    let cases: [Field<bool>; 2] = [
        ("METRALE_GLM_MOE_PREFILL_GEMM", |l| l.prefill_gemm),
        ("METRALE_GLM_MOE_PREFILL_GEMM_EXACT_TILES", |l| {
            l.exact_tiles
        }),
    ];
    for (env, field) in cases {
        let changes = check(
            env,
            Old::OptOut,
            |probe| probe != Some("0"),
            |s| field(&MlpLevers::from_settings(s).expect("one switch resolves")),
        );
        assert_eq!(
            changes,
            [
                (env, Some("off"), Outcome::Changed),
                (env, Some("garbage"), Outcome::Rejected)
            ],
            "{env}"
        );
    }
}

#[test]
fn the_strict_switches_read_as_before_and_take_the_switch_spellings() {
    let cases: [Field<bool>; 2] = [
        ("METRALE_GLM_MOE_W4A4_GROUPED", |l| l.w4a4_grouped),
        ("METRALE_GLM_MOE_W4A4_GU_FUSED", |l| l.gu_fused),
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

/// 2026-10-17: Both widths keep their clamp; a padded number now reads, and a value that is not
/// a non-negative number is refused where it ran as the default.
#[test]
fn the_widths_read_as_before_but_refuse_what_is_not_a_row_count() {
    let max = MOE_ROW_BATCH_MAX_ROWS;
    let cases: [(&str, usize, fn(usize) -> usize, fn(&MlpLevers) -> usize); 2] = [
        (
            "METRALE_GLM_MOE_ROW_BATCH_MAX",
            max,
            |m| m.clamp(1, MOE_ROW_BATCH_MAX_ROWS),
            |l| l.row_batch_max,
        ),
        (
            "METRALE_GLM_MOE_PREFILL_GEMM_MIN_ROWS",
            DEFAULT_GEMM_MIN_ROWS,
            |m| m.max(1),
            |l| l.prefill_gemm_min_rows,
        ),
    ];
    for (env, default, bound, field) in cases {
        let old = |p: Option<&str>| bound(old_width(p, default));
        for probe in [
            None,
            Some("0"),
            Some("1"),
            Some("8"),
            Some("20"),
            Some("200"),
        ] {
            assert_eq!(
                new_at(env, probe, field),
                Some(old(probe)),
                "{env}={probe:?}"
            );
        }
        assert_eq!(
            (old(Some(" 8")), new_at(env, Some(" 8"), field)),
            (default, Some(bound(8))),
            "{env}"
        );
        for probe in ["garbage", "", "-1", "8x"] {
            assert_eq!(
                (old(Some(probe)), new_at(env, Some(probe), field)),
                (default, None),
                "{env}={probe:?}"
            );
        }
    }
}

/// 2026-10-17: The three count levers, with the old acceptance `ok` and the field as an
/// optional count; a padded number and `00` now read as their value.
#[test]
fn the_counts_read_as_before_but_trim_their_number() {
    let floor = MIN_ROWS_FLOOR;
    let cases: [(
        &str,
        fn(usize) -> bool,
        fn(&MlpLevers) -> Option<usize>,
        usize,
    ); 3] = [
        (
            "METRALE_GLM_DENSE_W4A4_GROUPED_MIN_ROWS",
            |r| r >= MIN_ROWS_FLOOR,
            |l| l.dense_grouped_min_rows,
            floor,
        ),
        (
            "METRALE_GLM_SHARED_FORK",
            |r| (1..=SHARED_FORK_MAX_ROWS).contains(&r),
            |l| (l.shared_fork > 0).then_some(l.shared_fork),
            SHARED_FORK_MAX_ROWS,
        ),
        (
            "METRALE_GLM_MOE_W4A4_NC",
            |n| (1..=MAX_NC as usize).contains(&n),
            |l| l.nc.map(|n| n as usize),
            MAX_NC as usize,
        ),
    ];
    for (env, ok, field, top) in cases {
        let top_s = top.to_string();
        let over = (top + 1).to_string();
        let below = (floor - 1).to_string();
        let floor_s = floor.to_string();
        for probe in [
            None,
            Some("0"),
            Some("1"),
            Some(below.as_str()),
            Some(floor_s.as_str()),
            Some(top_s.as_str()),
            Some(over.as_str()),
            Some("-1"),
            Some("garbage"),
            Some(""),
        ] {
            assert_eq!(
                new_at(env, probe, field),
                old_count(probe, ok),
                "{env}={probe:?}"
            );
        }
        let padded = format!(" {top}");
        assert_eq!(
            (
                old_count(Some(&padded), ok),
                new_at(env, Some(&padded), field)
            ),
            (None, Some(Some(top))),
            "{env}"
        );
        assert_eq!(
            (old_count(Some("00"), ok), new_at(env, Some("00"), field)),
            (None, Some(None)),
            "{env}"
        );
    }
}

/// 2026-10-17: The raster text reaches the one parser untouched; unset is the declared `0,4`.
#[test]
fn the_raster_reads_as_before() {
    let env = "METRALE_GLM_MOE_W4A4_RASTER";
    for probe in [
        None,
        Some("0"),
        Some("0,4"),
        Some("16,4"),
        Some("0,4096"),
        Some("16,4097"),
        Some(" 16,4"),
        Some("1"),
        Some("garbage"),
        Some(""),
    ] {
        assert_eq!(
            new_at(env, probe, |l| l.raster),
            old_raster(probe),
            "{probe:?}"
        );
    }
}

/// 2026-10-17: Every tile name the reader took reads as before; an unknown one is now refused.
#[test]
fn the_tile_reads_as_before_but_refuses_an_unknown_name() {
    let env = "METRALE_GLM_MOE_GEMM_TILE";
    let other = GEMM_TILES
        .iter()
        .find(|t| *t != &DEFAULT_GEMM_TILE && *t != &GEMM_TILES[0])
        .expect("a tile that is neither the default nor the base");
    let suffix = other
        .name
        .strip_prefix("moe_w4a16_grouped_gemm_ptrtable_")
        .expect("tile names carry the entry prefix");
    for probe in [
        None,
        Some("base"),
        Some(" BASE "),
        Some(""),
        Some(other.name),
        Some(suffix),
        Some(DEFAULT_GEMM_TILE.name),
    ] {
        assert_eq!(
            new_at(env, probe, |l| l.gemm_tile),
            Some(old_tile(probe)),
            "{probe:?}"
        );
    }
    assert_ne!(
        new_at(env, Some(suffix), |l| l.gemm_tile),
        Some(DEFAULT_GEMM_TILE)
    );
    for probe in ["garbage", "ptrtable_"] {
        assert_eq!(
            (
                old_tile(Some(probe)),
                new_at(env, Some(probe), |l| l.gemm_tile)
            ),
            (DEFAULT_GEMM_TILE, None),
            "{probe:?}"
        );
    }
}

/// 2026-10-17: Every lever at the declared defaults reads as each old reader read unset.
#[test]
fn the_declared_defaults_are_the_old_unset_values() {
    assert_eq!(
        MlpLevers::from_settings(&defaults()).expect("defaults resolve"),
        MlpLevers {
            row_batch_off: false,
            row_batch_max: MOE_ROW_BATCH_MAX_ROWS,
            host_dispatch: false,
            gemm_tile: DEFAULT_GEMM_TILE,
            prefill_gemm: true,
            prefill_gemm_min_rows: DEFAULT_GEMM_MIN_ROWS,
            exact_tiles: true,
            dense_grouped_min_rows: None,
            shared_fork: 0,
            w4a4_grouped: false,
            raster: Some(RasterGroups { gu: 0, down: 4 }),
            gu_fused: false,
            nc: None,
            sweep_serial: true,
            zero_in_swiglu: true,
        }
    );
}
