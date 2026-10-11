// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Tests for the calibration profile: ratio math, fleet
//! aggregation, metric extraction, and the TOML round trip.
//!
//! Owner: bench hardware.
//! Invariants: none beyond the types.

use super::*;
use std::collections::BTreeMap;

fn entry(decode: f64, cold: f64, warm: f64, energy: f64, idle: f64) -> ProfileEntry {
    let raw = Raw {
        decode_tok_s: Some(decode),
        prefill_cold32k_ms: Some(cold),
        restore_warm32k_ms: Some(warm),
        energy_c1_j_per_tok: Some(energy),
        idle_power_w: Some(idle),
    };
    ProfileEntry {
        git_sha: "abc123".into(),
        profile: BoxProfile::new(1_000, raw, &raw),
    }
}

#[test]
fn a_lone_box_is_its_own_reference_so_every_ratio_is_one() {
    let p = entry(20.0, 6800.0, 340.0, 0.52, 12.4).profile;
    assert_eq!(p.decode_tok_s_ratio, Some(1.0));
    assert_eq!(p.prefill_cold32k_ms_ratio, Some(1.0));
    assert_eq!(p.restore_warm32k_ms_ratio, Some(1.0));
    assert_eq!(p.energy_c1_ratio, Some(1.0));
    assert_eq!(p.idle_power_w_ratio, Some(1.0));
    assert_eq!(p.stability_sigma_pct, None, "not yet measured, v1");
}

/// 2026-10-04: The dgx2/dgx3 shape: dgx2 reads 9% more energy than the fleet
/// mean of the two, and the ratio says so in the documented direction (higher
/// = costlier for energy, unlike decode tok/s where higher = faster).
#[test]
fn a_box_above_the_fleet_mean_reads_above_one_in_the_documented_direction() {
    let mut profiles = BTreeMap::new();
    profiles.insert(
        "gb10@dgx3".to_string(),
        entry(23.44, 6600.0, 137.0, 0.40, 11.0),
    );
    let dgx2_raw = Raw {
        decode_tok_s: Some(22.78),
        prefill_cold32k_ms: Some(6676.0),
        restore_warm32k_ms: Some(345.0),
        energy_c1_j_per_tok: Some(0.436), // ~9% above dgx3's 0.40
        idle_power_w: Some(11.0),
    };
    let reference = Raw::fleet_mean(&profiles, Some(&dgx2_raw));
    let dgx2 = BoxProfile::new(2_000, dgx2_raw, &reference);
    // 2026-10-04: energy_c1 ratio to the TWO-box mean (0.418) is above 1: dgx2
    // reads worse than the fleet it is part of.
    assert!(
        dgx2.energy_c1_ratio.unwrap() > 1.0,
        "{:?}",
        dgx2.energy_c1_ratio
    );
    // 2026-10-04: decode tok/s: dgx2 is slightly slower than dgx3, so its
    // ratio to the mean is below 1 (lower is worse for this field).
    assert!(dgx2.decode_tok_s_ratio.unwrap() < 1.0);
}

#[test]
fn fleet_mean_of_zero_profiles_and_no_extra_is_entirely_none() {
    let r = Raw::fleet_mean(&BTreeMap::new(), None);
    assert_eq!(r, Raw::default());
}

/// 2026-10-04: A field missing from every profile stays `None`; a field
/// present on only SOME profiles still means over the ones that have it,
/// rather than refusing the whole reference.
#[test]
fn fleet_mean_is_per_field_and_ignores_missing_readings() {
    let mut profiles = BTreeMap::new();
    let mut a = entry(10.0, 100.0, 50.0, 1.0, 10.0);
    a.profile.restore_warm32k_ms = None; // this box never got a warm reading
    a.profile.restore_warm32k_ms_ratio = None;
    profiles.insert("gb10@a".to_string(), a);
    profiles.insert("gb10@b".to_string(), entry(20.0, 200.0, 60.0, 2.0, 20.0));
    let r = Raw::fleet_mean(&profiles, None);
    assert_eq!(r.decode_tok_s, Some(15.0));
    // 2026-10-04: Only box b reports a warm reading, so the mean is just its
    // own value, not averaged against a missing zero.
    assert_eq!(r.restore_warm32k_ms, Some(60.0));
}

/// 2026-10-04: A zero, negative or non-finite reading never yields a ratio —
/// "absent, never zero" extends to the ratio math, not just the collector.
#[test]
fn a_non_positive_or_non_finite_reading_yields_no_ratio() {
    for bad in [0.0, -5.0, f64::NAN, f64::INFINITY] {
        assert_eq!(ratio(Some(bad), Some(10.0)), None, "{bad}");
        assert_eq!(ratio(Some(10.0), Some(bad)), None, "{bad}");
    }
    assert_eq!(ratio(None, Some(10.0)), None);
    assert_eq!(ratio(Some(10.0), None), None);
    assert_eq!(ratio(Some(10.0), Some(5.0)), Some(2.0));
}

#[test]
fn extract_reads_the_exact_keys_each_benchmarks_record_writes() {
    let mut decode_floor = BTreeMap::new();
    decode_floor.insert("server_decode_tok_s".to_string(), 21.3);
    decode_floor.insert("gpu_rail_joules_per_token".to_string(), 0.52);
    decode_floor.insert("gpu_rail_idle_power_w".to_string(), 12.4);
    let mut cold = BTreeMap::new();
    cold.insert("median_ms".to_string(), 6800.0);
    let mut warm = BTreeMap::new();
    warm.insert("median_ms".to_string(), 340.0);

    let raw = extract(LegMetrics {
        decode_floor: Some(&decode_floor),
        high_isl_cold: Some(&cold),
        high_isl_warm: Some(&warm),
    });
    assert_eq!(raw.decode_tok_s, Some(21.3));
    assert_eq!(raw.energy_c1_j_per_tok, Some(0.52));
    assert_eq!(raw.idle_power_w, Some(12.4));
    assert_eq!(raw.prefill_cold32k_ms, Some(6800.0));
    assert_eq!(raw.restore_warm32k_ms, Some(340.0));
}

/// 2026-10-04: A leg that did not run (or whose record lacked the key) leaves
/// its field `None`, not 0.0 — a vacuous decode-floor run must not read as a
/// free, zero-cost, instant box.
#[test]
fn extract_of_a_missing_leg_is_none_not_zero() {
    let raw = extract(LegMetrics {
        decode_floor: None,
        high_isl_cold: None,
        high_isl_warm: None,
    });
    assert_eq!(raw, Raw::default());
}

#[test]
fn a_wrong_key_in_box_profiles_toml_is_refused_by_name() {
    let dir = std::env::temp_dir().join(format!("boxprofiles-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("kernels").join("x1")).unwrap();
    std::fs::write(
        dir.join("kernels").join("x1").join("BOX_PROFILES.toml"),
        "[profiles.\"x1@h\"]\ngit_sha = \"a\"\nmeasured_at = 1\nnonsense_key = 1\n",
    )
    .unwrap();
    assert!(load(&dir, "x1").is_err());
}

#[test]
fn an_absent_box_profiles_toml_is_an_empty_table_not_an_error() {
    let dir = std::env::temp_dir().join(format!("boxprofiles-none-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("kernels").join("x2")).unwrap();
    let p = load(&dir, "x2").expect("absent file is Ok");
    assert!(p.profiles.is_empty());
}

#[test]
fn save_then_load_round_trips_every_field() {
    let dir = std::env::temp_dir().join(format!("boxprofiles-rt-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("kernels").join("x3")).unwrap();
    let mut profiles = BoxProfiles::default();
    profiles
        .profiles
        .insert("x3@dgx9".to_string(), entry(1.0, 2.0, 3.0, 4.0, 5.0));
    save(&dir, "x3", &profiles).unwrap();
    let back = load(&dir, "x3").unwrap();
    assert_eq!(back, profiles);
}

#[test]
fn save_without_a_kernels_directory_is_refused_by_name() {
    let dir = std::env::temp_dir().join(format!("boxprofiles-missing-{}", std::process::id()));
    let err = save(&dir, "nope", &BoxProfiles::default()).unwrap_err();
    assert!(format!("{err:#}").contains("nope"), "{err:#}");
}

/// 2026-10-04: No profile is exactly one concern, `Missing`, never anything
/// else — there is nothing to check a ratio against.
#[test]
fn no_profile_is_missing_and_nothing_else() {
    let concerns = health(None, &V1_BANDS, 1_000_000);
    assert_eq!(concerns, vec![HealthConcern::Missing]);
}

#[test]
fn a_fresh_in_band_profile_has_no_concerns() {
    let p = entry(20.0, 6800.0, 340.0, 0.52, 12.4).profile;
    let concerns = health(Some(&p), &V1_BANDS, p.measured_at + 1);
    assert!(concerns.is_empty(), "{concerns:?}");
}

/// 2026-10-04: Explicitly the dgx2/dgx3 shape: a 9% energy ratio is outside
/// the ±5% energy band (unlike decode/prefill at their tighter ±3%, this one
/// alone should already catch it).
#[test]
fn an_out_of_band_energy_ratio_is_flagged_by_name() {
    let mut p = entry(20.0, 6800.0, 340.0, 0.52, 12.4).profile;
    p.energy_c1_ratio = Some(1.09);
    let concerns = health(Some(&p), &V1_BANDS, p.measured_at);
    assert_eq!(
        concerns,
        vec![HealthConcern::OutOfBand {
            dimension: "C1 energy",
            ratio: 1.09,
            band: 0.05,
        }]
    );
    assert!(concerns[0].to_string().contains("C1 energy"));
    assert!(concerns[0].to_string().contains("above"));
}

#[test]
fn a_ratio_below_one_reads_below_not_above() {
    let mut p = entry(20.0, 6800.0, 340.0, 0.52, 12.4).profile;
    p.decode_tok_s_ratio = Some(0.90); // 10% slower than the fleet
    let concerns = health(Some(&p), &V1_BANDS, p.measured_at);
    assert!(concerns[0].to_string().contains("below"), "{concerns:?}");
}

#[test]
fn every_dimension_can_be_flagged_at_once() {
    let mut p = entry(20.0, 6800.0, 340.0, 0.52, 12.4).profile;
    p.decode_tok_s_ratio = Some(1.50);
    p.prefill_cold32k_ms_ratio = Some(1.50);
    p.restore_warm32k_ms_ratio = Some(1.50);
    p.energy_c1_ratio = Some(1.50);
    let concerns = health(Some(&p), &V1_BANDS, p.measured_at);
    assert_eq!(concerns.len(), 4, "{concerns:?}");
}

/// 2026-10-04: A dimension with no ratio at all (that leg never measured) is
/// not itself a concern — only a genuinely out-of-band ratio is.
#[test]
fn a_dimension_with_no_ratio_is_not_flagged() {
    let mut p = entry(20.0, 6800.0, 340.0, 0.52, 12.4).profile;
    p.restore_warm32k_ms_ratio = None;
    p.restore_warm32k_ms = None;
    let concerns = health(Some(&p), &V1_BANDS, p.measured_at);
    assert!(concerns.is_empty(), "{concerns:?}");
}

#[test]
fn a_profile_older_than_the_limit_is_stale() {
    let p = entry(20.0, 6800.0, 340.0, 0.52, 12.4).profile;
    let now = p.measured_at + MAX_PROFILE_AGE_S + 1;
    let concerns = health(Some(&p), &V1_BANDS, now);
    assert_eq!(
        concerns,
        vec![HealthConcern::Stale {
            age_s: MAX_PROFILE_AGE_S + 1,
            max_age_s: MAX_PROFILE_AGE_S,
        }]
    );
    assert!(concerns[0].to_string().contains("30"));
}

#[test]
fn a_profile_exactly_at_the_age_limit_is_not_yet_stale() {
    let p = entry(20.0, 6800.0, 340.0, 0.52, 12.4).profile;
    let now = p.measured_at + MAX_PROFILE_AGE_S;
    assert!(health(Some(&p), &V1_BANDS, now).is_empty());
}
