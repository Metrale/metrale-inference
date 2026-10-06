// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the one-box equivalence rule.
//!
//! Owner: bench hardware.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: The GB10 policy as `kernels/gb10/HARDWARE.toml` declares it: 1 %
/// clock, 5 % memory, 15 °C chassis.
fn speed() -> EquivalencePolicy {
    EquivalencePolicy {
        clock_spread: 0.01,
        mem_spread: 0.05,
        chassis_delta_c: 15.0,
    }
}
use crate::hardware::state::{ExtendedTelemetry, ThermalZone, ThrottleActive};

fn state(chassis: f64, thermal: Option<bool>) -> HardwareState {
    HardwareState {
        sm_clock_max_mhz: Some(3_003.0),
        mem_total_kb: Some(125_000_000),
        extended: ExtendedTelemetry {
            vbios: Some("9A.0B.1E.00.00".into()),
            ..ExtendedTelemetry::default()
        },
        chassis_temps_c: Some(vec![
            ThermalZone {
                name: "acpitz".into(),
                temp_c: chassis,
            },
            ThermalZone {
                name: "acpitz".into(),
                temp_c: chassis - 5.0,
            },
        ]),
        throttle_active: ThrottleActive {
            sw_power_cap: Some(true),
            sw_thermal: thermal,
            hw_thermal: thermal,
            hw_power_brake: thermal,
        },
        ..Default::default()
    }
}

fn hw(gpu: &str, driver: &str) -> Hardware {
    Hardware {
        gpu: gpu.into(),
        driver: driver.into(),
        sm_clock_mhz: None,
        gpu_count: None,
        source: "test".into(),
    }
}

fn gb10(chassis: f64) -> HardwareFingerprint {
    HardwareFingerprint::from_live(
        &hw("NVIDIA GB10", "580.95.05"),
        &state(chassis, Some(false)),
    )
}

#[test]
fn two_healthy_gb10s_are_one_box_and_the_incident_pair_is_not() {
    assert_eq!(equivalent(&gb10(65.0), &gb10(70.0), &speed()), Ok(()));
    // 2026-09-26: 8 °C and 13 °C apart: inside the 15 °C delta.
    assert_eq!(equivalent(&gb10(65.0), &gb10(73.0), &speed()), Ok(()));
    assert_eq!(equivalent(&gb10(55.0), &gb10(68.0), &speed()), Ok(()));
    // 2026-09-26: 65 °C vs 89 °C with a thermal reason on the hot one: both are reported.
    let hot =
        HardwareFingerprint::from_live(&hw("NVIDIA GB10", "580.95.05"), &state(89.0, Some(true)));
    let why = equivalent(&gb10(65.0), &hot, &speed()).unwrap_err();
    assert!(
        why.contains(&Mismatch::ChassisDelta {
            a: 65.0,
            b: 89.0,
            limit: 15.0
        }),
        "{why:?}"
    );
    assert!(
        why.contains(&Mismatch::ThermalAlert { a: false, b: true }),
        "{why:?}"
    );
    // 2026-09-26: 16 °C apart with no throttle reason: the chassis delta alone refuses.
    let why = equivalent(&gb10(65.0), &gb10(81.0), &speed()).unwrap_err();
    assert_eq!(why.len(), 1, "{why:?}");
    assert!(matches!(why[0], Mismatch::ChassisDelta { .. }));
}

#[test]
fn every_static_field_is_checked_and_every_mismatch_is_named() {
    let base = gb10(65.0);
    // 2026-09-26: Negative controls, one field at a time.
    let mut other_gpu = base.clone();
    other_gpu.gpu = "NVIDIA H100".into();
    assert!(matches!(
        equivalent(&base, &other_gpu, &speed()).unwrap_err()[..],
        [Mismatch::Gpu(..)]
    ));
    // 2026-10-04: Full driver string, not major only: dgx2 and dgx3 share major
    // 580 and are exactly the pair that does not read as one box.
    let mut other_driver = base.clone();
    other_driver.driver_full = Some("580.159.03".into());
    assert!(matches!(
        &equivalent(&base, &other_driver, &speed()).unwrap_err()[..],
        [Mismatch::DriverVersion(a, b)] if a == "580.95.05" && b == "580.159.03"
    ));
    let dgx2 = HardwareFingerprint {
        driver_full: Some("580.126.09".into()),
        vbios: Some("9A.0B.1E".into()),
        ..base.clone()
    };
    let dgx3 = HardwareFingerprint {
        driver_full: Some("580.159.03".into()),
        vbios: Some("9A.0B.25".into()),
        ..base.clone()
    };
    let why = equivalent(&dgx2, &dgx3, &speed()).unwrap_err();
    assert!(
        why.contains(&Mismatch::DriverVersion(
            "580.126.09".into(),
            "580.159.03".into()
        )),
        "{why:?}"
    );
    assert!(
        why.contains(&Mismatch::Vbios("9A.0B.1E".into(), "9A.0B.25".into())),
        "{why:?}"
    );
    let mut other_vbios = base.clone();
    other_vbios.vbios = Some("9A.0B.25".into());
    assert!(matches!(
        &equivalent(&base, &other_vbios, &speed()).unwrap_err()[..],
        [Mismatch::Vbios(a, b)] if a == "9A.0B.1E.00.00" && b == "9A.0B.25"
    ));
    // 2026-09-26: `driver_major` keeps only the major version (still used for
    // display in the fleet plan; [`equivalent`] no longer reads it).
    assert_eq!(driver_major("580.126.09"), Some(580));
    assert_eq!(driver_major(""), None);
    assert_eq!(driver_major("unknown"), None);
    let mut clock = base.clone();
    clock.sm_clock_max_mhz = Some(3_003.0 * 0.97);
    assert!(matches!(
        equivalent(&base, &clock, &speed()).unwrap_err()[..],
        [Mismatch::ClockSpread { .. }]
    ));
    let mut clock_ok = base.clone();
    clock_ok.sm_clock_max_mhz = Some(3_003.0 * 0.995);
    assert_eq!(equivalent(&base, &clock_ok, &speed()), Ok(()));
    let mut mem = base.clone();
    mem.mem_total_kb = Some(125_000_000 * 9 / 10);
    assert!(matches!(
        equivalent(&base, &mem, &speed()).unwrap_err()[..],
        [Mismatch::MemSpread { .. }]
    ));
    let mut post = base.clone();
    post.postcheck_valid = Some(false);
    assert!(matches!(
        equivalent(&base, &post, &speed()).unwrap_err()[..],
        [Mismatch::PostcheckInvalid]
    ));
    // 2026-09-26: A valid postcheck on one side and none on the other (live) is fine.
    let mut post_ok = base.clone();
    post_ok.postcheck_valid = Some(true);
    assert_eq!(equivalent(&base, &post_ok, &speed()), Ok(()));
    // 2026-09-26: Every variant's Display is non-empty.
    for m in [
        Mismatch::Gpu("a".into(), "b".into()),
        Mismatch::DriverVersion("a".into(), "b".into()),
        Mismatch::Vbios("a".into(), "b".into()),
        Mismatch::ClockSpread {
            a: 1.0,
            b: 2.0,
            limit: 0.01,
        },
        Mismatch::MemSpread {
            a: 1,
            b: 2,
            limit: 0.05,
        },
        Mismatch::ChassisDelta {
            a: 1.0,
            b: 2.0,
            limit: 15.0,
        },
        Mismatch::ThermalAlert { a: true, b: true },
        Mismatch::PostcheckInvalid,
        Mismatch::Undecidable("gpu"),
    ] {
        assert!(!m.to_string().is_empty());
    }
}

/// 2026-10-04: The incident pair, named: dgx2 (580.126.09, VBIOS 9A.0B.1E) and
/// dgx3 (580.159.03, VBIOS 9A.0B.25) — the box calibration catalogue's
/// documented fleet versions — share driver major 580 and clock/memory/chassis
/// readings close enough to pass every other check, yet are the exact pair
/// measured 6.6-13% apart on J/token. A major-only driver check would wave
/// them through; the full-version-and-VBIOS check must not.
#[test]
fn dgx2_and_dgx3_are_not_equivalent() {
    let dgx2 = HardwareFingerprint::from_live(
        &hw("NVIDIA GB10", "580.126.09"),
        &HardwareState {
            extended: ExtendedTelemetry {
                vbios: Some("9A.0B.1E".into()),
                ..ExtendedTelemetry::default()
            },
            ..state(60.0, Some(false))
        },
    );
    let dgx3 = HardwareFingerprint::from_live(
        &hw("NVIDIA GB10", "580.159.03"),
        &HardwareState {
            extended: ExtendedTelemetry {
                vbios: Some("9A.0B.25".into()),
                ..ExtendedTelemetry::default()
            },
            ..state(60.0, Some(false))
        },
    );
    // 2026-10-04: Same major version, so the old check would have passed them.
    assert_eq!(dgx2.driver_major, dgx3.driver_major);
    let why = equivalent(&dgx2, &dgx3, &speed()).expect_err("not one box");
    assert!(
        why.contains(&Mismatch::DriverVersion(
            "580.126.09".into(),
            "580.159.03".into()
        )),
        "{why:?}"
    );
    assert!(
        why.contains(&Mismatch::Vbios("9A.0B.1E".into(), "9A.0B.25".into())),
        "{why:?}"
    );
}

#[test]
fn a_missing_field_is_undecidable_which_is_not_equivalent() {
    let base = gb10(65.0);
    for (field, strip) in [
        (
            "gpu",
            Box::new(|f: &mut HardwareFingerprint| f.gpu.clear())
                as Box<dyn Fn(&mut HardwareFingerprint)>,
        ),
        ("driver", Box::new(|f| f.driver_full = None)),
        ("vbios", Box::new(|f| f.vbios = None)),
        ("sm_clock_max_mhz", Box::new(|f| f.sm_clock_max_mhz = None)),
        ("mem_total_kb", Box::new(|f| f.mem_total_kb = None)),
        ("throttle reasons", Box::new(|f| f.thermal_alert = None)),
        (
            "chassis temperature",
            Box::new(|f| f.hottest_chassis_c = None),
        ),
    ] {
        let mut stripped = base.clone();
        strip(&mut stripped);
        let why = equivalent(&base, &stripped, &speed()).expect_err(field);
        assert_eq!(why, vec![Mismatch::Undecidable(field)], "{field}");
        let why = equivalent(&stripped, &base, &speed()).expect_err(field);
        assert_eq!(why, vec![Mismatch::Undecidable(field)], "{field}");
    }
    // 2026-09-26: A box with no state at all: from_live over a default state.
    let bare =
        HardwareFingerprint::from_live(&hw("NVIDIA GB10", "580.1"), &HardwareState::default());
    assert!(equivalent(&base, &bare, &speed()).unwrap_err().len() >= 4);
}

#[test]
fn a_record_carries_its_before_capture_and_its_postcheck() {
    // 2026-09-26: The committed decode-floor record (a Speed benchmark) under
    // `test_data/gate-records`, rather than a hand-built one.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("workspace root");
    let dir = root.join("test_data/gate-records/.benchmarks/decode-floor");
    let newest = std::fs::read_dir(&dir)
        .expect("the decode-floor fixture record exists")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .max()
        .expect("at least one record");
    let rec = crate::gate::read_record(&newest).expect("parses");
    let fp = HardwareFingerprint::from_record(&rec);
    assert_eq!(fp.gpu, "NVIDIA GB10");
    assert_eq!(fp.driver_major, Some(580));
    assert_eq!(fp.driver_full.as_deref(), Some("580.159.03"));
    assert_eq!(fp.sm_clock_max_mhz, Some(3_003.0));
    assert!(fp.mem_total_kb.is_some_and(|kb| kb > 100_000_000));
    assert_eq!(fp.thermal_alert, Some(false));
    assert!(fp.hottest_chassis_c.is_some());
    assert_eq!(fp.postcheck_valid, Some(true));
    // 2026-09-26: It is one box with a live GB10 at the same driver, chassis
    // temperature and memory —
    // 2026-10-04: except vbios: this fixture predates vbios capture
    // (`extended` deserializes to its default), so the pair is Undecidable
    // there, not Ok. A historical record cannot be vbios-compared until it is
    // re-measured under this change.
    let mut live = HardwareFingerprint::from_live(
        &hw("NVIDIA GB10", "580.159.03"),
        &state(fp.hottest_chassis_c.unwrap(), Some(false)),
    );
    live.mem_total_kb = fp.mem_total_kb;
    assert_eq!(fp.vbios, None, "the fixture predates vbios capture");
    assert_eq!(
        equivalent(&fp, &live, &speed()),
        Err(vec![Mismatch::Undecidable("vbios")])
    );
    // 2026-09-26: A record with no report is undecidable on each of the five live
    // fields (the four state-derived fields, plus vbios).
    let mut bare = rec.clone();
    bare.hardware_state = None;
    let fp = HardwareFingerprint::from_record(&bare);
    assert_eq!(fp.postcheck_valid, None);
    let why = equivalent(&fp, &live, &speed()).unwrap_err();
    assert!(
        why.iter().all(|m| matches!(m, Mismatch::Undecidable(_))),
        "{why:?}"
    );
    assert_eq!(why.len(), 5);
}
