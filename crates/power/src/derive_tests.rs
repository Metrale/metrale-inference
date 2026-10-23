// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The steady-state derivation against synthetic traces with known truth: W and J per
//! launch recovered within 3 % with intervals that cover the truth; interleaved baselines cancel
//! a linear static drift (the sabotage target: a one-sided baseline fails this test); a missing
//! idle B refuses the row; short windows are refused; sensor disagreement is flagged; the row is
//! a pure function of the trace text (HPAR).

use alloc::string::ToString;
use alloc::vec::Vec;

use metrale_error::ErrorCode;

use crate::derive::derive;
use crate::policy::{OF_RECORD, REFUSAL_FLOOR, WindowPolicy};
use crate::row::Label;
use crate::synth_tests::{BASE, Synth};
use crate::trace::{Phase, PowerTrace, sha256_hex};

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

fn row_of(t: &PowerTrace) -> crate::derive::Derived {
    derive(t, &sha256_hex(t.encode().as_bytes())).unwrap()
}

#[test]
fn a_synthetic_trace_recovers_w_and_j_per_launch_within_three_percent() {
    for (dyn_w, launch_us) in [(5.3, 4613.0), (31.0, 837.0)] {
        let s = Synth {
            dyn_w,
            launch_us,
            ..BASE
        };
        let d = row_of(&s.record());
        let (w, jt, jd) = s.truth(1);
        let r = &d.row;
        assert!(rel(r.avg_w, w) < 0.03, "W {} vs {w}", r.avg_w);
        assert!(rel(r.energy_j, jt) < 0.03, "J total {} vs {jt}", r.energy_j);
        assert!(
            rel(r.energy_dyn_j, jd) < 0.03,
            "J dyn {} vs {jd}",
            r.energy_dyn_j
        );
        assert!(
            r.energy_ci_j.lo <= jt && jt <= r.energy_ci_j.hi,
            "{:?} misses {jt}",
            r.energy_ci_j
        );
        assert!(
            r.energy_dyn_ci_j.lo <= jd && jd <= r.energy_dyn_ci_j.hi,
            "{:?} misses {jd}",
            r.energy_dyn_ci_j
        );
        assert_eq!(r.energy_label, Label::Measured);
        assert_eq!(r.energy_samples, 3);
        assert_eq!(r.power_source.as_str(), "nvml_gpu_rail_counter");
        assert_eq!(r.energy_by_source["nvml_gpu_rail_counter"], r.energy_j);
    }
}

#[test]
fn interleaved_baselines_cancel_a_linear_static_drift() {
    // 2026-10-22: Static power climbs 3 W over the run (thermal). The A-B mean is the static
    // power at the load's midpoint, so the dynamic estimate stays unbiased; a one-sided baseline
    // (A only) would under-read it by ~0.38 W of 5.3 W (7 %).
    let s = Synth {
        static_drift_w: 3.0,
        noise_w: 0.1,
        ..BASE
    };
    let d = row_of(&s.record());
    let (_, _, jd) = s.truth(1);
    assert!(
        rel(d.row.energy_dyn_j, jd) < 0.03,
        "J dyn {} vs {jd}",
        d.row.energy_dyn_j
    );
}

#[test]
fn a_missing_idle_b_refuses_the_dynamic_row() {
    let mut t = BASE.record();
    let last = t
        .windows
        .iter()
        .rposition(|w| w.phase == Phase::Idle)
        .unwrap();
    t.windows.remove(last);
    let e = derive(&t, "x").unwrap_err();
    assert_eq!(e.code(), ErrorCode::EnergyBaselineMissing, "{e}");
}

#[test]
fn baselines_that_drift_apart_refuse_the_row() {
    let s = Synth {
        static_drift_w: 40.0,
        noise_w: 0.05,
        ..BASE
    };
    let e = derive(&s.record(), "x").unwrap_err();
    assert_eq!(e.code(), ErrorCode::EnergyBaselineDrift, "{e}");
}

#[test]
fn under_length_windows_are_refused_with_the_reason() {
    // 2026-10-22: 2.9 s, then 499 launches; each refused by the floor policy.
    let s = Synth {
        load_s: 2.9,
        idle_s: 4.0,
        ..BASE
    };
    let e = derive(&s.trace(&REFUSAL_FLOOR), "x").unwrap_err();
    assert_eq!(e.code(), ErrorCode::EnergyWindowRefused);
    assert!(e.to_string().contains("shorter than 3 s"), "{e}");
    let s = Synth {
        load_s: 4.0,
        idle_s: 4.0,
        launch_us: 4.0e6 / 499.5,
        ..BASE
    };
    let e = derive(&s.trace(&REFUSAL_FLOOR), "x").unwrap_err();
    assert!(e.to_string().contains("499 launches"), "{e}");
    // 2026-10-22: Just above the floor passes.
    let s = Synth {
        load_s: 4.0,
        idle_s: 4.0,
        launch_us: 1000.0,
        ..BASE
    };
    assert!(derive(&s.trace(&REFUSAL_FLOOR), "x").is_ok());
}

#[test]
fn too_few_counter_updates_are_refused() {
    // 2026-10-22: 29 updates: a policy that demands more updates than a 3 s window holds.
    let p = WindowPolicy {
        min_updates: 31,
        ..REFUSAL_FLOOR
    };
    let s = Synth {
        load_s: 3.0,
        idle_s: 4.0,
        launch_us: 1000.0,
        ..BASE
    };
    let e = derive(&s.trace(&p), "x").unwrap_err();
    assert!(e.to_string().contains("updates is fewer than 31"), "{e}");
}

#[test]
fn a_policy_below_the_floor_is_refused() {
    let p = WindowPolicy {
        min_load_s: 2.0,
        ..OF_RECORD
    };
    assert_eq!(p.check().unwrap_err().code(), ErrorCode::EnergyInputInvalid);
    assert_eq!(WindowPolicy::parse(&OF_RECORD.encode()).unwrap(), OF_RECORD);
}

#[test]
fn a_clamped_sm_clock_refuses_the_run() {
    let s = Synth {
        sm_mhz: 513,
        ..BASE
    };
    assert_eq!(
        derive(&s.record(), "x").unwrap_err().code(),
        ErrorCode::EnergyClockFloor
    );
}

#[test]
fn a_sensor_thirty_one_percent_high_is_flagged_and_four_percent_is_not() {
    let hi = row_of(
        &Synth {
            sensor_bias: 0.31,
            ..BASE
        }
        .record(),
    )
    .row;
    let key = "nvml_gpu_rail_counter~nvml_gpu_rail_sensor";
    assert!(
        rel(hi.sensor_delta_pct[key], 31.0) < 0.1,
        "{}",
        hi.sensor_delta_pct[key]
    );
    assert!(
        hi.energy_flags
            .iter()
            .any(|f| f.starts_with("sensor_delta>10%")),
        "{:?}",
        hi.energy_flags
    );
    let lo = row_of(
        &Synth {
            sensor_bias: 0.04,
            ..BASE
        }
        .record(),
    )
    .row;
    assert!(
        lo.energy_flags
            .iter()
            .all(|f| !f.starts_with("sensor_delta")),
        "{:?}",
        lo.energy_flags
    );
    // 2026-10-22: The sensor reads 4 % high and lags up to one 250 ms update at every edge, as
    // GB10's does: Bland-Altman's bias is the 4 % (of ~15 W mean), the slope stays near 1.
    let a = lo.energy_agreement[key];
    assert!(rel(a.ba_bias_w, 0.04 * 15.1) < 0.15, "{a:?}");
    assert!((0.95..1.10).contains(&a.deming_slope), "{a:?}");
    assert_eq!(a.n, 9);
    assert!(a.ba_bias_w > 0.0 && a.ba_loa_w.lo < a.ba_bias_w && a.ba_bias_w < a.ba_loa_w.hi);
}

#[test]
fn the_row_is_a_pure_function_of_the_trace_text() {
    let t = BASE.record();
    let text = t.encode();
    let back = PowerTrace::decode(&text).unwrap();
    assert_eq!(back, t);
    assert_eq!(back.encode(), text);
    let sha = sha256_hex(text.as_bytes());
    let a = derive(&t, &sha).unwrap().row.toml_lines();
    let b = derive(&back, &sha).unwrap().row.toml_lines();
    assert_eq!(a, b);
    assert!(
        a.iter()
            .any(|l| l == &alloc::format!("trace_sha256 = \"{sha}\""))
    );
}

#[test]
fn a_missing_metadata_key_is_refused_never_defaulted() {
    for k in [
        "platform",
        "scope",
        "policy",
        "seed",
        "ci_level",
        "bootstrap_reps",
        "order",
        "fingerprint",
    ] {
        let mut t = BASE.record();
        t.meta.remove(k);
        let e = derive(&t, "x").unwrap_err();
        assert_eq!(e.code(), ErrorCode::EnergyInputInvalid, "{k}: {e}");
    }
}

#[test]
fn an_autocorrelated_run_has_fewer_effective_samples_than_samples() {
    let d = row_of(&Synth { phi: 0.9, ..BASE }.record());
    let n = d.channels[0].1[0].load.series_w.len() as f64;
    assert!(
        d.row.energy_ess < n / 2.0,
        "ESS {} of {n}",
        d.row.energy_ess
    );
}

#[test]
fn every_source_is_stored_and_the_declared_order_picks_the_headline() {
    let mut t = BASE.record();
    let d = row_of(&t).row;
    assert_eq!(d.energy_by_source.len(), 2);
    // 2026-10-22: Without the counter the sensor heads the row (second in GB10's order).
    t.sources.remove(0);
    let d = row_of(&t).row;
    assert_eq!(d.power_source.as_str(), "nvml_gpu_rail_sensor");
    let names: Vec<&str> = d.energy_by_source.keys().map(|k| k.as_str()).collect();
    assert_eq!(names, ["nvml_gpu_rail_sensor"]);
    // 2026-10-22: A platform whose order names no present source refuses (no guessed fallback).
    t.meta.insert("platform".to_string(), "apple".to_string());
    assert_eq!(
        derive(&t, "x").unwrap_err().code(),
        ErrorCode::EnergyNoPreferredSource
    );
}
