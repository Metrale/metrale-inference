// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The capability declaration refuses a guess: every field of the universe is
//! declared one way, a sample carrying an undeclared field is refused, and the headline
//! preference resolves only to a present source.

use alloc::string::ToString;
use alloc::vec;

use metrale_error::ErrorCode;

use super::*;
use crate::MonoNs;

fn gb10_counter() -> PowerCapabilities {
    PowerCapabilities::builder(
        SourceId("nvml0.counter".to_string()),
        SourceKind::GpuDriverCounter,
        "580.159.03".to_string(),
    )
    .counter(
        Rail::GpuCore,
        CounterSpec {
            unit_j: 0.001,
            update_period_ms: 100.0,
        },
    )
    .field(Field::SmClock)
    .unavailable(
        Field::Energy(Rail::Module),
        "GB10 NVML exposes the GPU rail only",
    )
    .rest_unavailable(
        &Field::universe(),
        "not reported by nvmlDeviceGetTotalEnergyConsumption",
    )
    .build(&Field::universe())
    .unwrap()
}

#[test]
fn every_field_is_declared_and_the_reason_is_kept() {
    let c = gb10_counter();
    assert_eq!(
        c.declares(Field::Energy(Rail::GpuCore)),
        Declared::Available
    );
    assert_eq!(
        c.declares(Field::Energy(Rail::Module)),
        Declared::Unavailable("GB10 NVML exposes the GPU rail only")
    );
    for f in Field::universe() {
        assert_ne!(c.declares(f), Declared::Undeclared, "{}", f.name());
    }
}

#[test]
fn an_undeclared_field_refuses_the_build() {
    let e = PowerCapabilities::builder(
        SourceId("x".to_string()),
        SourceKind::Hwmon,
        "v".to_string(),
    )
    .field(Field::GpuTemp)
    .build(&Field::universe())
    .unwrap_err();
    assert_eq!(e.code(), ErrorCode::PowerCapabilitiesIncomplete);
}

#[test]
fn a_field_both_available_and_unavailable_refuses_the_build() {
    let e = PowerCapabilities::builder(
        SourceId("x".to_string()),
        SourceKind::Hwmon,
        "v".to_string(),
    )
    .field(Field::GpuTemp)
    .unavailable(Field::GpuTemp, "contradiction")
    .build(&[Field::GpuTemp])
    .unwrap_err();
    assert_eq!(e.code(), ErrorCode::PowerCapabilitiesIncomplete);
}

#[test]
fn an_energy_field_without_a_counter_spec_refuses_the_build() {
    let e = PowerCapabilities::builder(
        SourceId("x".to_string()),
        SourceKind::Hwmon,
        "v".to_string(),
    )
    .field(Field::Energy(Rail::Soc))
    .build(&[Field::Energy(Rail::Soc)])
    .unwrap_err();
    assert_eq!(e.code(), ErrorCode::PowerCapabilitiesIncomplete);
}

#[test]
fn a_sample_with_an_unavailable_field_is_refused_and_a_declared_one_passes() {
    let c = gb10_counter();
    let mut s = PowerSample::at(MonoNs(5));
    s.energy.insert(Rail::GpuCore, EnergyCount(10));
    s.clocks.sm_mhz = Some(2400);
    assert!(c.check(&s).is_ok());
    s.rails.insert(
        Rail::GpuCore,
        RailReading {
            instant_w: Some(10.0),
            ..RailReading::default()
        },
    );
    let e = c.check(&s).unwrap_err();
    assert_eq!(e.code(), ErrorCode::PowerFieldUndeclared);
}

#[test]
fn the_preference_picks_the_first_present_and_refuses_when_none_is() {
    let p = HeadlinePreference {
        platform: "gb10",
        scope: Scope::GpuRail,
        order: vec![
            SourceKind::GpuDriverCounter,
            SourceKind::GpuDriverSensor,
            SourceKind::NvidiaSmi,
        ],
    };
    assert_eq!(
        p.resolve(&[SourceKind::NvidiaSmi, SourceKind::GpuDriverSensor])
            .unwrap(),
        (1, SourceKind::GpuDriverSensor)
    );
    let e = p.resolve(&[SourceKind::External]).unwrap_err();
    assert_eq!(e.code(), ErrorCode::EnergyNoPreferredSource);
}

#[test]
fn spellings_round_trip() {
    for k in SourceKind::ALL {
        assert_eq!(SourceKind::parse(k.as_str()), Some(k));
    }
    for r in Rail::ALL {
        assert_eq!(Rail::parse(r.as_str()), Some(r));
    }
    for f in Field::universe() {
        assert_eq!(Field::parse(&f.name()), Some(f));
    }
    assert_eq!(Field::parse("energy(nowhere)"), None);
}
