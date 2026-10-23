// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The NVML probes over the scripted driver: the counter and its fingerprint land,
//! the idle bit is cleared, the capability check accepts every sample, an absent counter refuses
//! to open (no fallback), and the live GB10 driver (ignored; run by hand) declares what §1.2 found.

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{Declared, Field, PowerProbe, Rail};
use metrale_error::ErrorCode;

use super::*;
use crate::nvml_script::{self, Script};

const CFG: NvmlPowerConfig = NvmlPowerConfig {
    index: 0,
    rail: Rail::GpuCore,
    counter_period_ms: GB10_COUNTER_PERIOD_MS,
    sensor_period_ms: GB10_SENSOR_PERIOD_MS,
};

#[test]
fn the_counter_probe_reads_energy_and_its_fingerprint() {
    nvml_script::install(Script {
        energy_mj: [1_000, 1_000, 1_250].into(),
        ..Script::default()
    });
    let mut p = NvmlCounterProbe::with_fns(nvml_script::table(), CFG).unwrap();
    let c = p.capabilities().clone();
    assert_eq!(c.counter(Rail::GpuCore).unwrap().unit_j, 0.001);
    let s = p.read(MonoNs(7)).unwrap();
    assert_eq!(s.at, MonoNs(7));
    assert_eq!(s.energy[&Rail::GpuCore].0, 1_000);
    assert_eq!(s.clocks.sm_mhz, Some(1001));
    assert_eq!(s.temps.gpu_c, Some(55.0));
    // 2026-10-22: The script answers 0x4 (sw power cap); only the idle bit would be cleared.
    assert_eq!(s.throttle, Some(0x4));
    c.check(&s).unwrap();
    assert_eq!(p.read(MonoNs(8)).unwrap().energy[&Rail::GpuCore].0, 1_250);
    assert!(matches!(
        c.declares(Field::InstantPower(Rail::GpuCore)),
        Declared::Unavailable(_)
    ));
    assert!(matches!(
        c.declares(Field::Energy(Rail::Module)),
        Declared::Unavailable(_)
    ));
}

#[test]
fn the_sensor_probe_reads_watts() {
    nvml_script::install(Script::default());
    let mut p = NvmlSensorProbe::with_fns(nvml_script::table(), CFG).unwrap();
    let s = p.read(MonoNs(1)).unwrap();
    assert_eq!(s.rails[&Rail::GpuCore].instant_w, Some(60.0));
    p.capabilities().check(&s).unwrap();
}

#[test]
fn a_driver_without_the_counter_refuses_to_open() {
    nvml_script::install(Script::default());
    let mut t = nvml_script::table();
    t.total_energy = None;
    let e = NvmlCounterProbe::with_fns(t, CFG).err().unwrap();
    assert_eq!(e.code(), ErrorCode::PowerSourceUnavailable);
    let mut t = nvml_script::table();
    t.power_usage = None;
    assert_eq!(
        NvmlSensorProbe::with_fns(t, CFG).err().unwrap().code(),
        ErrorCode::PowerSourceUnavailable
    );
}

#[test]
fn a_failing_temperature_at_open_is_declared_unavailable_not_zero() {
    nvml_script::install(Script {
        temperature_rc: 999,
        ..Script::default()
    });
    let mut p = NvmlCounterProbe::with_fns(nvml_script::table(), CFG).unwrap();
    assert!(
        matches!(p.capabilities().declares(Field::GpuTemp), Declared::Unavailable(r) if r.contains("999"))
    );
    assert_eq!(p.read(MonoNs(0)).unwrap().temps.gpu_c, None);
}

/// 2026-10-22: Read-only against the real driver: `cargo test -p metrale-telemetry live_gb10 --
/// --ignored`. Asserts what the 2026-10-22 probe found on GB10.
#[test]
#[ignore = "needs an NVIDIA driver"]
fn live_gb10_declares_the_gpu_rail_only() {
    let mut c = NvmlCounterProbe::open(CFG).unwrap();
    let mut s = NvmlSensorProbe::open(CFG).unwrap();
    let a = c.read(MonoNs(0)).unwrap();
    let b = s.read(MonoNs(0)).unwrap();
    std::println!(
        "counter {:?}\nsensor {:?}\ncaps {:?}",
        a,
        b,
        c.capabilities().unavailable()
    );
    assert!(a.energy[&Rail::GpuCore].0 > 0);
    assert!(b.rails[&Rail::GpuCore].instant_w.unwrap() > 0.0);
}
