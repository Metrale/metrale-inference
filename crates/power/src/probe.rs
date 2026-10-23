// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Import of the 2026-10-22 feasibility probe's raw trace
//! (`energy-probe-2026-10-22/kenergy_trace.csv`: `t_s,energy_mj,power_mw,sm_mhz,mem_mhz,temp_c,
//! reasons,util`, one NVML read of every field per 25 ms) into a [`PowerTrace`] of two sources:
//! the GPU-rail energy counter (with the clocks, temperature, throttle and utilization the probe
//! read beside it) and the GPU-rail power sensor. The probe did not record its windows or its
//! launch counts; the caller states the windows, and their load windows carry no launch count.
//!
//! Owner: metrale-power.
//! Invariants: times are parsed from the decimal text exactly (no float rounding); the idle bit
//! (0x1) of the clocks-event mask is cleared, so a non-zero throttle means held-down clocks.

use alloc::string::ToString;

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{
    CounterSpec, EnergyCount, Field, PowerCapabilities, PowerSample, Rail, RailCaps, RailReading,
    SourceId, SourceKind,
};

use crate::Result;
use crate::trace::{PowerTrace, bad, num};

/// 2026-10-22: The column header the probe wrote.
pub const PROBE_HEADER: &str = "t_s,energy_mj,power_mw,sm_mhz,mem_mhz,temp_c,reasons,util";

const DRIVER: &str = "580.159.03";

fn counter_caps() -> Result<PowerCapabilities> {
    PowerCapabilities::builder(
        SourceId("nvml0.counter".to_string()),
        SourceKind::GpuDriverCounter,
        DRIVER.to_string(),
    )
    .counter(
        Rail::GpuCore,
        CounterSpec {
            unit_j: 0.001,
            update_period_ms: 100.0,
        },
    )
    .field(Field::SmClock)
    .field(Field::GpuTemp)
    .field(Field::Throttle)
    .field(Field::UtilGpu)
    .unavailable(
        Field::MemClock,
        "GB10 NVML: the memory clock query is not supported (the probe read 0)",
    )
    .build(&[
        Field::Energy(Rail::GpuCore),
        Field::SmClock,
        Field::GpuTemp,
        Field::Throttle,
        Field::UtilGpu,
        Field::MemClock,
    ])
}

fn sensor_caps() -> Result<PowerCapabilities> {
    PowerCapabilities::builder(
        SourceId("nvml0.sensor".to_string()),
        SourceKind::GpuDriverSensor,
        DRIVER.to_string(),
    )
    .sensor(
        Rail::GpuCore,
        RailCaps {
            update_period_ms: 250.0,
            average_window_ms: None,
        },
    )
    .unavailable(
        Field::AveragePower(Rail::GpuCore),
        "GB10 NVML: field POWER_AVERAGE (187) is not supported",
    )
    .build(&[
        Field::InstantPower(Rail::GpuCore),
        Field::AveragePower(Rail::GpuCore),
    ])
}

/// 2026-10-22: Decimal seconds to integer nanoseconds, exactly.
fn ns(s: &str, line: usize) -> Result<u64> {
    let (int, frac) = s.split_once('.').unwrap_or((s, ""));
    if frac.len() > 9 {
        return Err(bad(line, "more than nine fractional digits"));
    }
    let mut f: u64 = if frac.is_empty() { 0 } else { num(frac, line)? };
    for _ in frac.len()..9 {
        f *= 10;
    }
    let i: u64 = num(int, line)?;
    Ok(i * 1_000_000_000 + f)
}

/// 2026-10-22: The probe's CSV as a trace with no windows and the probe's facts as metadata.
pub fn import(csv: &str) -> Result<PowerTrace> {
    let mut lines = csv.lines().enumerate();
    match lines.next() {
        Some((_, PROBE_HEADER)) => {}
        _ => return Err(bad(1, "not the 2026-10-22 probe header")),
    }
    let mut t = PowerTrace::default();
    let c = t.add_source(counter_caps()?);
    let s = t.add_source(sensor_caps()?);
    for (i, line) in lines {
        let n = i + 1;
        let f: alloc::vec::Vec<&str> = line.split(',').collect();
        let [ts, e, p, sm, _mem, temp, reasons, util] = f.as_slice() else {
            return Err(bad(n, "expected eight columns"));
        };
        let at = MonoNs(ns(ts, n)?);
        let mut x = PowerSample::at(at);
        x.energy.insert(Rail::GpuCore, EnergyCount(num(e, n)?));
        x.clocks.sm_mhz = Some(num(sm, n)?);
        x.temps.gpu_c = Some(f64::from(num::<u32>(temp, n)?));
        x.throttle = Some(num::<u64>(reasons, n)? & !1);
        x.util.gpu_pct = Some(num(util, n)?);
        t.push(c, x)?;
        let mut y = PowerSample::at(at);
        let mw: u32 = num(p, n)?;
        y.rails.insert(
            Rail::GpuCore,
            RailReading {
                instant_w: Some(f64::from(mw) / 1000.0),
                ..RailReading::default()
            },
        );
        t.push(s, y)?;
    }
    Ok(t)
}
