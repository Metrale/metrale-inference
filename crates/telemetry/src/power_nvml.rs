// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The two NVML power instruments as [`PowerProbe`]s (DESIGN-kernel-energy.md §2):
//!
//! - [`NvmlCounterProbe`]: `nvmlDeviceGetTotalEnergyConsumption` (1 mJ; GB10 updates it every
//!   100 ms), with the SM clock, the memory clock where supported, the GPU temperature and the
//!   clocks-event reasons beside it (the fingerprint the attribution's rules read);
//! - [`NvmlSensorProbe`]: `nvmlDeviceGetPowerUsage` (mW; GB10 updates it every 250 ms).
//!
//! Each is opened once, and what the driver answers at open decides its capabilities (ROTI,
//! PCND): a query the driver does not support is declared unavailable with that reason, never
//! read as 0. A host adapter (`metrale-io-host-power::ThreadedMeter`) turns a probe into an
//! issue/complete `PowerMeter`; the binary chooses it (NDIC).
//!
//! Owner: telemetry.
//! Invariants: the clocks-event mask is reported with its GPU-idle bit (0x1) cleared, so a
//! non-zero throttle means held-down clocks; a field declared available that stops answering is a
//! `PowerReadFailed`, not an absent value.

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{
    CounterSpec, EnergyCount, Field, PowerCapabilities, PowerProbe, PowerSample, Rail, RailCaps,
    RailReading, SourceId, SourceKind,
};
use metrale_error::{ErrorCode, error};
use metrale_gpu_sys::nvml::{NVML_CLOCK_MEM, NVML_CLOCK_SM, Nvml, NvmlDevice, NvmlFns};

/// 2026-10-22: GB10's counter cadence (probe 2026-10-22: median 99.98 ms).
pub const GB10_COUNTER_PERIOD_MS: f64 = 100.0;
/// 2026-10-22: GB10's sensor cadence (probe 2026-10-22: median 250 ms).
pub const GB10_SENSOR_PERIOD_MS: f64 = 250.0;
/// 2026-10-22: The clocks-event bit that means "idle", not "throttled".
const GPU_IDLE: u64 = 0x1;

/// 2026-10-22: Where and how to open: the device ordinal, the rail the driver's counter and
/// sensor cover on this platform (GB10: `GpuCore`; SXM boards: `Board`), and their update periods
/// as measured for this platform (explicit: the driver does not report them).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NvmlPowerConfig {
    pub index: u32,
    pub rail: Rail,
    pub counter_period_ms: f64,
    pub sensor_period_ms: f64,
}

fn unavailable(what: &str, why: String) -> metrale_error::Error {
    error!(ErrorCode::PowerSourceUnavailable, what.to_string(), why)
}

fn open(fns: Option<NvmlFns>, index: u32) -> metrale_abstract_io::Result<Nvml> {
    let nvml = match fns {
        Some(f) => Nvml::with_fns(f),
        None => Nvml::open(),
    }
    .map_err(|e| unavailable("nvml", e.to_string()))?;
    nvml.device(index)
        .map_err(|e| unavailable("nvml", e.to_string()))?;
    Ok(nvml)
}

fn version(nvml_index: u32) -> String {
    format!("nvml device {nvml_index}")
}

fn failed(source: &str, what: &str, rc: i32) -> metrale_error::Error {
    error!(
        ErrorCode::PowerReadFailed,
        source.to_string(),
        format!("{what} returned {rc}")
    )
}

fn stopped(source: &str, what: &str) -> metrale_error::Error {
    error!(
        ErrorCode::PowerReadFailed,
        source.to_string(),
        format!("{what} stopped answering")
    )
}

/// 2026-10-22: The NVML energy counter and its fingerprint fields.
pub struct NvmlCounterProbe {
    nvml: Nvml,
    cfg: NvmlPowerConfig,
    caps: PowerCapabilities,
}

impl NvmlCounterProbe {
    /// 2026-10-22: Open over the system library.
    pub fn open(cfg: NvmlPowerConfig) -> metrale_abstract_io::Result<Self> {
        Self::build(None, cfg)
    }

    /// 2026-10-22: Open over an explicit function table (tests, or a binary-linked driver).
    pub fn with_fns(fns: NvmlFns, cfg: NvmlPowerConfig) -> metrale_abstract_io::Result<Self> {
        Self::build(Some(fns), cfg)
    }

    fn build(fns: Option<NvmlFns>, cfg: NvmlPowerConfig) -> metrale_abstract_io::Result<Self> {
        let nvml = open(fns, cfg.index)?;
        let caps = Self::declare(&nvml, cfg)?;
        Ok(Self { nvml, cfg, caps })
    }

    fn declare(
        nvml: &Nvml,
        cfg: NvmlPowerConfig,
    ) -> metrale_abstract_io::Result<PowerCapabilities> {
        let dev = nvml
            .device(cfg.index)
            .map_err(|e| unavailable("nvml", e.to_string()))?;
        let id = SourceId(format!("nvml{}.counter", cfg.index));
        if !matches!(dev.total_energy_mj(), Ok(Some(_))) {
            return Err(unavailable(
                &id.0,
                "nvmlDeviceGetTotalEnergyConsumption is not supported".to_string(),
            ));
        }
        let mut b =
            PowerCapabilities::builder(id, SourceKind::GpuDriverCounter, version(cfg.index))
                .counter(
                    cfg.rail,
                    CounterSpec {
                        unit_j: 0.001,
                        update_period_ms: cfg.counter_period_ms,
                    },
                );
        let probes = [
            (
                Field::SmClock,
                dev.clock_mhz(NVML_CLOCK_SM).map(|v| v.is_some()),
            ),
            (
                Field::MemClock,
                dev.clock_mhz(NVML_CLOCK_MEM)
                    .map(|v| v.is_some_and(|m| m > 0)),
            ),
            (Field::GpuTemp, dev.temperature_c().map(|v| v.is_some())),
            (
                Field::Throttle,
                dev.clocks_event_reasons().map(|v| v.is_some()),
            ),
        ];
        for (f, ok) in probes {
            b = match ok {
                Ok(true) => b.field(f),
                Ok(false) => b.unavailable(f, "the driver answers not supported (or 0) for it"),
                Err(rc) => b.unavailable(f, &format!("the driver query failed at open ({rc})")),
            };
        }
        b.rest_unavailable(
            &Field::universe(),
            "not reported by the NVML energy counter probe (the GPU-rail counter is NVML's only energy field)",
        )
        .build(&Field::universe())
    }

    fn dev(&self) -> metrale_abstract_io::Result<NvmlDevice<'_>> {
        self.nvml.device(self.cfg.index).map_err(|e| {
            error!(
                ErrorCode::PowerReadFailed,
                self.caps.source().0.clone(),
                e.to_string()
            )
        })
    }
}

impl PowerProbe for NvmlCounterProbe {
    fn capabilities(&self) -> &PowerCapabilities {
        &self.caps
    }

    fn read(&mut self, at: MonoNs) -> metrale_abstract_io::Result<PowerSample> {
        let src = self.caps.source().0.clone();
        let dev = self.dev()?;
        let mut s = PowerSample::at(at);
        let e = dev
            .total_energy_mj()
            .map_err(|rc| failed(&src, "the energy counter", rc))?
            .ok_or_else(|| stopped(&src, "the energy counter"))?;
        s.energy.insert(self.cfg.rail, EnergyCount(e));
        let has = |f| self.caps.available().contains(&f);
        if has(Field::SmClock) {
            s.clocks.sm_mhz = dev
                .clock_mhz(NVML_CLOCK_SM)
                .map_err(|rc| failed(&src, "the SM clock", rc))?;
        }
        if has(Field::MemClock) {
            s.clocks.mem_mhz = dev
                .clock_mhz(NVML_CLOCK_MEM)
                .map_err(|rc| failed(&src, "the memory clock", rc))?;
        }
        if has(Field::GpuTemp) {
            s.temps.gpu_c = dev
                .temperature_c()
                .map_err(|rc| failed(&src, "the temperature", rc))?
                .map(f64::from);
        }
        if has(Field::Throttle) {
            s.throttle = dev
                .clocks_event_reasons()
                .map_err(|rc| failed(&src, "the clocks-event reasons", rc))?
                .map(|m| m & !GPU_IDLE);
        }
        Ok(s)
    }
}

/// 2026-10-22: The NVML power sensor.
pub struct NvmlSensorProbe {
    nvml: Nvml,
    cfg: NvmlPowerConfig,
    caps: PowerCapabilities,
}

impl NvmlSensorProbe {
    /// 2026-10-22: Open over the system library.
    pub fn open(cfg: NvmlPowerConfig) -> metrale_abstract_io::Result<Self> {
        Self::build(None, cfg)
    }

    /// 2026-10-22: Open over an explicit function table.
    pub fn with_fns(fns: NvmlFns, cfg: NvmlPowerConfig) -> metrale_abstract_io::Result<Self> {
        Self::build(Some(fns), cfg)
    }

    fn build(fns: Option<NvmlFns>, cfg: NvmlPowerConfig) -> metrale_abstract_io::Result<Self> {
        let nvml = open(fns, cfg.index)?;
        let id = SourceId(format!("nvml{}.sensor", cfg.index));
        let ok = matches!(
            nvml.device(cfg.index).map(|d| d.power_mw()),
            Ok(Ok(Some(_)))
        );
        if !ok {
            return Err(unavailable(
                &id.0,
                "nvmlDeviceGetPowerUsage is not supported".to_string(),
            ));
        }
        let caps = PowerCapabilities::builder(id, SourceKind::GpuDriverSensor, version(cfg.index))
            .sensor(
                cfg.rail,
                RailCaps {
                    update_period_ms: cfg.sensor_period_ms,
                    average_window_ms: None,
                },
            )
            .unavailable(
                Field::AveragePower(cfg.rail),
                "field POWER_AVERAGE (187) is not bound here (GB10 does not support it)",
            )
            .rest_unavailable(
                &Field::universe(),
                "not reported by the NVML power sensor probe",
            )
            .build(&Field::universe())?;
        Ok(Self { nvml, cfg, caps })
    }
}

impl PowerProbe for NvmlSensorProbe {
    fn capabilities(&self) -> &PowerCapabilities {
        &self.caps
    }

    fn read(&mut self, at: MonoNs) -> metrale_abstract_io::Result<PowerSample> {
        let src = self.caps.source().0.clone();
        let dev = self
            .nvml
            .device(self.cfg.index)
            .map_err(|e| error!(ErrorCode::PowerReadFailed, src.clone(), e.to_string()))?;
        let mw = dev
            .power_mw()
            .map_err(|rc| failed(&src, "the power sensor", rc))?
            .ok_or_else(|| stopped(&src, "the power sensor"))?;
        let mut s = PowerSample::at(at);
        s.rails.insert(
            self.cfg.rail,
            RailReading {
                instant_w: Some(f64::from(mw) / 1000.0),
                ..RailReading::default()
            },
        );
        Ok(s)
    }
}

#[cfg(test)]
#[path = "power_nvml_tests.rs"]
mod tests;
