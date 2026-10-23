// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: What one power instrument reports: [`PowerCapabilities`], built once when a meter
//! is opened (ROTI) and immutable after. Every field of the universe a meter is built against is
//! either available or declared unavailable with a reason (PCND); a sample carrying anything else
//! is refused ([`PowerCapabilities::check`]).
//!
//! Owner: metrale-abstract-io.
//! Invariants:
//! - A field is never both available and unavailable.
//! - An available `Energy(rail)` has a [`CounterSpec`]; an available `InstantPower(rail)` has a
//!   [`RailCaps`] update period; and the converse.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use metrale_error::{ErrorCode, error};

use crate::power_record::{PowerSample, Rail};

/// 2026-10-22: Which kind of instrument a meter is. A replayed trace keeps the kind it was
/// recorded with, so a recompute is byte-identical (HPAR).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SourceKind {
    /// 2026-10-22: `nvmlDeviceGetTotalEnergyConsumption`: an integral, no sampling lag.
    NvmlCounter,
    /// 2026-10-22: `nvmlDeviceGetPowerUsage`: a sensor value, integrated by the caller.
    NvmlSensor,
    /// 2026-10-22: An `nvidia-smi -lms` child's `power.draw.average` lines.
    NvidiaSmi,
    /// 2026-10-22: Read-only Linux hwmon `power*_input`/`power*_average`/`energy*_input`.
    Hwmon,
    /// 2026-10-22: AMD SMI (`amdsmi_get_energy_count`, socket power).
    AmdSmi,
    /// 2026-10-22: Apple IOReport "Energy Model" channels.
    AppleIoReport,
    /// 2026-10-22: An external meter (inline USB-C PD meter or smart wall plug).
    External,
}

impl SourceKind {
    /// 2026-10-22: Every kind, in declaration order.
    pub const ALL: [SourceKind; 7] = [
        SourceKind::NvmlCounter,
        SourceKind::NvmlSensor,
        SourceKind::NvidiaSmi,
        SourceKind::Hwmon,
        SourceKind::AmdSmi,
        SourceKind::AppleIoReport,
        SourceKind::External,
    ];

    /// 2026-10-22: The stored spelling (`power_source` on a measurement row).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NvmlCounter => "nvml_gpu_rail_counter",
            Self::NvmlSensor => "nvml_gpu_rail_sensor",
            Self::NvidiaSmi => "nvidia_smi_power_draw_average",
            Self::Hwmon => "hwmon",
            Self::AmdSmi => "amdsmi",
            Self::AppleIoReport => "apple_ioreport",
            Self::External => "external_meter",
        }
    }

    /// 2026-10-22: Inverse of [`SourceKind::as_str`].
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.as_str() == s)
    }
}

/// 2026-10-22: One meter instance's id, unique within a run (e.g. `nvml0.counter`).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceId(pub String);

/// 2026-10-22: A quantity a meter may report.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Field {
    Energy(Rail),
    InstantPower(Rail),
    AveragePower(Rail),
    PowerLimit(Rail),
    SmClock,
    MemClock,
    GfxClock,
    GpuTemp,
    MemTemp,
    Throttle,
    UtilGpu,
    UtilMem,
    DriverTimestamp,
}

impl Field {
    /// 2026-10-22: The stored spelling, e.g. `energy(gpu_core)`, `sm_clock`.
    #[must_use]
    pub fn name(self) -> String {
        let rail = |what: &str, r: Rail| format!("{what}({})", r.as_str());
        match self {
            Self::Energy(r) => rail("energy", r),
            Self::InstantPower(r) => rail("instant_power", r),
            Self::AveragePower(r) => rail("average_power", r),
            Self::PowerLimit(r) => rail("power_limit", r),
            Self::SmClock => "sm_clock".to_string(),
            Self::MemClock => "mem_clock".to_string(),
            Self::GfxClock => "gfx_clock".to_string(),
            Self::GpuTemp => "gpu_temp".to_string(),
            Self::MemTemp => "mem_temp".to_string(),
            Self::Throttle => "throttle".to_string(),
            Self::UtilGpu => "util_gpu".to_string(),
            Self::UtilMem => "util_mem".to_string(),
            Self::DriverTimestamp => "driver_timestamp".to_string(),
        }
    }

    /// 2026-10-22: Every field a meter on any platform could report: each rail's four power
    /// fields, then the scalars. A meter is built against this universe unless it states a
    /// narrower one.
    #[must_use]
    pub fn universe() -> Vec<Field> {
        let mut out = Vec::new();
        for r in Rail::ALL {
            out.extend([
                Field::Energy(r),
                Field::InstantPower(r),
                Field::AveragePower(r),
                Field::PowerLimit(r),
            ]);
        }
        out.extend([
            Field::SmClock,
            Field::MemClock,
            Field::GfxClock,
            Field::GpuTemp,
            Field::MemTemp,
            Field::Throttle,
            Field::UtilGpu,
            Field::UtilMem,
            Field::DriverTimestamp,
        ]);
        out
    }
}

/// 2026-10-22: A cumulative counter's unit and cadence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CounterSpec {
    /// 2026-10-22: Joules per raw count (NVML: 0.001; amdsmi: 15.3e-6).
    pub unit_j: f64,
    /// 2026-10-22: How often the device updates the value, ms (NVML on GB10: 100).
    pub update_period_ms: f64,
}

/// 2026-10-22: A rail's power sensor cadence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RailCaps {
    /// 2026-10-22: How often the device updates the instant value, ms (NVML on GB10: 250).
    pub update_period_ms: f64,
    /// 2026-10-22: The averaging window of `AveragePower`, ms, when the device has one.
    pub average_window_ms: Option<f64>,
}

/// 2026-10-22: A field a meter cannot report, and why (stored with the trace).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unavailable {
    pub field: Field,
    pub reason: String,
}

/// 2026-10-22: What a capability set says about one field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Declared<'a> {
    Available,
    Unavailable(&'a str),
    Undeclared,
}

/// 2026-10-22: One meter's declared capabilities; build with [`PowerCapabilities::builder`].
#[derive(Clone, Debug, PartialEq)]
pub struct PowerCapabilities {
    source: SourceId,
    kind: SourceKind,
    version: String,
    counters: BTreeMap<Rail, CounterSpec>,
    sensors: BTreeMap<Rail, RailCaps>,
    fields: BTreeSet<Field>,
    unavailable: Vec<Unavailable>,
}

/// 2026-10-22: Collects a declaration; [`CapsBuilder::build`] checks it whole.
#[derive(Clone, Debug)]
pub struct CapsBuilder(PowerCapabilities);

impl PowerCapabilities {
    /// 2026-10-22: Start a declaration for meter `source` of `kind`, `version` naming its driver,
    /// firmware or protocol.
    #[must_use]
    pub fn builder(source: SourceId, kind: SourceKind, version: String) -> CapsBuilder {
        CapsBuilder(Self {
            source,
            kind,
            version,
            counters: BTreeMap::new(),
            sensors: BTreeMap::new(),
            fields: BTreeSet::new(),
            unavailable: Vec::new(),
        })
    }

    #[must_use]
    pub fn source(&self) -> &SourceId {
        &self.source
    }

    #[must_use]
    pub fn kind(&self) -> SourceKind {
        self.kind
    }

    #[must_use]
    pub fn version(&self) -> &str {
        &self.version
    }

    #[must_use]
    pub fn counter(&self, rail: Rail) -> Option<&CounterSpec> {
        self.counters.get(&rail)
    }

    #[must_use]
    pub fn sensor(&self, rail: Rail) -> Option<&RailCaps> {
        self.sensors.get(&rail)
    }

    #[must_use]
    pub fn available(&self) -> &BTreeSet<Field> {
        &self.fields
    }

    #[must_use]
    pub fn unavailable(&self) -> &[Unavailable] {
        &self.unavailable
    }

    /// 2026-10-22: What this declaration says about `f`.
    #[must_use]
    pub fn declares(&self, f: Field) -> Declared<'_> {
        if self.fields.contains(&f) {
            return Declared::Available;
        }
        self.unavailable
            .iter()
            .find(|u| u.field == f)
            .map_or(Declared::Undeclared, |u| Declared::Unavailable(&u.reason))
    }

    /// 2026-10-22: `Ok` when `s` carries only available fields; `PowerFieldUndeclared` naming the
    /// first one it should not carry.
    pub fn check(&self, s: &PowerSample) -> crate::Result<()> {
        match s.fields().into_iter().find(|f| !self.fields.contains(f)) {
            None => Ok(()),
            Some(f) => Err(error!(
                ErrorCode::PowerFieldUndeclared,
                self.source.0.clone(),
                f.name()
            )),
        }
    }
}

impl CapsBuilder {
    /// 2026-10-22: An available cumulative counter on `rail`.
    #[must_use]
    pub fn counter(mut self, rail: Rail, spec: CounterSpec) -> Self {
        self.0.counters.insert(rail, spec);
        self.0.fields.insert(Field::Energy(rail));
        self
    }

    /// 2026-10-22: An available instant power sensor on `rail`.
    #[must_use]
    pub fn sensor(mut self, rail: Rail, caps: RailCaps) -> Self {
        self.0.sensors.insert(rail, caps);
        self.0.fields.insert(Field::InstantPower(rail));
        self
    }

    /// 2026-10-22: An available field that needs no cadence (clocks, temperature, ...).
    #[must_use]
    pub fn field(mut self, f: Field) -> Self {
        self.0.fields.insert(f);
        self
    }

    /// 2026-10-22: `f` is not reported here, because `reason`.
    #[must_use]
    pub fn unavailable(mut self, f: Field, reason: &str) -> Self {
        self.0.unavailable.push(Unavailable {
            field: f,
            reason: reason.to_string(),
        });
        self
    }

    /// 2026-10-22: Every field of `universe` neither available nor declared is declared
    /// unavailable with `reason`: the meter states once why the rest is absent.
    #[must_use]
    pub fn rest_unavailable(mut self, universe: &[Field], reason: &str) -> Self {
        for &f in universe {
            if !matches!(self.0.declares(f), Declared::Undeclared) {
                continue;
            }
            self.0.unavailable.push(Unavailable {
                field: f,
                reason: reason.to_string(),
            });
        }
        self
    }

    /// 2026-10-22: The declaration, when every field of `universe` is declared one way, no field
    /// is declared both ways or twice, and every cadence is positive.
    pub fn build(self, universe: &[Field]) -> crate::Result<PowerCapabilities> {
        let c = self.0;
        let id = || c.source.0.clone();
        let mut seen = BTreeSet::new();
        for u in &c.unavailable {
            if c.fields.contains(&u.field) || !seen.insert(u.field) || u.reason.trim().is_empty() {
                return Err(error!(
                    ErrorCode::PowerCapabilitiesIncomplete,
                    id(),
                    u.field.name()
                ));
            }
        }
        if let Some(f) = universe
            .iter()
            .find(|f| matches!(c.declares(**f), Declared::Undeclared))
        {
            return Err(error!(
                ErrorCode::PowerCapabilitiesIncomplete,
                id(),
                f.name()
            ));
        }
        let unspecified = c.fields.iter().find(|f| match f {
            Field::Energy(r) => !c.counters.contains_key(r),
            Field::InstantPower(r) => !c.sensors.contains_key(r),
            _ => false,
        });
        if let Some(f) = unspecified {
            return Err(error!(
                ErrorCode::PowerCapabilitiesIncomplete,
                id(),
                f.name()
            ));
        }
        let bad_counter = c
            .counters
            .iter()
            .find(|(_, s)| !(s.unit_j > 0.0 && s.update_period_ms > 0.0));
        let bad_sensor = c.sensors.iter().find(|(_, s)| !(s.update_period_ms > 0.0));
        if let Some((r, _)) = bad_counter {
            return Err(error!(
                ErrorCode::PowerCapabilitiesIncomplete,
                id(),
                Field::Energy(*r).name()
            ));
        }
        if let Some((r, _)) = bad_sensor {
            return Err(error!(
                ErrorCode::PowerCapabilitiesIncomplete,
                id(),
                Field::InstantPower(*r).name()
            ));
        }
        Ok(c)
    }
}
