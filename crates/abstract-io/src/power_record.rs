// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The records a [`crate::power::PowerMeter`] produces: one [`PowerSample`] per read
//! and one [`EnergyReading`] per integrated window. Every field is optional because what an
//! instrument reports is a capability ([`crate::power::PowerCapabilities`]), not a guess.
//!
//! Owner: metrale-abstract-io.
//! Invariants: physical quantities are `f64` in J, W, ms, MHz and °C; a raw device counter is the
//! device's integer in the unit its [`crate::power::CounterSpec`] declares.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::MonoNs;
use crate::power_caps::{Field, SourceId, SourceKind};

/// 2026-10-22: A supply an instrument can see. GB10 exposes only [`Rail::GpuCore`]; an APU socket
/// (gfx1151) is [`Rail::Soc`] (CPU and GPU together, not separable); an inline USB-C PD meter sees
/// [`Rail::Module`]; a wall plug sees [`Rail::Wall`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Rail {
    GpuCore,
    GpuMemory,
    Board,
    Module,
    Soc,
    CpuPackage,
    Dram,
    Platform,
    Wall,
}

impl Rail {
    /// 2026-10-22: Every rail, in declaration order.
    pub const ALL: [Rail; 9] = [
        Rail::GpuCore,
        Rail::GpuMemory,
        Rail::Board,
        Rail::Module,
        Rail::Soc,
        Rail::CpuPackage,
        Rail::Dram,
        Rail::Platform,
        Rail::Wall,
    ];

    /// 2026-10-22: The stored spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GpuCore => "gpu_core",
            Self::GpuMemory => "gpu_memory",
            Self::Board => "board",
            Self::Module => "module",
            Self::Soc => "soc",
            Self::CpuPackage => "cpu_package",
            Self::Dram => "dram",
            Self::Platform => "platform",
            Self::Wall => "wall",
        }
    }

    /// 2026-10-22: Inverse of [`Rail::as_str`].
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|r| r.as_str() == s)
    }
}

/// 2026-10-22: A raw cumulative energy counter, in the unit its `CounterSpec` declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EnergyCount(pub u64);

/// 2026-10-22: One rail's power, W.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RailReading {
    /// 2026-10-22: The sensor's instantaneous (or most recent update's) power.
    pub instant_w: Option<f64>,
    /// 2026-10-22: The device's own moving average, over its declared window.
    pub average_w: Option<f64>,
    /// 2026-10-22: The enforced power limit.
    pub limit_w: Option<f64>,
}

/// 2026-10-22: Clocks at the read, MHz.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Clocks {
    pub sm_mhz: Option<u32>,
    pub mem_mhz: Option<u32>,
    pub gfx_mhz: Option<u32>,
}

/// 2026-10-22: Temperatures at the read, °C.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Temps {
    pub gpu_c: Option<f64>,
    pub mem_c: Option<f64>,
}

/// 2026-10-22: Utilization at the read, percent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Util {
    pub gpu_pct: Option<u8>,
    pub mem_pct: Option<u8>,
}

/// 2026-10-22: One read of one instrument.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PowerSample {
    /// 2026-10-22: The caller's monotonic clock at issue (the time base).
    pub at: MonoNs,
    /// 2026-10-22: The device's own timestamp of the value, µs, as data only.
    pub driver_ts_us: Option<u64>,
    pub rails: BTreeMap<Rail, RailReading>,
    pub energy: BTreeMap<Rail, EnergyCount>,
    pub clocks: Clocks,
    pub temps: Temps,
    /// 2026-10-22: The vendor's throttle (clocks-event) reason mask with its idle and "none" bits
    /// cleared by the meter, so any non-zero value means the clocks are being held down.
    pub throttle: Option<u64>,
    pub util: Util,
}

impl PowerSample {
    /// 2026-10-22: An empty sample at `at`.
    #[must_use]
    pub fn at(at: MonoNs) -> Self {
        Self {
            at,
            ..Self::default()
        }
    }

    /// 2026-10-22: The fields this sample carries a value for.
    #[must_use]
    pub fn fields(&self) -> Vec<Field> {
        let mut out = Vec::new();
        for (r, v) in &self.rails {
            if v.instant_w.is_some() {
                out.push(Field::InstantPower(*r));
            }
            if v.average_w.is_some() {
                out.push(Field::AveragePower(*r));
            }
            if v.limit_w.is_some() {
                out.push(Field::PowerLimit(*r));
            }
        }
        out.extend(self.energy.keys().map(|r| Field::Energy(*r)));
        let flags = [
            (self.driver_ts_us.is_some(), Field::DriverTimestamp),
            (self.clocks.sm_mhz.is_some(), Field::SmClock),
            (self.clocks.mem_mhz.is_some(), Field::MemClock),
            (self.clocks.gfx_mhz.is_some(), Field::GfxClock),
            (self.temps.gpu_c.is_some(), Field::GpuTemp),
            (self.temps.mem_c.is_some(), Field::MemTemp),
            (self.throttle.is_some(), Field::Throttle),
            (self.util.gpu_pct.is_some(), Field::UtilGpu),
            (self.util.mem_pct.is_some(), Field::UtilMem),
        ];
        out.extend(flags.into_iter().filter(|(p, _)| *p).map(|(_, f)| f));
        out
    }
}

/// 2026-10-22: One source's energy over one window `[from, to]`.
#[derive(Clone, Debug, PartialEq)]
pub struct EnergyReading {
    pub source: SourceId,
    pub kind: SourceKind,
    pub rail: Rail,
    pub from: MonoNs,
    pub to: MonoNs,
    /// 2026-10-22: Energy over the window, J.
    pub joules: f64,
    /// 2026-10-22: `joules` over the window's seconds, W.
    pub avg_w: f64,
    /// 2026-10-22: Samples inside the window.
    pub samples: u64,
    /// 2026-10-22: Distinct value updates inside the window (a counter that ticks every 100 ms
    /// read at 40 Hz has a quarter as many updates as samples).
    pub updates: u64,
}
