// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The power port: one [`PowerMeter`] per instrument (the NVML energy counter, the
//! NVML power sensor, an `nvidia-smi` child, read-only hwmon, amdsmi, Apple IOReport, an external
//! inline or wall meter, a replayed trace), each read the same way: [`PowerMeter::issue`] starts a
//! read and returns at once, [`PowerMeter::complete`] polls it and never blocks (FILL, ACNB). A
//! run samples every meter it has concurrently; no instrument is privileged in the data model,
//! and which one heads a display is the platform's declared [`HeadlinePreference`] (PCND).
//!
//! The records are in `power_record.rs` and the capability declaration in `power_caps.rs`.
//! Physical quantities are `f64` (J, W, ms): BU64 governs byte sizes and offsets in wire or
//! stored data, not measured physics. A raw device counter stays the integer the device returns
//! ([`EnergyCount`]), so a wrap or a reset is seen exactly.
//!
//! Owner: metrale-abstract-io.
//! Invariants:
//! - [`PowerSample::at`] is the caller's [`crate::Clock`] reading at issue (or the recorded one
//!   on replay), never a device timestamp; a device's own stamp is data
//!   ([`PowerSample::driver_ts_us`]).
//! - A sample carries only fields its meter's [`PowerCapabilities`] declares available
//!   ([`PowerCapabilities::check`]).

use core::task::Poll;

use alloc::vec::Vec;

pub use crate::power_caps::{
    CounterSpec, Declared, Field, PowerCapabilities, RailCaps, SourceId, SourceKind, Unavailable,
};
pub use crate::power_record::{
    Clocks, EnergyCount, EnergyReading, PowerSample, Rail, RailReading, Temps, Util,
};

/// 2026-10-22: One issued read: the meter's own sequence number. A ticket completes once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SampleTicket(pub u64);

/// 2026-10-22: An instrument read through issue and complete.
pub trait PowerMeter {
    /// 2026-10-22: What this meter reports, resolved once when it was opened (ROTI).
    fn capabilities(&self) -> &PowerCapabilities;
    /// 2026-10-22: Start one read stamped `at` (the caller's clock) and return at once. A meter
    /// with a bounded queue refuses with `PowerQueueFull` rather than wait.
    fn issue(&mut self, at: crate::MonoNs) -> crate::Result<SampleTicket>;
    /// 2026-10-22: The read `ticket` started, if it has finished; `Pending` otherwise. Never
    /// blocks. A ticket this meter did not issue, or one already completed, is `PowerTicketUnknown`.
    fn complete(&mut self, ticket: SampleTicket) -> Poll<crate::Result<PowerSample>>;
}

/// 2026-10-22: One instrument's blocking read, for a host to drive on its own thread and expose
/// as a [`PowerMeter`] (the issue/complete adapter is a host's, `metrale-io-host-power`). Device
/// and host crates implement it; the core never calls it (ACNB).
pub trait PowerProbe {
    /// 2026-10-22: What this probe reports, resolved once when it was opened (ROTI).
    fn capabilities(&self) -> &PowerCapabilities;
    /// 2026-10-22: Read once, now, stamping the sample `at`. May block for the instrument's own
    /// latency (a driver call, a line from a child, a network frame).
    fn read(&mut self, at: crate::MonoNs) -> crate::Result<PowerSample>;
}

/// 2026-10-22: What a headline number covers: the GPU rail alone, or the whole system (module,
/// SoC, or wall).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Scope {
    /// 2026-10-22: The GPU rail (GB10's only internal rail).
    GpuRail,
    /// 2026-10-22: Everything behind one supply: an inline USB-C PD meter or a wall plug.
    WholeSystem,
}

impl Scope {
    /// 2026-10-22: The stored spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GpuRail => "gpu_rail",
            Self::WholeSystem => "whole_system",
        }
    }
}

/// 2026-10-22: A platform's declared order of sources for one scope's headline. It chooses only
/// which reading is displayed first; every present source is still sampled and stored.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadlinePreference {
    /// 2026-10-22: The platform this order is declared for, e.g. `gb10`.
    pub platform: &'static str,
    /// 2026-10-22: What the headline covers.
    pub scope: Scope,
    /// 2026-10-22: Most preferred first.
    pub order: Vec<SourceKind>,
}

impl HeadlinePreference {
    /// 2026-10-22: The first kind in the declared order that `present` contains, with its rank;
    /// `EnergyNoPreferredSource` when none is present (a declared, reported mode, never a guess).
    pub fn resolve(&self, present: &[SourceKind]) -> crate::Result<(usize, SourceKind)> {
        self.order
            .iter()
            .enumerate()
            .find(|(_, k)| present.contains(k))
            .map(|(i, k)| (i, *k))
            .ok_or_else(|| {
                let order: Vec<&str> = self.order.iter().map(|k| k.as_str()).collect();
                metrale_error::error!(
                    metrale_error::ErrorCode::EnergyNoPreferredSource,
                    self.scope.as_str(),
                    order.join(" > ")
                )
            })
    }
}

#[cfg(test)]
#[path = "power_tests.rs"]
mod tests;
