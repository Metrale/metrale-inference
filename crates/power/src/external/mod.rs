// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: External meters (an inline USB-C PD meter, a smart wall plug): the pure half. A
//! [`MeterProtocol`] turns one frame of bytes (a JSON document, a `key=value` line, a fixed HID
//! report) into a [`PowerSample`] carrying the meter's power and cumulative energy on the one
//! rail the caller declares it sits on (`Rail::Module` behind a USB-C PD meter, `Rail::Wall`
//! behind a plug). The transports (serial, hidraw, HTTP, MQTT) are hosts
//! (`metrale-io-host-power`); a recorded meter replays through the trace replay meter.
//!
//! Every quantity is configured explicitly: the key or offset, the scale to SI and, for energy,
//! the counter unit (PCND). Energy is stored as the raw counter `round(J / unit_j)`, so wraps and
//! resets are seen exactly by the attribution's counter tracker. Voltage, current, clocks and
//! temperatures are declared unavailable: the port has no field for V/A, and a meter outside the
//! box sees no GPU state.
//!
//! Owner: metrale-power (external).
//! Invariants:
//! - A protocol's capabilities declare every field of [`Field::universe`].
//! - `parse` returns only the fields its capabilities declare available, or
//!   `ExternalMeterProtocol`.

use alloc::format;
use alloc::string::String;

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{
    CounterSpec, Field, PowerCapabilities, PowerSample, Rail, RailCaps, SourceId, SourceKind,
};
use metrale_error::{ErrorCode, error};

mod hid;
mod json;
mod kv;

pub use hid::{HidField, HidReportProtocol};
pub use json::{JsonPathProtocol, json_number_at};
pub use kv::{KeyValueLineProtocol, KvKey};

/// 2026-10-22: The result type of this module.
pub type Result<T> = core::result::Result<T, metrale_error::Error>;

/// 2026-10-22: One frame of a meter's wire format to a sample.
pub trait MeterProtocol {
    /// 2026-10-22: What this meter reports, declared once at construction.
    fn capabilities(&self) -> &PowerCapabilities;
    /// 2026-10-22: The sample one frame carries, stamped `at`.
    fn parse(&self, frame: &[u8], at: MonoNs) -> Result<PowerSample>;
}

/// 2026-10-22: Where the meter sits and how often it updates: the part of every protocol's
/// declaration that is not about its wire format.
#[derive(Clone, Debug, PartialEq)]
pub struct MeterPlacement {
    /// 2026-10-22: The meter's id within the run, e.g. `usbc-pd0`, `plug-desk`.
    pub source: SourceId,
    /// 2026-10-22: The device and firmware, e.g. `tasmota 14.2 (Athom plug)`.
    pub device: String,
    /// 2026-10-22: The rail the meter measures (Module for an inline PD meter, Wall for a plug).
    pub rail: Rail,
    /// 2026-10-22: How often the meter publishes a new value, ms.
    pub update_period_ms: f64,
}

/// 2026-10-22: A configured quantity: its value times `to_si` is W (power) or J (energy).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scale {
    pub to_si: f64,
}

/// 2026-10-22: The capability set of an external meter on `p.rail`: instant power when `power`,
/// a counter of `energy_unit_j` when `Some`, everything else unavailable with the reason.
pub(crate) fn meter_caps(
    p: &MeterPlacement,
    protocol: &str,
    power: bool,
    energy_unit_j: Option<f64>,
) -> Result<PowerCapabilities> {
    let name = || p.source.0.clone();
    if !power && energy_unit_j.is_none() {
        return Err(error!(
            ErrorCode::ExternalMeterProtocol,
            name(),
            "configures neither power nor energy"
        ));
    }
    let mut b = PowerCapabilities::builder(
        p.source.clone(),
        SourceKind::External,
        format!("{protocol}; {}", p.device),
    );
    if power {
        b = b.sensor(
            p.rail,
            RailCaps {
                update_period_ms: p.update_period_ms,
                average_window_ms: None,
            },
        );
    } else {
        b = b.unavailable(
            Field::InstantPower(p.rail),
            "the meter's protocol is not configured with a power value",
        );
    }
    match energy_unit_j {
        Some(unit_j) => {
            b = b.counter(
                p.rail,
                CounterSpec {
                    unit_j,
                    update_period_ms: p.update_period_ms,
                },
            );
        }
        None => {
            b = b.unavailable(
                Field::Energy(p.rail),
                "the meter's protocol is not configured with an energy total",
            );
        }
    }
    b.rest_unavailable(
        &Field::universe(),
        "an external meter reports only power and energy on the rail it sits on; it sees no \
         GPU clocks, temperatures or other rails, and the port has no voltage/current field",
    )
    .build(&Field::universe())
}

/// 2026-10-22: `joules` as a raw count of `unit_j`; refuses a negative, non-finite or
/// out-of-range total.
pub(crate) fn energy_raw(meter: &SourceId, joules: f64, unit_j: f64) -> Result<u64> {
    let counts = libm::round(joules / unit_j);
    // 2026-10-22: 2^63 counts bounds the conversion below; no meter's total comes near it.
    if !counts.is_finite() || !(0.0..9.0e18).contains(&counts) {
        return Err(error!(
            ErrorCode::ExternalMeterProtocol,
            meter.0.clone(),
            format!("energy total {joules} J is not a non-negative counter value")
        ));
    }
    // 2026-10-22: counts is a finite integer in [0, 9e18), checked above.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok(counts as u64)
}

/// 2026-10-22: A power reading that must be finite and non-negative.
pub(crate) fn power_w(meter: &SourceId, w: f64) -> Result<f64> {
    if w.is_finite() && w >= 0.0 {
        Ok(w)
    } else {
        Err(error!(
            ErrorCode::ExternalMeterProtocol,
            meter.0.clone(),
            format!("power {w} W is not a non-negative finite value")
        ))
    }
}

/// 2026-10-22: A protocol error for `meter`.
pub(crate) fn protocol_error(meter: &SourceId, why: String) -> metrale_error::Error {
    error!(ErrorCode::ExternalMeterProtocol, meter.0.clone(), why)
}

#[cfg(test)]
#[path = "external_tests.rs"]
mod tests;
