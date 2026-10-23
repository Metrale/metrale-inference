// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: A fixed-layout binary report (a USB-HID inline meter read through hidraw): each
//! quantity at an explicit byte offset, width, byte order and signedness, scaled to SI. The
//! report length is explicit and a frame of any other length is refused.
//!
//! Owner: metrale-power (external).
//! Invariants: a field never reads past the declared report length (checked at construction).

use alloc::format;

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{EnergyCount, PowerCapabilities, PowerSample, RailReading};
use metrale_error::{ErrorCode, error};

use super::{
    MeterPlacement, MeterProtocol, Result, Scale, energy_raw, meter_caps, power_w, protocol_error,
};

/// 2026-10-22: One integer in the report.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HidField {
    pub offset: usize,
    /// 2026-10-22: 1, 2, 4 or 8 bytes.
    pub width: usize,
    pub little_endian: bool,
    pub signed: bool,
    pub scale: Scale,
}

impl HidField {
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_wrap)]
    fn read(&self, b: &[u8]) -> f64 {
        let bytes = &b[self.offset..self.offset + self.width];
        let mut v: u64 = 0;
        for i in 0..self.width {
            let byte = if self.little_endian {
                bytes[self.width - 1 - i]
            } else {
                bytes[i]
            };
            v = (v << 8) | u64::from(byte);
        }
        // 2026-10-22: Integer counts of a meter are far below 2^53; the f64 is exact for them.
        let raw = if self.signed && self.width < 8 {
            let shift = 64 - 8 * self.width;
            (((v << shift) as i64) >> shift) as f64
        } else if self.signed {
            v as i64 as f64
        } else {
            v as f64
        };
        raw * self.scale.to_si
    }
}

/// 2026-10-22: Power and energy at fixed places in a report of `report_len` bytes.
#[derive(Clone, Debug)]
pub struct HidReportProtocol {
    placement: MeterPlacement,
    report_len: usize,
    power: Option<HidField>,
    energy: Option<(HidField, f64)>,
    caps: PowerCapabilities,
}

impl HidReportProtocol {
    /// 2026-10-22: `energy` = (field scaled to J, counter unit J).
    pub fn new(
        placement: MeterPlacement,
        report_len: usize,
        power: Option<HidField>,
        energy: Option<(HidField, f64)>,
    ) -> Result<Self> {
        for f in power.iter().chain(energy.iter().map(|e| &e.0)) {
            if !matches!(f.width, 1 | 2 | 4 | 8) || f.offset + f.width > report_len {
                return Err(error!(
                    ErrorCode::ExternalMeterProtocol,
                    placement.source.0.clone(),
                    format!(
                        "field at offset {} width {} does not fit a {report_len}-byte report",
                        f.offset, f.width
                    )
                ));
            }
        }
        let caps = meter_caps(
            &placement,
            "hid-report",
            power.is_some(),
            energy.map(|e| e.1),
        )?;
        Ok(Self {
            placement,
            report_len,
            power,
            energy,
            caps,
        })
    }

    /// 2026-10-22: The declared report length, the read size of a hidraw transport.
    #[must_use]
    pub fn report_len(&self) -> usize {
        self.report_len
    }
}

impl MeterProtocol for HidReportProtocol {
    fn capabilities(&self) -> &PowerCapabilities {
        &self.caps
    }

    fn parse(&self, frame: &[u8], at: MonoNs) -> Result<PowerSample> {
        let id = &self.placement.source;
        if frame.len() != self.report_len {
            return Err(protocol_error(
                id,
                format!("report is {} bytes, not {}", frame.len(), self.report_len),
            ));
        }
        let mut s = PowerSample::at(at);
        if let Some(f) = &self.power {
            let w = power_w(id, f.read(frame))?;
            s.rails.insert(
                self.placement.rail,
                RailReading {
                    instant_w: Some(w),
                    ..RailReading::default()
                },
            );
        }
        if let Some((f, unit_j)) = &self.energy {
            let raw = energy_raw(id, f.read(frame), *unit_j)?;
            s.energy.insert(self.placement.rail, EnergyCount(raw));
        }
        Ok(s)
    }
}
