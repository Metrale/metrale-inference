// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: A `key=value[unit]` line meter (the ASCII output of USB-C PD inline meters and
//! serial logging firmwares, e.g. `V=20.01V I=0.85A P=17.02W E=1234.5J`). Tokens are separated
//! by whitespace, `,` or `;`; the key/value separator, the keys, the unit suffix each value must
//! carry and its scale to SI are configured.
//!
//! Owner: metrale-power (external).
//! Invariants: a configured key that is absent, repeated, or carries another unit is refused.

use alloc::format;
use alloc::string::String;

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{EnergyCount, PowerCapabilities, PowerSample, RailReading};

use super::{
    MeterPlacement, MeterProtocol, Result, Scale, energy_raw, meter_caps, power_w, protocol_error,
};

/// 2026-10-22: One configured key: its name, the unit suffix its value carries (may be empty)
/// and the scale of the number to W or J.
#[derive(Clone, Debug, PartialEq)]
pub struct KvKey {
    pub key: String,
    pub unit: String,
    pub scale: Scale,
}

/// 2026-10-22: Power and energy from one `key=value` line.
#[derive(Clone, Debug)]
pub struct KeyValueLineProtocol {
    placement: MeterPlacement,
    separator: char,
    power: Option<KvKey>,
    energy: Option<(KvKey, f64)>,
    caps: PowerCapabilities,
}

impl KeyValueLineProtocol {
    /// 2026-10-22: `energy` = (key, counter unit J).
    pub fn new(
        placement: MeterPlacement,
        separator: char,
        power: Option<KvKey>,
        energy: Option<(KvKey, f64)>,
    ) -> Result<Self> {
        let caps = meter_caps(
            &placement,
            "key-value-line",
            power.is_some(),
            energy.as_ref().map(|e| e.1),
        )?;
        Ok(Self {
            placement,
            separator,
            power,
            energy,
            caps,
        })
    }

    fn value(&self, line: &str, k: &KvKey) -> Result<f64> {
        let id = &self.placement.source;
        let mut found = None;
        for tok in line.split(|c: char| c.is_whitespace() || c == ',' || c == ';') {
            let Some((key, val)) = tok.split_once(self.separator) else {
                continue;
            };
            if key != k.key {
                continue;
            }
            if found.is_some() {
                return Err(protocol_error(id, format!("key {} appears twice", k.key)));
            }
            let num = val.strip_suffix(k.unit.as_str()).ok_or_else(|| {
                protocol_error(
                    id,
                    format!("{} = {val:?} lacks the unit {:?}", k.key, k.unit),
                )
            })?;
            let v: f64 = num
                .parse()
                .map_err(|_| protocol_error(id, format!("{} = {val:?} is not a number", k.key)))?;
            found = Some(v * k.scale.to_si);
        }
        found.ok_or_else(|| protocol_error(id, format!("key {} is absent", k.key)))
    }
}

impl MeterProtocol for KeyValueLineProtocol {
    fn capabilities(&self) -> &PowerCapabilities {
        &self.caps
    }

    fn parse(&self, frame: &[u8], at: MonoNs) -> Result<PowerSample> {
        let id = &self.placement.source;
        let line = core::str::from_utf8(frame)
            .map_err(|_| protocol_error(id, String::from("frame is not UTF-8")))?
            .trim();
        let mut s = PowerSample::at(at);
        if let Some(k) = &self.power {
            let w = power_w(id, self.value(line, k)?)?;
            s.rails.insert(
                self.placement.rail,
                RailReading {
                    instant_w: Some(w),
                    ..RailReading::default()
                },
            );
        }
        if let Some((k, unit_j)) = &self.energy {
            let raw = energy_raw(id, self.value(line, k)?, *unit_j)?;
            s.energy.insert(self.placement.rail, EnergyCount(raw));
        }
        Ok(s)
    }
}
