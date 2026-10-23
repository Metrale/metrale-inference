// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: A JSON meter document (Tasmota `Status 8` / `tele/<t>/SENSOR`, Shelly Gen2
//! `Switch.GetStatus`): power and energy read at configured dotted paths. The extractor is a
//! small no_std scanner that walks objects by key and skips everything else; it reads numbers
//! only and refuses anything that is not well-formed on the path it walks.
//!
//! Owner: metrale-power (external).
//! Invariants: a path segment that is all digits indexes an array; any other names an object key.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{EnergyCount, PowerCapabilities, PowerSample, RailReading};

use super::{
    MeterPlacement, MeterProtocol, Result, Scale, energy_raw, meter_caps, power_w, protocol_error,
};

/// 2026-10-22: Power and energy at dotted paths of a JSON document.
#[derive(Clone, Debug)]
pub struct JsonPathProtocol {
    placement: MeterPlacement,
    power: Option<(Vec<String>, Scale)>,
    energy: Option<(Vec<String>, Scale, f64)>,
    caps: PowerCapabilities,
}

fn split(path: &str) -> Vec<String> {
    path.split('.').map(String::from).collect()
}

impl JsonPathProtocol {
    /// 2026-10-22: `power` = (dotted path, scale to W); `energy` = (dotted path, scale to J,
    /// counter unit J). Tasmota: `("StatusSNS.ENERGY.Power", 1.0)` and
    /// `("StatusSNS.ENERGY.Total", 3.6e6 /* kWh */, 3.6 /* 1 mWh */)`; Shelly Gen2: `("apower", 1.0)`
    /// and `("aenergy.total", 3600.0 /* Wh */, 3.6)`.
    pub fn new(
        placement: MeterPlacement,
        power: Option<(&str, Scale)>,
        energy: Option<(&str, Scale, f64)>,
    ) -> Result<Self> {
        let caps = meter_caps(
            &placement,
            "json-path",
            power.is_some(),
            energy.map(|e| e.2),
        )?;
        Ok(Self {
            power: power.map(|(p, s)| (split(p), s)),
            energy: energy.map(|(p, s, u)| (split(p), s, u)),
            placement,
            caps,
        })
    }
}

impl MeterProtocol for JsonPathProtocol {
    fn capabilities(&self) -> &PowerCapabilities {
        &self.caps
    }

    fn parse(&self, frame: &[u8], at: MonoNs) -> Result<PowerSample> {
        let id = &self.placement.source;
        let text = core::str::from_utf8(frame)
            .map_err(|_| protocol_error(id, String::from("frame is not UTF-8")))?;
        let mut s = PowerSample::at(at);
        let at_path = |path: &[String]| {
            let segs: Vec<&str> = path.iter().map(String::as_str).collect();
            json_number_at(text, &segs)
                .map_err(|why| protocol_error(id, format!("{}: {why}", path.join("."))))
        };
        if let Some((path, scale)) = &self.power {
            let w = power_w(id, at_path(path)? * scale.to_si)?;
            s.rails.insert(
                self.placement.rail,
                RailReading {
                    instant_w: Some(w),
                    ..RailReading::default()
                },
            );
        }
        if let Some((path, scale, unit_j)) = &self.energy {
            let raw = energy_raw(id, at_path(path)? * scale.to_si, *unit_j)?;
            s.energy.insert(self.placement.rail, EnergyCount(raw));
        }
        Ok(s)
    }
}

struct Scan<'a> {
    b: &'a [u8],
    i: usize,
}

type Why = &'static str;

impl Scan<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.b.get(self.i).copied()
    }

    fn eat(&mut self, c: u8) -> core::result::Result<(), Why> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err("malformed JSON")
        }
    }

    /// 2026-10-22: The byte range of a string between its quotes (escapes kept, never unescaped: keys
    /// are compared raw).
    fn string(&mut self) -> core::result::Result<(usize, usize), Why> {
        self.eat(b'"')?;
        let start = self.i;
        while self.i < self.b.len() {
            match self.b[self.i] {
                b'\\' => self.i += 2,
                b'"' => {
                    let end = self.i;
                    self.i += 1;
                    return Ok((start, end));
                }
                _ => self.i += 1,
            }
        }
        Err("unterminated string")
    }

    fn number(&mut self) -> core::result::Result<f64, Why> {
        self.ws();
        let start = self.i;
        while self.i < self.b.len()
            && matches!(
                self.b[self.i],
                b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E'
            )
        {
            self.i += 1;
        }
        core::str::from_utf8(&self.b[start..self.i])
            .ok()
            .and_then(|t| t.parse::<f64>().ok())
            .ok_or("value is not a number")
    }

    /// 2026-10-22: Skip one value of any type, bounded by the input length.
    fn skip(&mut self) -> core::result::Result<(), Why> {
        match self.peek().ok_or("unexpected end")? {
            b'"' => self.string().map(|_| ()),
            open @ (b'{' | b'[') => {
                let close = if open == b'{' { b'}' } else { b']' };
                self.i += 1;
                if self.peek() == Some(close) {
                    self.i += 1;
                    return Ok(());
                }
                loop {
                    if open == b'{' {
                        self.string()?;
                        self.eat(b':')?;
                    }
                    self.skip()?;
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(c) if c == close => {
                            self.i += 1;
                            return Ok(());
                        }
                        _ => return Err("malformed JSON"),
                    }
                }
            }
            _ => {
                let start = self.i;
                while self.i < self.b.len()
                    && !matches!(
                        self.b[self.i],
                        b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r'
                    )
                {
                    self.i += 1;
                }
                if self.i == start {
                    Err("malformed JSON")
                } else {
                    Ok(())
                }
            }
        }
    }

    /// 2026-10-22: Position at the value of `seg` in the object or array that starts here.
    fn enter(&mut self, seg: &str) -> core::result::Result<(), Why> {
        match self.peek() {
            Some(b'{') => {
                self.i += 1;
                if self.peek() == Some(b'}') {
                    return Err("path not found");
                }
                loop {
                    let (a, b) = self.string()?;
                    self.eat(b':')?;
                    if &self.b[a..b] == seg.as_bytes() {
                        return Ok(());
                    }
                    self.skip()?;
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(b'}') => return Err("path not found"),
                        _ => return Err("malformed JSON"),
                    }
                }
            }
            Some(b'[') => {
                let want: usize = seg.parse().map_err(|_| "path indexes an array by name")?;
                self.i += 1;
                for _ in 0..want {
                    self.skip()?;
                    if self.peek() != Some(b',') {
                        return Err("path not found");
                    }
                    self.i += 1;
                }
                if self.peek() == Some(b']') {
                    Err("path not found")
                } else {
                    Ok(())
                }
            }
            _ => Err("path descends into a scalar"),
        }
    }
}

/// 2026-10-22: The number at `path` in the JSON document `text`; the reason it is not there
/// otherwise. The whole document must be well-formed (a truncated frame is refused even when
/// the number it reaches is intact).
pub fn json_number_at(text: &str, path: &[&str]) -> core::result::Result<f64, &'static str> {
    let mut whole = Scan {
        b: text.as_bytes(),
        i: 0,
    };
    whole.skip()?;
    if whole.peek().is_some() {
        return Err("trailing bytes after the document");
    }
    let mut s = Scan {
        b: text.as_bytes(),
        i: 0,
    };
    for seg in path {
        s.enter(seg)?;
    }
    s.number()
}
