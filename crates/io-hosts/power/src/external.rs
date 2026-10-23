// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: [`ExternalMeter`]: an external meter (inline USB-C PD meter, smart wall plug) as
//! a [`PowerProbe`]: a [`Transport`] yields one frame per read and a
//! `metrale_power::external::MeterProtocol` turns it into a sample. Transports here: newline
//! frames from a read-only serial device or file ([`SerialLines`]), fixed-size reports from a
//! read-only hidraw node ([`HidrawReports`]), an HTTP GET poll (`http.rs`) and an MQTT
//! subscription (`mqtt.rs`). The device is opened as it is already configured; nothing here
//! sets a baud rate, a report descriptor or any host setting.
//!
//! Owner: metrale-io-host-power.
//! Invariants: a frame source that ends is `ExternalMeterTransport`, never an empty sample.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{PowerCapabilities, PowerProbe, PowerSample};
use metrale_error::{ErrorCode, error};
use metrale_power::external::MeterProtocol;

/// 2026-10-22: One frame per call, blocking for the meter's own cadence.
pub trait Transport {
    /// 2026-10-22: The next frame's bytes.
    fn next_frame(&mut self) -> metrale_abstract_io::Result<Vec<u8>>;
}

pub(crate) fn transport_error(meter: &str, why: String) -> metrale_error::Error {
    error!(ErrorCode::ExternalMeterTransport, meter.to_string(), why)
}

/// 2026-10-22: A transport read through a protocol.
pub struct ExternalMeter<T: Transport, P: MeterProtocol> {
    transport: T,
    protocol: P,
}

impl<T: Transport, P: MeterProtocol> ExternalMeter<T, P> {
    #[must_use]
    pub fn new(transport: T, protocol: P) -> Self {
        Self {
            transport,
            protocol,
        }
    }
}

impl<T: Transport, P: MeterProtocol> PowerProbe for ExternalMeter<T, P> {
    fn capabilities(&self) -> &PowerCapabilities {
        self.protocol.capabilities()
    }

    fn read(&mut self, at: MonoNs) -> metrale_abstract_io::Result<PowerSample> {
        let frame = self.transport.next_frame()?;
        self.protocol.parse(&frame, at)
    }
}

fn open_read_only(meter: &str, path: &Path) -> metrale_abstract_io::Result<File> {
    OpenOptions::new().read(true).open(path).map_err(|e| {
        error!(
            ErrorCode::PowerSourceUnavailable,
            meter.to_string(),
            format!("cannot open {} read-only: {e}", path.display())
        )
    })
}

/// 2026-10-22: Newline-terminated frames from a serial device (or a file), read-only. Blank
/// lines are skipped; a trailing `\r` is dropped.
pub struct SerialLines {
    meter: String,
    lines: BufReader<File>,
}

impl SerialLines {
    pub fn open(meter: &str, path: &Path) -> metrale_abstract_io::Result<Self> {
        Ok(Self {
            meter: meter.to_string(),
            lines: BufReader::new(open_read_only(meter, path)?),
        })
    }
}

impl Transport for SerialLines {
    fn next_frame(&mut self) -> metrale_abstract_io::Result<Vec<u8>> {
        loop {
            let mut buf = Vec::new();
            let n = self
                .lines
                .read_until(b'\n', &mut buf)
                .map_err(|e| transport_error(&self.meter, e.to_string()))?;
            if n == 0 {
                return Err(transport_error(
                    &self.meter,
                    "the device closed".to_string(),
                ));
            }
            while matches!(buf.last(), Some(b'\n' | b'\r')) {
                buf.pop();
            }
            if !buf.is_empty() {
                return Ok(buf);
            }
        }
    }
}

/// 2026-10-22: Fixed-size input reports from a hidraw node, read-only; the size is the
/// protocol's declared report length.
pub struct HidrawReports {
    meter: String,
    dev: File,
    report_len: usize,
}

impl HidrawReports {
    pub fn open(meter: &str, path: &Path, report_len: usize) -> metrale_abstract_io::Result<Self> {
        Ok(Self {
            meter: meter.to_string(),
            dev: open_read_only(meter, path)?,
            report_len,
        })
    }
}

impl Transport for HidrawReports {
    fn next_frame(&mut self) -> metrale_abstract_io::Result<Vec<u8>> {
        let mut buf = vec![0u8; self.report_len];
        self.dev
            .read_exact(&mut buf)
            .map_err(|e| transport_error(&self.meter, format!("short report: {e}")))?;
        Ok(buf)
    }
}

#[cfg(test)]
#[path = "external_tests.rs"]
mod tests;
