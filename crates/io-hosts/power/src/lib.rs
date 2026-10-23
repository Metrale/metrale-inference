// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Process hosts for the power port (DESIGN-kernel-energy.md §2): [`ThreadedMeter`]
//! turns any blocking `PowerProbe` into an issue/complete `PowerMeter`; the probes here are the
//! `nvidia-smi` child, read-only hwmon, external meters (serial, hidraw, HTTP, MQTT transports
//! under the core's protocols) and, on macOS, Apple IOReport. NVML and amdsmi probes are device
//! code and live in `metrale-telemetry` (VBOP). A binary chooses which to open.
//!
//! Owner: metrale-io-host-power.
//! Invariants:
//! - Read-only on the host: no sysfs, driver or tty writes, no root.
//! - `unsafe` only in the macOS IOReport FFI module (`#![deny(unsafe_code)]` here, allowed there
//!   with a SAFETY comment on every block): IOReport has no safe binding.
#![deny(unsafe_code)]

pub mod external;
pub mod http;
pub mod hwmon;
#[cfg(target_os = "macos")]
pub mod ioreport;
pub mod mqtt;
pub mod nvidia_smi;
pub mod threaded;

pub use external::{ExternalMeter, HidrawReports, SerialLines, Transport};
pub use http::HttpPoll;
pub use hwmon::{HwmonMapping, HwmonProbe};
pub use mqtt::{MqttConfig, MqttSubscribe};
pub use nvidia_smi::NvidiaSmiProbe;
pub use threaded::ThreadedMeter;
