// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The GPU-rail energy counter, read from the SERVE's own
//! `/metrics` (the NVML cumulative counter the telemetry crate exports as
//! `metrale_gpu_energy_millijoules_total`, already reset- and wrap-corrected
//! server-side — `crates/telemetry/src/export/prometheus.rs`), at exactly two
//! boundary scrapes per measured window. Never linked into this crate: the
//! scrape goes over the HTTP connection `metrale-bench` already holds to the
//! target, reusing [`crate::http::get_text`] and the spec-cost benchmark's
//! Prometheus reader ([`crate::benchmarks::spec_cost::prom::Scrape`],
//! [`crate::benchmarks::spec_cost::cell::ENERGY_METRIC`]) rather than a
//! second copy of either.
//!
//! This is a SECOND, independently-sourced energy reading beside
//! [`super::energy`]'s rail-power integral (`nvidia-smi power.draw.average`,
//! local-process, loopback-only). The two are not reconciled here: both are
//! recorded under different key names (`gpu_rail_*` vs `gpu_energy_counter_*`)
//! so a reader can compare them per record. No existing `gpu_rail_*` key or
//! BENCH.toml ceiling changes meaning.
//!
//! Unlike the rail sampler, this one is not loopback-restricted: a scrape
//! reads the SERVING box's own counter over HTTP, wherever that box is.
//!
//! Owner: bench hardware.
//! Invariants:
//! - No polling inside a measured window: exactly one scrape at the window's
//!   start and one at its end, both strictly outside the timed section (the
//!   caller captures its `Instant`s only after the start scrape returns and
//!   before the end scrape begins), so this never contends with, or delays,
//!   the traffic being measured. This was the whole point of the re-cut: the
//!   original attempt here polled every 250 ms for the life of the run,
//!   which risked contaminating the measured windows it was supposed to
//!   describe, alongside duplicating `get_text` and the Prometheus parser
//!   that had since landed on main for spec-cost.
//! - A scrape that cannot reach the endpoint, or whose body carries no
//!   [`ENERGY_METRIC`] line, is absent — never a fabricated 0.0.
//! - A window whose end reading is not strictly above its start reading is
//!   `None`, never a negative or wrapped-around joule count: the server
//!   already corrects for driver-level counter resets, so a non-increase
//!   between two scrapes means the scrape pair itself is not trustworthy
//!   (e.g. a restart between the two requests), not a reset to compensate
//!   for again.

use std::collections::BTreeMap;
use std::time::Duration;

use crate::benchmarks::spec_cost::cell::ENERGY_METRIC;
use crate::benchmarks::spec_cost::prom::Scrape;
use crate::http;
use crate::plugin::TargetEndpoint;

/// 2026-10-05: Bound on the one GET this module ever makes per boundary.
const SCRAPE_TIMEOUT: Duration = Duration::from_secs(2);

/// 2026-10-05: One `/metrics` scrape's [`ENERGY_METRIC`] value, millijoules.
/// `None` on any failure (unreachable endpoint, non-200, the metric absent,
/// an unparseable page) — logged by nothing here; a caller that wants a line
/// for its run log reads the `None` and says so itself, the same way
/// `EnergyMeter::start` already does for the rail sampler.
pub async fn read_mj(target: &TargetEndpoint) -> Option<f64> {
    let text = http::get_text(target, "/metrics", SCRAPE_TIMEOUT)
        .await
        .ok()?;
    let scrape = Scrape::parse(&text).ok()?;
    scrape.value(ENERGY_METRIC, &[]).ok()?
}

/// 2026-10-05: Joules between two boundary readings. `None` when either
/// reading is absent, or the counter did not strictly increase (a flat or
/// backwards pair) — a delivered token never costs zero energy, so this is
/// "no evidence", not "a free window".
pub fn window_joules(before_mj: Option<f64>, after_mj: Option<f64>) -> Option<f64> {
    let (b, a) = (before_mj?, after_mj?);
    let delta_mj = a - b;
    (delta_mj > 0.0 && delta_mj.is_finite()).then(|| delta_mj / 1000.0)
}

/// 2026-10-05: The record keys for one window, under `prefix` — the same
/// shape as [`super::energy::EnergyWindow::metrics`]'s `gpu_rail_*` keys, so
/// the two are directly comparable per rung. `energy_j: None` writes nothing,
/// never a zero.
pub fn metrics(prefix: &str, energy_j: Option<f64>, tokens: usize, m: &mut BTreeMap<String, f64>) {
    let Some(j) = energy_j else { return };
    m.insert(format!("{prefix}gpu_energy_counter_j"), j);
    if let Some(r) = super::energy::joules_per_token(j, tokens) {
        m.insert(format!("{prefix}gpu_energy_counter_jpt"), r);
    }
}

#[cfg(test)]
#[path = "gpu_energy_counter_tests.rs"]
mod tests;
