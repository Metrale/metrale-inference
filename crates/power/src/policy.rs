// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The window rules a steady-state attribution is held to. Every bound is explicit
//! (PCND): a run states its policy, the policy is written into the trace (`meta,policy,...`), and
//! a recompute reads it back, so the same trace always meets the same rules (HPAR).
//!
//! [`REFUSAL_FLOOR`] is the hard floor no policy may go below (DESIGN §1.3: a load window
//! shorter than 3 s, with fewer than 30 counter updates or fewer than 500 launches is refused,
//! never integrated); [`OF_RECORD`] is what the `kernel-energy` bench runs (§3.1).
//!
//! Owner: metrale-power.
//! Invariants: [`WindowPolicy::parse`] inverts [`WindowPolicy::encode`]; a policy below the floor
//! is refused by [`WindowPolicy::check`].

use alloc::format;
use alloc::string::String;

use metrale_error::{ErrorCode, error};

use crate::Result;

/// 2026-10-22: The bounds one attribution is held to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowPolicy {
    /// 2026-10-22: Shortest load window, s.
    pub min_load_s: f64,
    /// 2026-10-22: Shortest idle baseline, s.
    pub min_idle_s: f64,
    /// 2026-10-22: Fewest value updates of a channel inside a load window.
    pub min_updates: u64,
    /// 2026-10-22: Fewest launches inside a load window.
    pub min_launches: u64,
    /// 2026-10-22: Largest |A − B| / mean(A, B) of a repeat's two idle baselines, percent.
    pub max_baseline_drift_pct: f64,
    /// 2026-10-22: Largest share of load samples carrying a throttle reason, percent.
    pub max_throttle_pct: f64,
    /// 2026-10-22: Lowest mean SM clock under load, MHz (the sm_clock_fault floor).
    pub min_sm_mhz: f64,
    /// 2026-10-22: |sensor delta| above which a source pair is flagged, percent.
    pub sensor_flag_pct: f64,
}

/// 2026-10-22: The floor every policy must meet (DESIGN §1.3).
pub const REFUSAL_FLOOR: WindowPolicy = WindowPolicy {
    min_load_s: 3.0,
    min_idle_s: 3.0,
    min_updates: 30,
    min_launches: 500,
    max_baseline_drift_pct: 15.0,
    max_throttle_pct: 5.0,
    min_sm_mhz: 2000.0,
    sensor_flag_pct: 10.0,
};

/// 2026-10-22: The steady-state method of record (DESIGN §3.1): ≥ 10 s windows, ≥ 2000 launches.
pub const OF_RECORD: WindowPolicy = WindowPolicy {
    min_load_s: 10.0,
    min_idle_s: 10.0,
    min_updates: 90,
    min_launches: 2000,
    ..REFUSAL_FLOOR
};

impl WindowPolicy {
    /// 2026-10-22: `Ok` when every bound is at least as strict as [`REFUSAL_FLOOR`].
    pub fn check(&self) -> Result<()> {
        let f = REFUSAL_FLOOR;
        let loose = self.min_load_s < f.min_load_s
            || self.min_idle_s < f.min_idle_s
            || self.min_updates < f.min_updates
            || self.min_launches < f.min_launches
            || self.max_baseline_drift_pct > f.max_baseline_drift_pct
            || self.max_throttle_pct > f.max_throttle_pct
            || self.min_sm_mhz < f.min_sm_mhz
            || self.sensor_flag_pct > f.sensor_flag_pct;
        if loose {
            return Err(error!(
                ErrorCode::EnergyInputInvalid,
                format!(
                    "policy {} is looser than the refusal floor {}",
                    self.encode(),
                    f.encode()
                )
            ));
        }
        Ok(())
    }

    /// 2026-10-22: The trace form, `key=value` pairs separated by `;`.
    #[must_use]
    pub fn encode(&self) -> String {
        format!(
            "min_load_s={:?};min_idle_s={:?};min_updates={};min_launches={};\
             max_baseline_drift_pct={:?};max_throttle_pct={:?};min_sm_mhz={:?};sensor_flag_pct={:?}",
            self.min_load_s,
            self.min_idle_s,
            self.min_updates,
            self.min_launches,
            self.max_baseline_drift_pct,
            self.max_throttle_pct,
            self.min_sm_mhz,
            self.sensor_flag_pct
        )
    }

    /// 2026-10-22: Inverse of [`WindowPolicy::encode`]; every key is required.
    pub fn parse(s: &str) -> Result<Self> {
        let get = |k: &str| -> Result<&str> {
            s.split(';')
                .filter_map(|p| p.split_once('='))
                .find(|(key, _)| *key == k)
                .map(|(_, v)| v)
                .ok_or_else(|| error!(ErrorCode::EnergyInputInvalid, format!("policy lacks {k}")))
        };
        let f = |k: &str| -> Result<f64> {
            get(k)?.parse().map_err(|_| {
                error!(
                    ErrorCode::EnergyInputInvalid,
                    format!("policy {k} is not a number")
                )
            })
        };
        let u = |k: &str| -> Result<u64> {
            get(k)?.parse().map_err(|_| {
                error!(
                    ErrorCode::EnergyInputInvalid,
                    format!("policy {k} is not an integer")
                )
            })
        };
        Ok(Self {
            min_load_s: f("min_load_s")?,
            min_idle_s: f("min_idle_s")?,
            min_updates: u("min_updates")?,
            min_launches: u("min_launches")?,
            max_baseline_drift_pct: f("max_baseline_drift_pct")?,
            max_throttle_pct: f("max_throttle_pct")?,
            min_sm_mhz: f("min_sm_mhz")?,
            sensor_flag_pct: f("sensor_flag_pct")?,
        })
    }
}
