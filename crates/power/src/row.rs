// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The energy group of a `docs/kernel-perf/measurements.toml` `[[m]]` row (DESIGN §4.1
//! as revised 2026-10-22: floats in J, W and ms; every source's reading stored; the CI with its
//! method and level, n per level, ESS, outliers, seed and order), and its TOML text.
//! `scripts/lib/kernel_perf_energy.py` checks the same fields.
//!
//! Owner: metrale-power.
//! Invariants: [`EnergyRow::toml_lines`] is deterministic: the same row renders the same bytes
//! (floats are rounded to [`SIG`] significant digits, then printed in shortest round-trip form).

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use metrale_abstract_io::power::{Rail, SourceKind};

/// 2026-10-22: Significant digits a stored float keeps.
pub const SIG: usize = 6;

/// 2026-10-22: The confidence of a row (owner decision 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Label {
    /// 2026-10-22: Fewer than 3 repeats.
    Preliminary,
    /// 2026-10-22: ≥ 3 repeats on one box.
    Measured,
    /// 2026-10-22: ≥ 3 repeats, confirmed on two or more boxes (the box random effect estimated).
    Certified,
}

impl Label {
    /// 2026-10-22: The rule.
    #[must_use]
    pub fn of(repeats: u64, boxes: u64) -> Self {
        match (repeats >= 3, boxes) {
            (true, b) if b >= 2 => Self::Certified,
            (true, 1) => Self::Measured,
            _ => Self::Preliminary,
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Preliminary => "preliminary",
            Self::Measured => "measured",
            Self::Certified => "certified",
        }
    }
}

/// 2026-10-22: A closed interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ci {
    pub lo: f64,
    pub hi: f64,
}

/// 2026-10-22: Agreement of two sources over the windows of one run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Agreement {
    pub deming_slope: f64,
    pub deming_intercept_w: f64,
    pub ba_bias_w: f64,
    pub ba_loa_w: Ci,
    pub n: u64,
}

/// 2026-10-22: The energy group.
#[derive(Clone, Debug, PartialEq)]
pub struct EnergyRow {
    pub energy_j: f64,
    pub energy_dyn_j: f64,
    pub avg_w: f64,
    pub static_w: f64,
    pub energy_ci_j: Ci,
    pub energy_dyn_ci_j: Ci,
    pub avg_w_ci: Ci,
    pub energy_ci_level: f64,
    pub energy_ci_method: String,
    pub energy_method: &'static str,
    pub energy_samples: u64,
    pub energy_boxes: u64,
    pub energy_launches: u64,
    pub energy_window_ms: f64,
    pub energy_ess: f64,
    pub energy_outliers: u64,
    pub energy_seed: u64,
    pub energy_order: String,
    pub energy_label: Label,
    pub sm_mhz: Option<f64>,
    pub temp_c: Option<f64>,
    pub throttle_pct: Option<f64>,
    pub power_source: SourceKind,
    pub power_rail: Rail,
    pub energy_by_source: BTreeMap<String, f64>,
    pub energy_dyn_by_source: BTreeMap<String, f64>,
    pub sensor_delta_pct: BTreeMap<String, f64>,
    pub energy_flags: Vec<String>,
    pub energy_agreement: BTreeMap<String, Agreement>,
    pub energy_fingerprint: String,
    pub energy_source: String,
    pub trace_sha256: String,
    pub energy_floor_class: Option<String>,
    pub energy_floor_j: Option<f64>,
    pub pct_of_energy_floor: Option<f64>,
}

/// 2026-10-22: `v` rounded to [`SIG`] significant digits, printed so TOML reads a float back.
#[must_use]
pub fn float(v: f64) -> String {
    if !v.is_finite() {
        return format!("{v:?}");
    }
    let rounded: f64 = format!("{v:.*e}", SIG - 1).parse().unwrap_or(v);
    format!("{rounded:?}")
}

fn quote(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn ci(c: &Ci) -> String {
    format!("[{}, {}]", float(c.lo), float(c.hi))
}

fn table(m: &BTreeMap<String, f64>) -> String {
    let body: Vec<String> = m
        .iter()
        .map(|(k, v)| format!("{} = {}", quote(k), float(*v)))
        .collect();
    format!("{{ {} }}", body.join(", "))
}

impl EnergyRow {
    /// 2026-10-22: The group as `key = value` lines, in the order the design lists them.
    #[must_use]
    pub fn toml_lines(&self) -> Vec<String> {
        let mut o = Vec::new();
        let mut kv = |k: &str, v: String| o.push(format!("{k} = {v}"));
        kv("energy_j", float(self.energy_j));
        kv("energy_dyn_j", float(self.energy_dyn_j));
        kv("avg_w", float(self.avg_w));
        kv("static_w", float(self.static_w));
        kv("energy_ci_j", ci(&self.energy_ci_j));
        kv("energy_dyn_ci_j", ci(&self.energy_dyn_ci_j));
        kv("avg_w_ci", ci(&self.avg_w_ci));
        kv("energy_ci_level", float(self.energy_ci_level));
        kv("energy_ci_method", quote(&self.energy_ci_method));
        kv("energy_method", quote(self.energy_method));
        kv("energy_samples", format!("{}", self.energy_samples));
        kv("energy_boxes", format!("{}", self.energy_boxes));
        kv("energy_launches", format!("{}", self.energy_launches));
        kv("energy_window_ms", float(self.energy_window_ms));
        kv("energy_ess", float(self.energy_ess));
        kv("energy_outliers", format!("{}", self.energy_outliers));
        kv("energy_seed", format!("{}", self.energy_seed));
        kv("energy_order", quote(&self.energy_order));
        kv("energy_label", quote(self.energy_label.as_str()));
        for (k, v) in [
            ("sm_mhz", self.sm_mhz),
            ("temp_c", self.temp_c),
            ("throttle_pct", self.throttle_pct),
        ] {
            if let Some(v) = v {
                kv(k, float(v));
            }
        }
        kv("power_source", quote(self.power_source.as_str()));
        kv("power_rail", quote(self.power_rail.as_str()));
        kv("energy_by_source", table(&self.energy_by_source));
        kv("energy_dyn_by_source", table(&self.energy_dyn_by_source));
        kv("sensor_delta_pct", table(&self.sensor_delta_pct));
        let flags: Vec<String> = self.energy_flags.iter().map(|f| quote(f)).collect();
        kv("energy_flags", format!("[{}]", flags.join(", ")));
        let agree: Vec<String> = self
            .energy_agreement
            .iter()
            .map(|(k, a)| {
                format!(
                    "{} = {{ deming_slope = {}, deming_intercept_w = {}, ba_bias_w = {}, ba_loa_w = {}, n = {} }}",
                    quote(k),
                    float(a.deming_slope),
                    float(a.deming_intercept_w),
                    float(a.ba_bias_w),
                    ci(&a.ba_loa_w),
                    a.n
                )
            })
            .collect();
        kv("energy_agreement", format!("{{ {} }}", agree.join(", ")));
        kv("energy_fingerprint", quote(&self.energy_fingerprint));
        kv("energy_source", quote(&self.energy_source));
        kv("trace_sha256", quote(&self.trace_sha256));
        if let Some(c) = &self.energy_floor_class {
            kv("energy_floor_class", quote(c));
        }
        if let Some(v) = self.energy_floor_j {
            // 2026-10-22: Unrounded: the checker re-derives it to a relative 1e-6.
            kv("energy_floor_j", format!("{v:?}"));
        }
        if let Some(v) = self.pct_of_energy_floor {
            kv("pct_of_energy_floor", format!("{:.2}", v));
        }
        o
    }
}
