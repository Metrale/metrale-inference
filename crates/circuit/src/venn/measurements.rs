// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: The rows of `docs/kernel-perf/measurements.toml` an evidence record can cite,
//! keyed `<kernel> @ <regime>`. The numbers stay in that file (its schema and checks are
//! KERNEL-PERF.md's); the manifest only maps a row to a family's parameter point.
//!
//! Owner: metrale-circuit (venn).
//! Invariants: a key names at most one row; a duplicate key is a load error.

use std::collections::BTreeMap;

use serde::Deserialize;

/// 2026-09-29: What an evidence cell shows of one row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measured {
    /// 2026-09-29: Measured kernel time.
    pub time_us: f64,
    /// 2026-09-29: Roofline floor over measured time, percent.
    pub pct_of_floor: f64,
}

/// 2026-09-29: The rows by key.
pub type Measurements = BTreeMap<String, Measured>;

#[derive(Deserialize)]
struct File {
    m: Vec<Row>,
}

#[derive(Deserialize)]
struct Row {
    kernel: String,
    regime: String,
    time_us: f64,
    pct_of_floor: f64,
}

/// 2026-09-29: Parse measurements.toml (only the fields evidence shows; the file's other
/// fields are its generator's business).
pub fn parse_measurements(text: &str) -> Result<Measurements, String> {
    let file: File = toml::from_str(text).map_err(|e| format!("measurements.toml: {e}"))?;
    let mut out = Measurements::new();
    for r in file.m {
        let key = format!("{} @ {}", r.kernel, r.regime);
        let row = Measured {
            time_us: r.time_us,
            pct_of_floor: r.pct_of_floor,
        };
        if out.insert(key.clone(), row).is_some() {
            return Err(format!("measurements.toml: `{key}` names two rows"));
        }
    }
    Ok(out)
}
