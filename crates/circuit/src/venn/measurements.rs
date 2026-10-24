// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: The rows of `docs/kernel-perf/measurements.toml` an evidence record can cite,
//! keyed `<kernel> @ <regime>`. The numbers stay in that file (its schema and checks are
//! KERNEL-PERF.md's); the manifest only maps a row to a family's parameter point.
//!
//! Owner: metrale-circuit (venn).
//! Invariants: a key names at most one row; a duplicate key is a load error.

use crate::prelude::*;

use alloc::collections::BTreeMap;

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

/// 2026-10-23: One `[[m]]` row with every field a reader outside the circuit uses (the knowledge
/// graph, metrale-kg-derive), so measurements.toml keeps one Rust reader. Fields the generator
/// adds later (the energy group) land in `extra` by name, unparsed.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct MeasurementRow {
    pub kernel: String,
    pub regime: String,
    pub file: String,
    pub hardware: String,
    pub model: String,
    pub time_us: f64,
    pub bytes: f64,
    pub flops: f64,
    pub bound: String,
    pub floor_us: f64,
    pub pct_of_floor: f64,
    pub source: String,
    pub notes: String,
    /// 2026-10-23: Every other field (`energy_*`, `sm_mhz`, ...), as written.
    #[serde(flatten)]
    pub extra: BTreeMap<String, toml::Value>,
}

#[derive(Deserialize)]
struct FullFile {
    m: Vec<MeasurementRow>,
}

/// 2026-10-23: Every row in file order, with the same duplicate-key refusal as
/// [`parse_measurements`].
pub fn parse_measurement_rows(text: &str) -> Result<Vec<MeasurementRow>, String> {
    let file: FullFile = toml::from_str(text).map_err(|e| format!("measurements.toml: {e}"))?;
    let mut seen = alloc::collections::BTreeSet::new();
    for r in &file.m {
        let key = format!("{} @ {}", r.kernel, r.regime);
        if !seen.insert(key.clone()) {
            return Err(format!("measurements.toml: `{key}` names two rows"));
        }
    }
    Ok(file.m)
}
