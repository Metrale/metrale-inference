// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: One envelope measurement: a candidate entry point timed at one cell (projection
//! class, N x K, rows) on one box, with its accuracy verdict and the digests of its output bytes
//! per input class. The GPU runner (`met envelope sweep`) appends one JSON line per record; the
//! selection reads them back.
//!
//! Owner: metrale-accuracy (envelope).
//! Invariants:
//! - Every field is stated (PCND): a record that misses one is refused, never defaulted.
//! - `time_us` holds one median per repetition, in repetition order; the record's time is their
//!   median ([`Measurement::median_us`]).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// 2026-10-10: Schema of a record line.
pub const SCHEMA: u32 = 1;

/// 2026-10-10: The cell: what one schedule decision covers.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cell {
    /// 2026-10-10: Op base name (`linear`, `lm_head`, `router`).
    pub op: String,
    /// 2026-10-10: Weight format (`nvfp4/g16`, `bf16`, `fp8/block128x128`, ...).
    pub weight: String,
    /// 2026-10-10: Activation (first input) format.
    pub activation: String,
    /// 2026-10-10: Input width K.
    pub k: u64,
    /// 2026-10-10: Output width N.
    pub n: u64,
    /// 2026-10-10: Rows of the launch.
    pub rows: u64,
}

/// 2026-10-10: The candidate's accuracy verdict at the cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// 2026-10-10: Every input class inside its contract (and every mutation caught).
    Pass,
    /// 2026-10-10: Outside its contract on some input class, or a fault.
    Fail,
    /// 2026-10-10: The candidate cannot run this cell (rows outside its launcher, a split it
    /// has no launcher for, not compiled): no time, never a winner.
    Unavailable,
}

/// 2026-10-10: One record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Measurement {
    /// 2026-10-10: [`SCHEMA`].
    pub schema: u32,
    /// 2026-10-10: Hardware class (`gb10`).
    pub hardware: String,
    /// 2026-10-10: The box that measured it (host name).
    pub host: String,
    /// 2026-10-10: The cell.
    pub cell: Cell,
    /// 2026-10-10: The candidate entry point (`module::function`).
    pub kernel: String,
    /// 2026-10-10: Its family (KERNEL_FAMILIES.toml id).
    pub family: String,
    /// 2026-10-10: The entry point today's engine routes this cell to, when the cell belongs to
    /// a served (golden) plan; `None` when no rule plans it.
    pub default: Option<String>,
    /// 2026-10-10: Accuracy verdict.
    pub verdict: Verdict,
    /// 2026-10-10: Why, for `fail` / `unavailable`; empty for `pass`.
    pub detail: String,
    /// 2026-10-10: SHA-256 of the output bytes per input class (empty unless it ran).
    pub digests: BTreeMap<String, String>,
    /// 2026-10-10: Per repetition, the median microseconds of one launch sequence (empty unless
    /// it passed).
    pub time_us: Vec<f64>,
    /// 2026-10-10: The cell's roofline floor, microseconds.
    pub floor_us: f64,
    /// 2026-10-10: A power or thermal slowdown was active during the timing: rerun it.
    pub throttled: bool,
    /// 2026-10-10: GPU temperature after the timing, Celsius.
    pub temp_c: f64,
    /// 2026-10-10: The binary's kernel target closure (`hw/model/quant=hash`).
    pub closure: String,
    /// 2026-10-10: UTC timestamp, RFC 3339.
    pub at: String,
}

impl Measurement {
    /// 2026-10-10: The median of the repetitions, or `None` when it did not run.
    pub fn median_us(&self) -> Option<f64> {
        median(&self.time_us)
    }

    /// 2026-10-10: The spread of the repetitions: (max - min) / median.
    pub fn spread(&self) -> Option<f64> {
        let m = self.median_us()?;
        let lo = self.time_us.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = self.time_us.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (m > 0.0).then(|| (hi - lo) / m)
    }
}

/// 2026-10-10: The median of `xs` (mean of the middle two for an even count).
pub fn median(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() || xs.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    let h = v.len() / 2;
    Some(if v.len() % 2 == 1 {
        v[h]
    } else {
        (v[h - 1] + v[h]) / 2.0
    })
}

/// 2026-10-10: Parse JSON-lines records; blank lines are skipped, a bad line is an error naming
/// its line number.
pub fn parse_records(text: &str) -> Result<Vec<Measurement>, String> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let m: Measurement =
            serde_json::from_str(line).map_err(|e| format!("record line {}: {e}", i + 1))?;
        if m.schema != SCHEMA {
            return Err(format!(
                "record line {}: schema {} (this build reads {SCHEMA})",
                i + 1,
                m.schema
            ));
        }
        out.push(m);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_median_is_the_middle_and_nonfinite_times_are_refused() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), Some(2.5));
        assert_eq!(median(&[]), None);
        assert_eq!(median(&[1.0, f64::NAN]), None);
    }
}
