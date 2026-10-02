// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The memory budget of a device class and the queries over it: the driver's share
//! (`kernels/<class>/HARDWARE.toml [memory]`, calibrated per class), the util budget, and the
//! inverse questions "the most sequences at this length" and "the longest length at this many
//! sequences", answered by bisection over the one evaluator the forward report uses.
//!
//! Owner: metrale-circuit (memory).
//! Invariants:
//! - Every driver term is read from the class's `[memory]` table; a missing key is an error,
//!   never a default (#72's `runtime_headroom.rs` constants are this table's first source).
//! - The inverse queries assume the footprint never shrinks as sequences or tokens grow, and
//!   they check the answer: the bound they return fits, the next value does not (or is the cap).

use serde::Deserialize;

/// 2026-10-02: What the driver holds beyond the allocation ledger, for one device class.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DriverTerms {
    /// 2026-10-02: The CUDA context, loaded modules, the local-memory reservation and the
    /// graphs, bytes.
    pub driver_fixed_bytes: u64,
    /// 2026-10-02: The driver's per-allocation bookkeeping, per mille of the util budget.
    pub driver_budget_per_mille: u64,
    /// 2026-10-02: Host memory is the device's memory (a unified-memory SoC): host caches compete
    /// for the same pool, outside the util budget.
    pub unified: bool,
    /// 2026-10-02: The largest `--gpu-memory-utilization` a serve of this class may run at.
    pub util_ceiling: f64,
}

/// 2026-10-02: Why the class's memory terms could not be read.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BudgetError {
    /// 2026-10-02: The HARDWARE.toml text, or its `[memory]` table.
    #[error("kernels/{class}/HARDWARE.toml [memory]: {detail}")]
    Terms {
        /// 2026-10-02: The class.
        class: String,
        /// 2026-10-02: What was wrong.
        detail: String,
    },
    /// 2026-10-02: An inverse query whose lower bound does not fit, or whose evaluator failed.
    #[error("{0}")]
    Query(String),
}

impl DriverTerms {
    /// 2026-10-02: The `[memory]` table of `class`'s HARDWARE.toml text.
    pub fn parse(class: &str, hardware_toml: &str) -> Result<Self, BudgetError> {
        let err = |detail: String| BudgetError::Terms {
            class: class.to_string(),
            detail,
        };
        let mut t: toml::Table = toml::from_str(hardware_toml).map_err(|e| err(e.to_string()))?;
        let mem = t
            .remove("memory")
            .ok_or_else(|| err("the class declares no [memory] table".into()))?;
        let terms: Self = mem
            .try_into()
            .map_err(|e: toml::de::Error| err(e.to_string()))?;
        if terms.driver_budget_per_mille > 1000
            || !(terms.util_ceiling > 0.0 && terms.util_ceiling <= 1.0)
        {
            return Err(err(
                "driver_budget_per_mille is at most 1000 and util_ceiling in (0, 1]".into(),
            ));
        }
        Ok(terms)
    }

    /// 2026-10-02: The driver's bytes under a util budget of `budget_bytes`, with
    /// `chunk_slack` bytes of small-allocation chunks (measured from a ledger, or 0 when no
    /// ledger is known).
    pub fn bytes(&self, budget_bytes: u64, chunk_slack: u64) -> u64 {
        self.driver_fixed_bytes
            .saturating_add(budget_bytes / 1000 * self.driver_budget_per_mille)
            .saturating_add(chunk_slack)
    }
}

/// 2026-10-02: The util budget: `device_bytes x util`, as the engine computes it (truncated).
pub fn util_budget(device_bytes: u64, util: f64) -> u64 {
    (device_bytes as f64 * util) as u64
}

/// 2026-10-02: The largest `x` in `lo..=hi` with `fits(x)`, by bisection; `None` when `lo` does
/// not fit. `fits` must be monotone (true up to some `x`, false after).
pub fn largest_fitting(
    lo: u64,
    hi: u64,
    fits: &mut dyn FnMut(u64) -> Result<bool, BudgetError>,
) -> Result<Option<u64>, BudgetError> {
    if lo > hi {
        return Err(BudgetError::Query(format!("empty range {lo}..={hi}")));
    }
    if !fits(lo)? {
        return Ok(None);
    }
    if fits(hi)? {
        return Ok(Some(hi));
    }
    // 2026-10-02: `good` fits, `bad` does not; halve the gap.
    let (mut good, mut bad) = (lo, hi);
    while bad - good > 1 {
        let mid = good + (bad - good) / 2;
        if fits(mid)? {
            good = mid;
        } else {
            bad = mid;
        }
    }
    Ok(Some(good))
}

#[cfg(test)]
#[path = "budget_tests.rs"]
mod budget_tests;
