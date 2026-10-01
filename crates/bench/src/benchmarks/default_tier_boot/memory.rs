// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: What a server reports about its memory (`GET /memory`), and the pure arithmetic
//! the default-tier boot check derives from it. The server fills [`MemoryReport`]; the driver
//! reads it back, so the one type is the wire contract on both sides.
//!
//! Owner: bench, default-tier-boot.
//! Invariants:
//! - A reading the server could not take is `None`, serialised as JSON `null`; [`footprint`]
//!   refuses to derive from one rather than substituting a number.
//! - Pure: no I/O.

use std::collections::BTreeMap;

use crate::result::Verdict;

/// 2026-10-01: Bytes per MiB, the unit of every byte-valued metric this check emits.
pub const MIB: f64 = 1024.0 * 1024.0;

/// 2026-10-01: The body of `GET /memory`. Byte counts unless the name says otherwise.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoryReport {
    /// 2026-10-01: Blocks in the model's main paged KV pool.
    pub kv_blocks: usize,
    /// 2026-10-01: Bytes of one block index across every layer, K and V.
    pub kv_block_bytes: u64,
    /// 2026-10-01: The slot count the model was built with.
    pub max_batch_size: usize,
    /// 2026-10-01: Total device memory times `--gpu-memory-utilization`.
    pub budget_bytes: u64,
    /// 2026-10-01: The GPU backend's allocation ledger at request time; `None` on a backend
    /// without one.
    pub ledger_live_bytes: Option<u64>,
    /// 2026-10-01: Host `MemAvailable`, sampled once before the GPU backend initialised.
    pub mem_available_at_start_bytes: Option<u64>,
    /// 2026-10-01: Host `MemAvailable` at request time.
    pub mem_available_now_bytes: Option<u64>,
    /// 2026-10-01: The server process's `RssAnon` at request time.
    pub rss_anon_bytes: Option<u64>,
}

/// 2026-10-01: The figures the check reports, derived from one [`MemoryReport`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Footprint {
    pub kv_blocks: usize,
    pub max_batch_size: usize,
    pub budget_mib: f64,
    /// 2026-10-01: `MemAvailable` at start minus now, minus the process's anonymous RSS: on a
    /// unified-memory box, what the serve holds on the device.
    pub device_footprint_mib: f64,
    /// 2026-10-01: Footprint minus budget; negative is within budget.
    pub footprint_over_budget_mib: f64,
    pub ledger_live_mib: Option<f64>,
}

/// 2026-10-01: Derive the [`Footprint`]. The subtraction is done in whole bytes, so a footprint
/// exactly at the budget reports 0, not a rounding residue. Errors, naming the fields, when a
/// host reading is missing.
pub fn footprint(report: &MemoryReport) -> Result<Footprint, String> {
    let (Some(start), Some(now), Some(rss)) = (
        report.mem_available_at_start_bytes,
        report.mem_available_now_bytes,
        report.rss_anon_bytes,
    ) else {
        let missing: Vec<&str> = [
            (
                "mem_available_at_start_bytes",
                report.mem_available_at_start_bytes,
            ),
            ("mem_available_now_bytes", report.mem_available_now_bytes),
            ("rss_anon_bytes", report.rss_anon_bytes),
        ]
        .into_iter()
        .filter(|(_, v)| v.is_none())
        .map(|(k, _)| k)
        .collect();
        return Err(format!(
            "/memory reported null for {} — the server could not read the host's memory \
             counters, so there is no device footprint to measure",
            missing.join(", ")
        ));
    };
    let device = i128::from(start) - i128::from(now) - i128::from(rss);
    let over = device - i128::from(report.budget_bytes);
    Ok(Footprint {
        kv_blocks: report.kv_blocks,
        max_batch_size: report.max_batch_size,
        budget_mib: report.budget_bytes as f64 / MIB,
        device_footprint_mib: device as f64 / MIB,
        footprint_over_budget_mib: over as f64 / MIB,
        ledger_live_mib: report.ledger_live_bytes.map(|b| b as f64 / MIB),
    })
}

impl Footprint {
    /// 2026-10-01: The record's metric keys. `ledger_live_mib` only when the backend has a
    /// ledger.
    pub fn metrics(&self, m: &mut BTreeMap<String, f64>) {
        m.insert("kv_blocks".to_string(), self.kv_blocks as f64);
        m.insert("max_batch_size".to_string(), self.max_batch_size as f64);
        m.insert("budget_mib".to_string(), self.budget_mib);
        m.insert(
            "device_footprint_mib".to_string(),
            self.device_footprint_mib,
        );
        m.insert(
            "footprint_over_budget_mib".to_string(),
            self.footprint_over_budget_mib,
        );
        if let Some(ledger) = self.ledger_live_mib {
            m.insert("ledger_live_mib".to_string(), ledger);
        }
    }
}

/// 2026-10-01: The run verdict. A missing reading fails as inconclusive; a measured run is
/// informational, because no bounds are declared for this check yet.
pub fn verdict_for(result: &Result<Footprint, String>) -> Verdict {
    match result {
        Err(why) => Verdict::fail(format!("INCONCLUSIVE: {why}")),
        Ok(f) => Verdict::info(format!(
            "{} KV blocks at {} slots; device footprint {:.0} MiB against a {:.0} MiB budget \
             ({:+.0} MiB) — informational, no bounds are declared yet",
            f.kv_blocks,
            f.max_batch_size,
            f.device_footprint_mib,
            f.budget_mib,
            f.footprint_over_budget_mib,
        )),
    }
}
