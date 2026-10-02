// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: What `GET /memory` reports: the KV pool and slot count the model was built
//! with, the serve's memory budget, and host memory readings, as the wire type
//! `metrale_bench::benchmarks::default_tier_boot::MemoryReport`. On a unified-memory device,
//! device allocations come out of host RAM, so the drop in host `MemAvailable` since process
//! start, less the process's own anonymous RSS, is the serve's device footprint.
//!
//! Owner: server (HTTP layer).
//! Invariants:
//! - Host `MemAvailable` at start is sampled at most once per process
//!   ([`record_mem_available_at_start`]), before the first GPU backend initialises.
//! - A reading that cannot be taken is `None` (JSON `null`), never a substituted number.
//! - Only [`read_proc`] touches the filesystem; parsing and assembly are pure.

use std::sync::{Arc, OnceLock};

use metrale_bench::benchmarks::default_tier_boot::MemoryReport;

/// 2026-10-01: Reads the GPU backend's allocation ledger, at request time.
pub type LedgerReader = Arc<dyn Fn() -> usize + Send + Sync>;

/// 2026-10-01: What the device side fixes once its backend is up.
#[derive(Clone)]
pub struct DeviceBudget {
    /// 2026-10-01: [`budget_bytes`] of the device's total memory.
    pub budget_bytes: u64,
    /// 2026-10-01: `None` on a backend without an allocation ledger.
    pub ledger: Option<LedgerReader>,
}

/// 2026-10-01: Fixed at load: the built model's KV pool and slots, and the device budget.
#[derive(Clone)]
pub struct MemoryFacts {
    pub kv_blocks: usize,
    pub kv_block_bytes: u64,
    pub max_batch_size: usize,
    pub device: DeviceBudget,
}

/// 2026-10-01: The readings one request takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Readings {
    pub ledger_live_bytes: Option<u64>,
    pub mem_available_at_start_bytes: Option<u64>,
    pub mem_available_now_bytes: Option<u64>,
    pub rss_anon_bytes: Option<u64>,
}

const MEMINFO: &str = "/proc/meminfo";
const SELF_STATUS: &str = "/proc/self/status";

static MEM_AVAILABLE_AT_START: OnceLock<Option<u64>> = OnceLock::new();

/// 2026-10-01: Sample host `MemAvailable` the first time it is called; later calls (a model
/// swap reloads) keep the first sample, which is the process-start baseline.
pub(crate) fn record_mem_available_at_start() {
    MEM_AVAILABLE_AT_START
        .get_or_init(|| read_proc(MEMINFO).as_deref().and_then(mem_available_bytes));
}

/// 2026-10-01: `total_bytes × utilization`, rounded down: how `factory::build` sizes the
/// serve's budget from `--gpu-memory-utilization`.
pub(crate) fn budget_bytes(total_bytes: usize, utilization: f64) -> u64 {
    (total_bytes as f64 * utilization) as u64
}

impl MemoryFacts {
    /// 2026-10-01: Take this request's readings: the ledger, and host memory now.
    pub(crate) fn readings(&self) -> Readings {
        Readings {
            ledger_live_bytes: self.device.ledger.as_ref().map(|read| read() as u64),
            mem_available_at_start_bytes: MEM_AVAILABLE_AT_START.get().copied().flatten(),
            mem_available_now_bytes: read_proc(MEMINFO).as_deref().and_then(mem_available_bytes),
            rss_anon_bytes: read_proc(SELF_STATUS).as_deref().and_then(rss_anon_bytes),
        }
    }

    /// 2026-10-01: The `/memory` document for `readings`.
    pub(crate) fn report(&self, readings: Readings) -> MemoryReport {
        MemoryReport {
            kv_blocks: self.kv_blocks,
            kv_block_bytes: self.kv_block_bytes,
            max_batch_size: self.max_batch_size,
            budget_bytes: self.device.budget_bytes,
            ledger_live_bytes: readings.ledger_live_bytes,
            mem_available_at_start_bytes: readings.mem_available_at_start_bytes,
            mem_available_now_bytes: readings.mem_available_now_bytes,
            rss_anon_bytes: readings.rss_anon_bytes,
        }
    }
}

/// 2026-10-01: `MemAvailable` from `/proc/meminfo` text, in bytes.
pub(crate) fn mem_available_bytes(meminfo: &str) -> Option<u64> {
    kib_field(meminfo, "MemAvailable")
}

/// 2026-10-01: `RssAnon` from `/proc/<pid>/status` text, in bytes.
pub(crate) fn rss_anon_bytes(status: &str) -> Option<u64> {
    kib_field(status, "RssAnon")
}

/// 2026-10-01: The value of a `Key:   <n> kB` line, in bytes; `None` when the key is absent
/// or its line is not in that form.
fn kib_field(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|line| {
        let rest = line.strip_prefix(key)?.strip_prefix(':')?;
        let mut parts = rest.split_whitespace();
        let kib: u64 = parts.next()?.parse().ok()?;
        (parts.next()? == "kB" && parts.next().is_none())
            .then(|| kib.checked_mul(1024))
            .flatten()
    })
}

/// 2026-10-01: One `/proc` file's text. A read failure is logged and gives `None`.
#[cfg(target_os = "linux")]
pub(crate) fn read_proc(path: &str) -> Option<String> {
    match std::fs::read_to_string(path) {
        Ok(text) => Some(text),
        Err(e) => {
            tracing::warn!("GET /memory: reading {path} failed: {e}; the field reports null");
            None
        }
    }
}

/// 2026-10-01: No `/proc` off Linux: every host reading is `None`.
#[cfg(not(target_os = "linux"))]
pub(crate) fn read_proc(_path: &str) -> Option<String> {
    None
}

#[cfg(test)]
#[path = "memory_probe_tests.rs"]
mod tests;
