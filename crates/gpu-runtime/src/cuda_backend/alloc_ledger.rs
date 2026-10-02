// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The device-allocation ledger's entries and reports: size and
//! allocating call site of every live allocation.
//! 2026-10-01: Also [`LedgerProbe`], which reads the ledger without the backend, and
//! [`chunk_slack`]: the device memory the driver holds beyond the ledger for small allocations.
//!
//! Owner: gpu-runtime (CUDA backend).
//! Invariants: none beyond the types.

use super::MetraleCudaBackend;

/// 2026-09-25: One live device allocation: its size in bytes and the call
/// site of `GpuBackend::alloc` or `alloc_managed` that made it (both are
/// `#[track_caller]` on the trait and on the CUDA impl).
#[derive(Clone, Copy)]
pub(super) struct AllocRecord {
    pub(super) bytes: usize,
    pub(super) site: &'static std::panic::Location<'static>,
}

/// 2026-10-01: A read handle on one backend's ledger, for a reader that cannot hold the
/// backend, which moves into the model it builds. It reads the map
/// [`MetraleCudaBackend::live_bytes`] reads, so the two always agree.
#[derive(Clone)]
pub struct LedgerProbe(
    std::sync::Arc<parking_lot::Mutex<std::collections::HashMap<u64, AllocRecord>>>,
);

impl LedgerProbe {
    /// 2026-10-01: Total bytes on the ledger now.
    pub fn live_bytes(&self) -> usize {
        ledger_bytes(&self.0.lock())
    }
}

fn ledger_bytes(ledger: &std::collections::HashMap<u64, AllocRecord>) -> usize {
    ledger.values().map(|r| r.bytes).sum()
}

/// 2026-10-01: The chunk the driver packs allocations smaller than itself into, on GB10: such
/// allocations come back at 512 B to 64 KiB alignment inside 2 MiB-aligned ranges, while larger
/// ones start 2 MiB-aligned (nsys device-memory events of two serves, 2026-10-01). A chunk that
/// holds any live small allocation stays resident whole.
pub const SMALL_ALLOC_CHUNK: u64 = 2 << 20;

/// 2026-10-01: The bytes of the `chunk`-sized ranges that hold live allocations smaller than
/// `chunk`, less those allocations' bytes: memory the driver keeps resident that no ledger entry
/// accounts for. `allocs` is `(device address, bytes)`; allocations of `chunk` bytes or more are
/// not packed and count nothing. On GB10 this is 52 MiB for the dense 27B (2,274 small
/// allocations in 270 chunks) and 429 MiB for Nemotron-3-Nano (24,456 in 2,033 chunks), which
/// matches the latter's driver use above the other reserve terms to within 11 MiB.
pub fn chunk_slack(allocs: impl Iterator<Item = (u64, usize)>, chunk: u64) -> usize {
    let mut chunks = std::collections::BTreeSet::new();
    let mut small = 0usize;
    for (addr, bytes) in allocs.filter(|&(_, b)| (b as u64) < chunk && b > 0) {
        small += bytes;
        let last = addr + bytes as u64 - 1;
        chunks.extend(addr / chunk..=last / chunk);
    }
    (chunks.len() as u64 * chunk).saturating_sub(small as u64) as usize
}

impl MetraleCudaBackend {
    /// 2026-09-25: Enter an allocation in the ledger. `site` is the caller of
    /// `GpuBackend::alloc` or `alloc_managed`.
    pub(crate) fn record_alloc(
        &self,
        ptr: crate::gpu::DevicePtr,
        bytes: usize,
        site: &'static std::panic::Location<'static>,
    ) {
        self.live_allocs
            .lock()
            .insert(ptr.0, AllocRecord { bytes, site });
    }

    pub(crate) fn forget_alloc(&self, ptr: crate::gpu::DevicePtr) {
        self.live_allocs.lock().remove(&ptr.0);
    }

    /// 2026-09-25: Total bytes on the ledger.
    pub fn live_bytes(&self) -> usize {
        ledger_bytes(&self.live_allocs.lock())
    }

    /// 2026-10-01: `chunk_slack` of the live allocations, at `SMALL_ALLOC_CHUNK`.
    pub fn chunk_slack_bytes(&self) -> usize {
        let ledger = self.live_allocs.lock();
        chunk_slack(ledger.iter().map(|(&a, r)| (a, r.bytes)), SMALL_ALLOC_CHUNK)
    }

    /// 2026-10-01: A [`LedgerProbe`] on this backend's ledger.
    pub fn ledger_probe(&self) -> LedgerProbe {
        LedgerProbe(std::sync::Arc::clone(&self.live_allocs))
    }

    /// 2026-09-25: A text report of the ledger: the total, then up to `top_n`
    /// call sites of at least `min_mb` MiB, largest first, then (if any are
    /// left) one line summing the other sites, then up to `top_n` source
    /// files, largest first.
    pub fn alloc_report(&self, top_n: usize, min_mb: usize) -> String {
        use std::collections::HashMap;
        let mut by_site: HashMap<String, (usize, usize)> = HashMap::new();
        let mut total = 0usize;
        for rec in self.live_allocs.lock().values() {
            total += rec.bytes;
            let key = format!("{}:{}", rec.site.file(), rec.site.line());
            let e = by_site.entry(key).or_insert((0, 0));
            e.0 += rec.bytes;
            e.1 += 1;
        }
        let mut rows: Vec<(String, usize, usize)> =
            by_site.into_iter().map(|(k, v)| (k, v.0, v.1)).collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1));

        let mut out = format!(
            "GPU allocation ledger: {:.2} GB live across {} sites\n",
            total as f64 / 1e9,
            rows.len()
        );
        let mut shown = 0usize;
        let mut folded_bytes = 0usize;
        let mut folded_sites = 0usize;
        for (site, bytes, count) in rows {
            if shown < top_n && bytes >= min_mb * 1024 * 1024 {
                out.push_str(&format!(
                    "  {:>9.1} MB  x{:<5} {}\n",
                    bytes as f64 / (1024.0 * 1024.0),
                    count,
                    site
                ));
                shown += 1;
            } else {
                folded_bytes += bytes;
                folded_sites += 1;
            }
        }
        if folded_sites > 0 {
            out.push_str(&format!(
                "  {:>9.1} MB  across {} smaller sites\n",
                folded_bytes as f64 / (1024.0 * 1024.0),
                folded_sites
            ));
        }

        // 2026-09-25: Per-file rollup: a file that allocates from many lines,
        // each under the cut above, still shows as one entry here.
        let mut by_file: HashMap<&str, (usize, usize)> = HashMap::new();
        for rec in self.live_allocs.lock().values() {
            let e = by_file.entry(rec.site.file()).or_insert((0, 0));
            e.0 += rec.bytes;
            e.1 += 1;
        }
        let mut frows: Vec<(&str, usize, usize)> =
            by_file.into_iter().map(|(k, v)| (k, v.0, v.1)).collect();
        frows.sort_by(|a, b| b.1.cmp(&a.1));
        out.push_str("  ── by file ──\n");
        for (file, bytes, count) in frows.into_iter().take(top_n) {
            out.push_str(&format!(
                "  {:>9.1} MB  x{:<5} {}\n",
                bytes as f64 / (1024.0 * 1024.0),
                count,
                file
            ));
        }
        out
    }
}

#[cfg(test)]
#[path = "alloc_ledger_tests.rs"]
mod tests;
