// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: `METRALE_DEBUG_ALLOC_DIGEST=<path>`: after the build, write one line per live
//! allocation on the ledger, `site<TAB>bytes<TAB>sha256`, sorted. Two boots' digests compare the
//! device bytes of every allocation a load made (weights, copies, scales), independent of their
//! addresses: the load-parity instrument of a loader change (LIFECYCLE-DESIGN.md section 5.4).
//!
//! Owner: metrale-model-engine.
//! Invariants:
//! - Reads device memory only (64 MiB host chunks, never one host buffer per allocation) and
//!   changes nothing the serve uses.
//! - Unset, it does nothing; a backend without a ledger writes nothing and says so.

use std::io::Write as _;

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use sha2::{Digest, Sha256};

/// 2026-10-02: Host chunk the copies go through.
const CHUNK: usize = 64 << 20;

/// 2026-10-02: Write the digest when `METRALE_DEBUG_ALLOC_DIGEST` names a path.
pub(super) fn write_if_requested(gpu: &dyn GpuBackend) -> Result<()> {
    let Some(path) = std::env::var_os("METRALE_DEBUG_ALLOC_DIGEST") else {
        return Ok(());
    };
    let Some(allocs) = gpu.live_allocations() else {
        tracing::warn!("METRALE_DEBUG_ALLOC_DIGEST: this backend keeps no allocation ledger");
        return Ok(());
    };
    let mut buf = vec![0u8; CHUNK];
    let mut lines = Vec::with_capacity(allocs.len());
    for (ptr, bytes, site) in allocs {
        let mut h = Sha256::new();
        let mut off = 0usize;
        while off < bytes {
            let n = CHUNK.min(bytes - off);
            gpu.copy_d2h(DevicePtr(ptr.0 + off as u64), &mut buf[..n])
                .with_context(|| format!("reading {bytes} B at {site}"))?;
            h.update(&buf[..n]);
            off += n;
        }
        let hex: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        lines.push(format!("{site}\t{bytes}\t{hex}"));
    }
    lines.sort();
    let mut f = std::fs::File::create(&path)
        .with_context(|| format!("METRALE_DEBUG_ALLOC_DIGEST {}", path.to_string_lossy()))?;
    for l in &lines {
        writeln!(f, "{l}")?;
    }
    tracing::info!(
        "METRALE_DEBUG_ALLOC_DIGEST: {} allocations hashed into {}",
        lines.len(),
        path.to_string_lossy()
    );
    Ok(())
}
