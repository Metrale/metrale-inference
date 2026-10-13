// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: The PLE n-gram table held whole in page-locked host memory.
//!
//! The row cache (`NgramRowCache`) faults a missing row off NVMe: on the reference H100
//! box about 110 µs a row, 5-6 s of every 4,096-token prefill chunk, a quarter of a
//! 32k-token cold TTFT. With the lever set, the loader reads every shard once, at boot,
//! into one page-locked buffer, and the gather kernel reads rows in place through its
//! mapped device address: a fully resident table (`NgramTable::Bf16`), the slot being
//! the row id. Placement only: the bytes are the checkpoint's. It costs the table's
//! size in host RAM (102.4 GB for Qwen3.8-Flash-Next), so it is opt-in with
//! `METRALE_PLE_HOST_RESIDENT=1`.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - shard `i` occupies bytes `[i * shard_bytes, (i + 1) * shard_bytes)`, so global row
//!   `r` is at `r * row_bytes`: the address the resident gather reads for slot `r`.
//! - the row count fits the u32 slot ids the resident gather uploads.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_layers::weight_map::DenseWeight;
use metrale_model_weights::weights::DeferredTensor;

/// 2026-10-13: The lever that holds the whole n-gram table in host memory.
pub(super) const PLE_HOST_LEVER: &str = "METRALE_PLE_HOST_RESIDENT";

/// 2026-10-13: Whether the lever is set (`1`).
pub(super) fn lever_set() -> bool {
    std::env::var(PLE_HOST_LEVER).as_deref() == Ok("1")
}

/// 2026-10-13: Read `shards` back to back into `dst`, on up to `threads` threads. Every
/// shard must have the same size and `dst` must hold exactly all of them.
pub(super) fn read_shards(shards: &[DeferredTensor], dst: &mut [u8], threads: usize) -> Result<()> {
    let shard_bytes = shards
        .first()
        .map(DeferredTensor::byte_size)
        .context("PLE host table: no shards")?;
    ensure!(shard_bytes > 0, "PLE host table: shard 0 is empty");
    ensure!(
        shards.iter().all(|s| s.byte_size() == shard_bytes),
        "PLE host table: shards differ in size; the table maps row r to r * row_bytes"
    );
    ensure!(
        dst.len() == shard_bytes * shards.len(),
        "PLE host table: buffer is {} B but {} shards of {shard_bytes} B need {}",
        dst.len(),
        shards.len(),
        shard_bytes * shards.len()
    );
    ensure!(threads > 0, "PLE host table: zero reader threads");
    let work = Mutex::new(dst.chunks_mut(shard_bytes).zip(shards).enumerate());
    let failed = AtomicBool::new(false);
    std::thread::scope(|s| {
        let readers: Vec<_> = (0..threads.min(shards.len()))
            .map(|_| {
                s.spawn(|| -> Result<()> {
                    while !failed.load(Ordering::Relaxed) {
                        let next = work
                            .lock()
                            .map_err(|_| anyhow::anyhow!("PLE host table: queue poisoned"))?
                            .next();
                        let Some((i, (dst, shard))) = next else {
                            return Ok(());
                        };
                        if let Err(e) = shard.read_into(dst) {
                            failed.store(true, Ordering::Relaxed);
                            return Err(e.context(format!("PLE host table: shard {i}")));
                        }
                    }
                    Ok(())
                })
            })
            .collect();
        readers.into_iter().try_for_each(|r| {
            r.join()
                .map_err(|_| anyhow::anyhow!("PLE host table: a reader panicked"))?
        })
    })
}

/// 2026-10-13: Read the table into page-locked host memory and return its mapped device
/// address. The buffer lives as long as the process, like every other weight.
pub(super) fn load(
    shards: &[DeferredTensor],
    row_bytes: usize,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    ensure!(row_bytes > 0, "PLE host table: zero-byte rows");
    let bytes: usize = shards.iter().map(DeferredTensor::byte_size).sum();
    let rows = (bytes / row_bytes) as u64;
    ensure!(
        rows <= u64::from(u32::MAX) + 1,
        "PLE host table: {rows} rows exceed the u32 slot ids the resident gather uploads"
    );
    let threads = std::thread::available_parallelism()
        .context("PLE host table: host parallelism")?
        .get();
    let host = gpu.alloc_host_pinned(bytes)?;
    // 2026-10-13: SAFETY: `host` is a fresh, zeroed `alloc_host_pinned` region of exactly
    // `bytes` bytes, and nothing else holds it until `host_ptr_to_device` below.
    let dst = unsafe { std::slice::from_raw_parts_mut(host, bytes) };
    let t0 = std::time::Instant::now();
    let mapped = read_shards(shards, dst, threads).and_then(|()| {
        gpu.host_ptr_to_device(host).with_context(|| {
            format!("this backend cannot map host memory for a kernel; unset {PLE_HOST_LEVER}")
        })
    });
    match mapped {
        Ok(weight) => {
            let secs = t0.elapsed().as_secs_f64();
            tracing::info!(
                "qwen4_exp PLE: the n-gram table ({rows} rows, {:.1} GB BF16) is resident in \
                 page-locked host memory ({PLE_HOST_LEVER}=1), read in {secs:.0} s \
                 ({:.2} GB/s); its bytes are the checkpoint's.",
                bytes as f64 / 1e9,
                bytes as f64 / 1e9 / secs.max(1e-9)
            );
            Ok(DenseWeight { weight })
        }
        Err(e) => {
            gpu.free_host_pinned(host, bytes)?;
            Err(e.context("qwen4_exp PLE: host-resident n-gram table"))
        }
    }
}

#[cfg(test)]
#[path = "ple_host_tests.rs"]
mod tests;
