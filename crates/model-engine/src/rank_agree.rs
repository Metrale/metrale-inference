// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Startup check that every rank holds the same values for the env-read scalars
//! that shape the collective schedule.
//!
//! Such a scalar is read independently on each rank. For example,
//! [`metrale_model_arch::glm5next_layer::prefill_rows`] (`METRALE_GLM_PREFILL_ROWS`) sets the row
//! count of each prefill sub-chunk, and each sub-chunk issues its own `reduce_partial`, so a skew
//! between ranks is a hang or a reduce over the wrong extent rather than a perf difference.
//! Rank 0 broadcasts its values, every rank compares them with its own, and a mismatch fails
//! model construction with an error naming each disagreeing entry. The caller,
//! `TransformerModel::new`, chooses the list.
//!
//! 2026-10-10: The check is also the ranks' first collective after loading their weights, so it
//! is a two-way handshake. Rank 0's values go out on a rendezvous broadcast
//! (`CommBackend::broadcast_rendezvous`), which waits out the spread of load times; then every
//! rank's verdict reaches every rank ([`gather_u32_via_broadcast`]), so a rank that disagrees, or
//! died at the rendezvous, fails every rank's start. Before, rank 0 learnt nothing from its own
//! broadcast: on a three-rank serve whose workers timed out at it (30 s, rank 0 arriving 24 s
//! late), rank 0 went on to serve and hung on the first request.
//!
//! Owner: model-engine.
//! Invariants:
//! - Every rank returns the same verdict: `Ok` exactly when every rank's values equal rank 0's.

use anyhow::{Result, bail};
use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

/// 2026-09-25: Collect one `u32` from every rank of `comm` with `world` rooted
/// broadcasts (root `r` contributes element `r`). Every rank returns the same vector,
/// so any pure function of it is a rank-agreed decision; `ep_min_u32` is its minimum.
///
/// `buf` is a rank-local 4-byte device buffer the broadcasts go through.
/// 2026-10-10: Moved here from the snapshot-restore agreement (`snap_agree`), which still uses
/// it, for the startup handshake below.
pub(crate) fn gather_u32_via_broadcast(
    gpu: &dyn GpuBackend,
    comm: &dyn CommBackend,
    buf: DevicePtr,
    world: usize,
    val: u32,
) -> Result<Vec<u32>> {
    let stream = gpu.default_stream();
    let mut out = Vec::with_capacity(world);
    for root in 0..world {
        let v = if comm.rank() == root {
            gpu.copy_h2d(&val.to_le_bytes(), buf)?;
            comm.broadcast(buf.0, 4, root)?;
            val
        } else {
            comm.broadcast(buf.0, 4, root)?;
            gpu.synchronize(stream)?;
            let mut bytes = [0u8; 4];
            gpu.copy_d2h(buf, &mut bytes)?;
            u32::from_le_bytes(bytes)
        };
        out.push(v);
    }
    Ok(out)
}

/// 2026-09-25: The entries of `items` whose value differs from rank 0's (`root`), as messages
/// naming the entry and both values.
fn mismatches(items: &[(&str, u64)], root: &[u64], rank: usize) -> Vec<String> {
    items
        .iter()
        .zip(root)
        .filter(|((_, mine), theirs)| mine != *theirs)
        .map(|((name, mine), theirs)| {
            format!("{name}: rank {rank} has {mine}, rank 0 has {theirs}")
        })
        .collect()
}

/// 2026-10-10: The ranks whose verdict is "disagrees" (0) in a gathered verdict vector.
fn disagreeing_ranks(verdicts: &[u32]) -> Vec<usize> {
    (0..verdicts.len()).filter(|&r| verdicts[r] == 0).collect()
}

/// 2026-09-25: Broadcast rank 0's `items` and bail if this rank's own values differ.
/// 2026-10-10: Then gather every rank's verdict and bail on every rank if any rank's differ.
///
/// `items` is `(name, value)`; the name appears only in the error message. A single-rank run or
/// an empty list returns `Ok(())` without communicating.
///
/// This is a collective: every rank must call it with the same `items.len()` at the same point.
/// The one caller is `TransformerModel::new`, which every rank runs.
pub(crate) fn assert_ranks_agree(
    gpu: &dyn GpuBackend,
    comm: &dyn CommBackend,
    items: &[(&str, u64)],
) -> Result<()> {
    if comm.world_size() < 2 || items.is_empty() {
        return Ok(());
    }
    let bytes = items.len() * 8;
    let buf = gpu.alloc(bytes)?;

    let gathered = (|| -> Result<(Vec<u64>, Vec<u32>)> {
        let mut host: Vec<u8> = items.iter().flat_map(|(_, v)| v.to_le_bytes()).collect();
        gpu.copy_h2d(&host, buf)?;
        comm.broadcast_rendezvous(buf.0, bytes, 0)?;
        gpu.synchronize(gpu.default_stream())?;
        gpu.copy_d2h(buf, &mut host)?;
        let root: Vec<u64> = host
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().expect("chunks_exact(8)")))
            .collect();
        let agrees = mismatches(items, &root, comm.rank()).is_empty();
        let verdicts =
            gather_u32_via_broadcast(gpu, comm, buf, comm.world_size(), u32::from(agrees))?;
        Ok((root, verdicts))
    })();

    // 2026-09-25: The scratch is freed before `gathered?`, so a failed broadcast does not leak
    // it; a failed free is logged and does not fail the check.
    if let Err(e) = gpu.free(buf) {
        tracing::warn!("rank-agreement scratch free failed (non-fatal): {e}");
    }

    let (root, verdicts) = gathered?;
    let rank = comm.rank();
    let bad = mismatches(items, &root, rank);
    if !bad.is_empty() {
        bail!(
            "ranks disagree on collective-shaping config — this is a hang or a wrong-extent \
             reduce, not a perf skew. Set the same value on EVERY rank: {}",
            bad.join("; ")
        );
    }
    let others = disagreeing_ranks(&verdicts);
    if !others.is_empty() {
        bail!(
            "rank(s) {others:?} disagree with rank 0 on collective-shaping config (their logs \
             name the entries); every rank stops rather than run a mismatched collective \
             schedule"
        );
    }
    tracing::info!(
        "rank-agreement OK on {} collective-shaping scalar(s) across {} ranks: {}",
        items.len(),
        verdicts.len(),
        items
            .iter()
            .map(|(n, v)| format!("{n}={v}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    Ok(())
}

#[cfg(test)]
#[path = "rank_agree_tests.rs"]
mod tests;
