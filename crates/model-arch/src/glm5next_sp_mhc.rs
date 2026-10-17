// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: `METRALE_GLM_SP_MHC=1`: sequence-parallel mHC for a TP prefill sub-chunk
//! (`glm5next_layer/steps/forward_sp.rs`). Every rank
//! runs the mixer and the MLP on all `k` rows as before, but the per-row mHC work (`hc_expand`,
//! the pre-norms, the post-mixes, `hc_head_mean`) only on its own shard of rows: rank `r` owns the
//! contiguous rows `[lo_r, hi_r)` ([`shards`]). The two all-reduces of a layer become a
//! reduce-scatter of rows (each rank receives the peers' partials of its own rows, one grouped
//! send/recv, [`reduce_scatter_rows`]) summed inside `glm5next_hc_post_rank_sum`, and the normed
//! rows every rank needs for its mixer / MLP are all-gathered in place ([`all_gather_rows`]).
//!
//! Numerics: each mHC kernel is per row, so an own-row launch gives a row the bits of today's
//! full launch; the reduce is the one-shot all-reduce's FP32 rank-order sum with one BF16
//! rounding (decode's arithmetic), where today's prefill ring sum depends on the element's place
//! in the message: row-invariant, cold == warm, but other bits than today's prefill (RO class,
//! accuracy-gated).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - [`shards`] covers `0..k` once, in order, with sizes differing by at most one row, so every
//!   rank derives the same shards from `k` alone.
//! - The receive workspace is reserved at load ([`reserve`]) for the widest sub-chunk
//!   (`prefill_rows`) and `world - 1` peers; a wider launch is an error, never an allocation.
//! - The exchanges enqueue nothing but `send_to` / `recv_from` inside one `group_start` /
//!   `group_end`, on the caller's stream, and never a zero-byte transfer.

use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, ensure};
use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

/// 2026-10-12: The narrowest sub-chunk that runs sequence-parallel: each shard of a 3-rank split
/// then has at least 64 rows, where the token-tiled pre-norm and the post-mix are efficient
/// (a measured choice to revisit).
pub const SP_MIN_ROWS: usize = 192;

/// 2026-10-12: `METRALE_GLM_SP_MHC`, read once: unset or `0` off, `1` on; another value is an
/// error.
pub fn enabled() -> Result<bool> {
    static V: OnceLock<Result<bool, String>> = OnceLock::new();
    V.get_or_init(|| parse(std::env::var("METRALE_GLM_SP_MHC").ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// 2026-10-12: [`enabled`]'s parse, pure.
pub(crate) fn parse(v: Option<&str>) -> Result<bool, String> {
    match v {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!("METRALE_GLM_SP_MHC={other:?}: expected 0 or 1")),
    }
}

/// 2026-10-12: Whether a sub-chunk of `k` rows at `world` ranks takes the sequence-parallel
/// body: the lever on, two or more ranks, and at least [`SP_MIN_ROWS`] rows.
pub fn applies(on: bool, world: usize, k: usize) -> bool {
    on && world >= 2 && k >= SP_MIN_ROWS
}

/// 2026-10-12: The `(lo, hi)` row shard of every rank for `k` rows: `lo_r = r * b + min(r, k %
/// world)` with `b = k / world`, so the first `k % world` ranks own one row more.
pub fn shards(k: usize, world: usize) -> Vec<(usize, usize)> {
    let (b, extra) = (k / world, k % world);
    (0..world)
        .map(|r| {
            let lo = r * b + r.min(extra);
            (lo, lo + b + usize::from(r < extra))
        })
        .collect()
}

/// 2026-10-12: The widest shard of `k` rows over `world` ranks (rank 0's).
pub fn max_shard(k: usize, world: usize) -> usize {
    k.div_ceil(world)
}

/// 2026-10-12: One point-to-point transfer of an exchange, in bytes from the buffers' bases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Xfer {
    pub peer: usize,
    pub send_off: usize,
    pub send_bytes: usize,
    pub recv_off: usize,
    pub recv_bytes: usize,
}

/// 2026-10-12: The receive slot of `peer` on `rank` (the one-shot all-reduce's layout, which
/// `glm5next_hc_post_rank_sum` reads: peer `r` in slot `r < rank ? r : r - 1`).
pub fn peer_slot(rank: usize, peer: usize) -> Option<usize> {
    match peer.cmp(&rank) {
        std::cmp::Ordering::Less => Some(peer),
        std::cmp::Ordering::Equal => None,
        std::cmp::Ordering::Greater => Some(peer - 1),
    }
}

/// 2026-10-12: The reduce-scatter's transfers on `rank`: to every peer `p` the rows
/// `[lo_p, hi_p)` of the partial (contiguous rows of `row_bytes`), and from every peer its
/// partial of this rank's rows into that peer's slot of `slot_bytes` in the receive workspace,
/// `slot_off` bytes into the slot. 2026-10-14: `shards` may be one row piece of every rank's
/// shard (`glm5next_sp_overlap`), received at the piece's place in the slot (`slot_off`); the
/// whole shards take `slot_off` 0.
pub fn reduce_scatter_plan(
    rank: usize,
    shards: &[(usize, usize)],
    row_bytes: usize,
    slot_bytes: usize,
    slot_off: usize,
) -> Vec<Xfer> {
    let (lo, hi) = shards[rank];
    shards
        .iter()
        .enumerate()
        .filter_map(|(p, &(plo, phi))| {
            let slot = peer_slot(rank, p)?;
            Some(Xfer {
                peer: p,
                send_off: plo * row_bytes,
                send_bytes: (phi - plo) * row_bytes,
                recv_off: slot * slot_bytes + slot_off,
                recv_bytes: (hi - lo) * row_bytes,
            })
        })
        .collect()
}

/// 2026-10-12: The in-place all-gather's transfers on `rank`: this rank's rows to every peer, and
/// every peer's rows into their own place of the same buffer.
pub fn all_gather_plan(rank: usize, shards: &[(usize, usize)], row_bytes: usize) -> Vec<Xfer> {
    let (lo, hi) = shards[rank];
    shards
        .iter()
        .enumerate()
        .filter(|&(p, _)| p != rank)
        .map(|(p, &(plo, phi))| Xfer {
            peer: p,
            send_off: lo * row_bytes,
            send_bytes: (hi - lo) * row_bytes,
            recv_off: plo * row_bytes,
            recv_bytes: (phi - plo) * row_bytes,
        })
        .collect()
}

/// 2026-10-12: Enqueue `plan` as one grouped exchange: `send` and `recv` are the buffers the
/// offsets count from. Errors on a zero-byte transfer (a shard of no rows), which NCCL refuses.
fn exchange(
    comm: &dyn CommBackend,
    plan: &[Xfer],
    send: DevicePtr,
    recv: DevicePtr,
    stream: u64,
) -> Result<()> {
    ensure!(
        plan.iter().all(|x| x.send_bytes > 0 && x.recv_bytes > 0),
        "SP-mHC: an exchange with an empty shard"
    );
    comm.group_start()?;
    for x in plan {
        comm.send_to(send.0 + x.send_off as u64, x.send_bytes, x.peer, stream)?;
        comm.recv_from(recv.0 + x.recv_off as u64, x.recv_bytes, x.peer, stream)?;
    }
    comm.group_end()
}

/// 2026-10-12: The reduce-scatter of a partial at `p` (`k` rows of `row_bytes`): after it,
/// `ws.recv` holds every peer's partial of this rank's shard, slot `peer_slot(rank, peer)`, and
/// the returned `PeerPartials` describes it for `glm5next_hc_post_rank_sum`. 2026-10-14: with
/// `shards` one row piece per rank, `slot_row` is this rank's piece's first row within its
/// shard: the piece lands there in every slot, and the returned partials start at it, so the
/// kernel reads them as it reads a whole shard's.
pub fn reduce_scatter_rows(
    comm: &dyn CommBackend,
    ws: &SpWorkspace,
    p: DevicePtr,
    shards: &[(usize, usize)],
    slot_row: usize,
    row_bytes: usize,
    stream: u64,
) -> Result<metrale_comm::PeerPartials> {
    let (rank, world) = (comm.rank(), comm.world_size());
    fits(ws, world, shards[rank], slot_row, row_bytes)?;
    let slot_off = slot_row * row_bytes;
    let plan = reduce_scatter_plan(rank, shards, row_bytes, ws.slot_bytes, slot_off);
    exchange(comm, &plan, p, ws.recv, stream)?;
    Ok(partials_at(received(ws, world, rank), slot_off))
}

/// 2026-10-12: Refuses a shard `(lo, hi)` that, from `slot_row`, does not fit `ws`'s slots, or
/// a workspace reserved at another world size.
fn fits(
    ws: &SpWorkspace,
    world: usize,
    (lo, hi): (usize, usize),
    slot_row: usize,
    row_bytes: usize,
) -> Result<()> {
    ensure!(
        world == ws.world && (slot_row + hi - lo) * row_bytes <= ws.slot_bytes,
        "SP-mHC: a shard of {} rows from slot row {slot_row} x {row_bytes} B at {world} ranks \
         exceeds the workspace reserved for {} B slots at {} ranks",
        hi - lo,
        ws.slot_bytes,
        ws.world
    );
    Ok(())
}

/// 2026-10-12: The peers' partials of this rank's whole shard as the reduce-scatter leaves them
/// in `ws`.
fn received(ws: &SpWorkspace, world: usize, rank: usize) -> metrale_comm::PeerPartials {
    metrale_comm::PeerPartials {
        recv: ws.recv.0,
        slot_elems: ws.slot_bytes / 2,
        world_size: world,
        rank,
    }
}

/// 2026-10-14: Ring step `step` (`1..world`) of a whole-shard reduce-scatter on `rank`: the send
/// of shard `to = (rank + step) % world` to `to`, and the receive of peer
/// `from = (rank + world - step) % world`'s partial of this rank's shard into `from`'s slot.
/// Steps `1..world` together are [`reduce_scatter_plan`]'s transfers (offsets taken from it), and
/// at step `s` peer `to` receives from this rank, so every rank enqueuing the steps in order
/// pairs every send with its receive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RingStep {
    pub to: usize,
    pub send_off: usize,
    pub send_bytes: usize,
    pub from: usize,
    pub recv_off: usize,
    pub recv_bytes: usize,
}

/// 2026-10-14: [`RingStep`] `step` on `rank` for `shards` (one per rank), `slot_bytes` per slot.
pub fn ring_step_plan(
    rank: usize,
    shards: &[(usize, usize)],
    row_bytes: usize,
    slot_bytes: usize,
    step: usize,
) -> Result<RingStep> {
    let world = shards.len();
    ensure!(
        rank < world && (1..world).contains(&step),
        "SP-mHC: ring step {step} on rank {rank} of {world}: steps are 1..{world}"
    );
    let (to, from) = ((rank + step) % world, (rank + world - step) % world);
    let plan = reduce_scatter_plan(rank, shards, row_bytes, slot_bytes, 0);
    let at = |peer: usize| {
        plan.iter()
            .find(|x| x.peer == peer)
            .context("SP-mHC: ring peer missing from the reduce-scatter plan")
    };
    let (s, r) = (at(to)?, at(from)?);
    Ok(RingStep {
        to,
        send_off: s.send_off,
        send_bytes: s.send_bytes,
        from,
        recv_off: r.recv_off,
        recv_bytes: r.recv_bytes,
    })
}

/// 2026-10-14: Enqueue ring step `step` of the whole-shard reduce-scatter of the partial at `p`
/// on `stream`, as one group of a send and a receive: the step needs only shard
/// `(rank + step) % world` of `p` written, so a producer that finishes the shards in ring order
/// (that shard first, this rank's own last) can start each step behind its shard. Every rank
/// must enqueue steps `1..world` in order. Returns the peers' partials (as
/// [`reduce_scatter_rows`] does) after the last step, `None` before it.
pub fn reduce_scatter_step(
    comm: &dyn CommBackend,
    ws: &SpWorkspace,
    p: DevicePtr,
    shards: &[(usize, usize)],
    step: usize,
    row_bytes: usize,
    stream: u64,
) -> Result<Option<metrale_comm::PeerPartials>> {
    let (rank, world) = (comm.rank(), comm.world_size());
    ensure!(
        shards.len() == world,
        "SP-mHC: {} shards at {world} ranks",
        shards.len()
    );
    fits(ws, world, shards[rank], 0, row_bytes)?;
    let st = ring_step_plan(rank, shards, row_bytes, ws.slot_bytes, step)?;
    ensure!(
        st.send_bytes > 0 && st.recv_bytes > 0,
        "SP-mHC: an exchange with an empty shard"
    );
    comm.group_start()?;
    comm.send_to(p.0 + st.send_off as u64, st.send_bytes, st.to, stream)?;
    comm.recv_from(
        ws.recv.0 + st.recv_off as u64,
        st.recv_bytes,
        st.from,
        stream,
    )?;
    comm.group_end()?;
    Ok((step + 1 == world).then(|| received(ws, world, rank)))
}

/// 2026-10-14: `p` advanced `off` bytes into every slot: the partials of rows that start `off`
/// bytes into the received shard.
pub fn partials_at(p: metrale_comm::PeerPartials, off: usize) -> metrale_comm::PeerPartials {
    metrale_comm::PeerPartials {
        recv: p.recv + off as u64,
        ..p
    }
}

/// 2026-10-12: The in-place all-gather of the rows at `buf` (`row_bytes` each): this rank's
/// shard goes to every peer, every peer's shard lands in its place.
pub fn all_gather_rows(
    comm: &dyn CommBackend,
    buf: DevicePtr,
    shards: &[(usize, usize)],
    row_bytes: usize,
    stream: u64,
) -> Result<()> {
    let plan = all_gather_plan(comm.rank(), shards, row_bytes);
    exchange(comm, &plan, buf, buf, stream)
}

/// 2026-10-12: The receive workspace: `world - 1` slots of `slot_bytes`.
#[derive(Clone, Copy, Debug)]
pub struct SpWorkspace {
    pub recv: DevicePtr,
    pub slot_bytes: usize,
    pub world: usize,
}

fn workspace() -> &'static Mutex<Option<SpWorkspace>> {
    static S: OnceLock<Mutex<Option<SpWorkspace>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(None))
}

/// 2026-10-12: Reserve the workspace for sub-chunks of up to `max_rows` rows of `hidden` BF16 at
/// `world` ranks when the lever is on; a no-op when it is off or the workspace already holds
/// it. Called at load, as each layer builds. 2026-10-14: also the overlap's comm stream and
/// events under `METRALE_GLM_SP_OVERLAP=1` (`glm5next_sp_overlap::reserve`), and the
/// pipeline's buffers under `METRALE_GLM_SP_PIPELINE=1` (`glm5next_sp_pipeline::reserve`).
/// 2026-10-14: and first reads `METRALE_GLM_MHC_POST_PRE_FUSED`, so a bad value or `1` without this lever fails
/// the load, not the first prefill.
pub fn reserve(gpu: &dyn GpuBackend, max_rows: usize, hidden: usize, world: usize) -> Result<()> {
    crate::glm5next_mhc_post_pre_tok::enabled()?;
    if !enabled()? || world < 2 {
        return Ok(());
    }
    crate::glm5next_sp_overlap::reserve(gpu)?;
    let slot_bytes = max_shard(max_rows, world) * hidden * 2;
    crate::glm5next_sp_pipeline::reserve(gpu, max_rows * hidden * 2, slot_bytes, world)?;
    let mut g = workspace().lock().unwrap_or_else(|e| e.into_inner());
    if g.is_some_and(|w| w.world == world && w.slot_bytes >= slot_bytes) {
        return Ok(());
    }
    if let Some(old) = g.take() {
        gpu.free(old.recv)?;
    }
    let recv = gpu.alloc(slot_bytes * (world - 1)).with_context(|| {
        format!(
            "SP-mHC: receive workspace of {} x {slot_bytes} B",
            world - 1
        )
    })?;
    *g = Some(SpWorkspace {
        recv,
        slot_bytes,
        world,
    });
    Ok(())
}

/// 2026-10-12: The workspace reserved at load, or an error when there is none.
pub fn workspace_get() -> Result<SpWorkspace> {
    workspace()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .context("SP-mHC: no receive workspace was reserved at load")
}

#[cfg(test)]
#[path = "glm5next_sp_mhc_tests.rs"]
pub(crate) mod tests;
