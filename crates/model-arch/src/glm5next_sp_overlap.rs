// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `METRALE_GLM_SP_OVERLAP=1`: overlap SP-mHC's exchanges with its mHC launches
//! (`glm5next_sp_mhc`). Each rank's shard of
//! a sub-chunk is cut into [`piece_count`] row pieces, the same on every rank. A layer's three
//! mHC sections (the pre-norm before the mixer, the post-mix and pre-norm between the mixer and
//! the MLP, the post-mix after the MLP) then run per piece on the caller's stream while the
//! reduce-scatter of the next pieces and the all-gather of the finished ones run on a comm
//! stream: [`schedule`] orders the two streams with events alone, never a host sync, and every
//! exchange stays one `group_start` / `group_end` of paired send/recv.
//!
//! Numerics: only the per-row mHC launches are split (each computes a row from that row alone,
//! the bytes of a whole-shard launch at any row count and offset: `mhc_slice.rs`,
//! `glm5next_hc_post_sum.cu`); every transfer moves the same bytes into the same place; the
//! mixer and the MLP launches are unchanged. The output is the serial SP-mHC body's, bit for
//! bit.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Off, or one piece: [`schedule`] is the serial body (exchange, compute, exchange on the
//!   caller's stream), what SP-mHC ran before this lever.
//! - A section ends with the caller's stream waiting for the comm stream's last exchange, and
//!   the comm stream starts after the caller's stream's earlier work, so nothing overlaps
//!   across sections, layers or sub-chunks.
//! - Every rank enqueues the same exchanges in the same host order, all on the comm stream.

use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, ensure};
use metrale_comm::{CommBackend, PeerPartials};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use crate::glm5next_sp_mhc as sp;

/// 2026-10-14: The narrowest row piece: each piece's exchange then moves at least 2 MB per peer
/// at H 4096, past the size where an NCCL send/recv runs near link rate. A choice to measure.
pub const MIN_PIECE_ROWS: usize = 256;

/// 2026-10-14: The most pieces a shard is cut into: with 4, the exposed exchange is a quarter
/// of a whole shard's, and a section enqueues at most 8 grouped exchanges.
pub const MAX_PIECES: usize = 4;

/// 2026-10-14: `METRALE_GLM_SP_OVERLAP`, read once: unset or `0` off, `1` on (which needs
/// `METRALE_GLM_SP_MHC=1`); another value is an error.
pub fn enabled() -> Result<bool> {
    static V: OnceLock<Result<bool, String>> = OnceLock::new();
    V.get_or_init(|| {
        let sp_mhc = sp::enabled().map_err(|e| e.to_string())?;
        parse(
            std::env::var("METRALE_GLM_SP_OVERLAP").ok().as_deref(),
            sp_mhc,
        )
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// 2026-10-14: [`enabled`]'s parse, pure: `sp_mhc` is `METRALE_GLM_SP_MHC`'s value.
pub(crate) fn parse(v: Option<&str>, sp_mhc: bool) -> Result<bool, String> {
    let on = match v {
        None | Some("0") => false,
        Some("1") => true,
        Some(other) => {
            return Err(format!("METRALE_GLM_SP_OVERLAP={other:?}: expected 0 or 1"));
        }
    };
    if on && !sp_mhc {
        return Err("METRALE_GLM_SP_OVERLAP=1 needs METRALE_GLM_SP_MHC=1".into());
    }
    Ok(on)
}

/// 2026-10-14: The row pieces of a `k`-row sub-chunk at `world` ranks: 1 when off, else the
/// narrowest shard over [`MIN_PIECE_ROWS`], between 1 and [`MAX_PIECES`]. Every rank derives
/// it from `k` and `world` alone.
pub fn piece_count(on: bool, k: usize, world: usize) -> usize {
    if !on {
        return 1;
    }
    (k / world / MIN_PIECE_ROWS).clamp(1, MAX_PIECES)
}

/// 2026-10-14: `pieces[j][r]`, the absolute `(lo, hi)` rows of rank `r`'s piece `j`: each shard
/// cut into `m` balanced pieces in order (`glm5next_sp_mhc::shards` of the shard's rows).
pub fn pieces(shards: &[(usize, usize)], m: usize) -> Vec<Vec<(usize, usize)>> {
    let per_rank: Vec<Vec<(usize, usize)>> = shards
        .iter()
        .map(|&(lo, hi)| {
            sp::shards(hi - lo, m)
                .into_iter()
                .map(|(a, b)| (lo + a, lo + b))
                .collect()
        })
        .collect();
    (0..m)
        .map(|j| per_rank.iter().map(|p| p[j]).collect())
        .collect()
}

/// 2026-10-14: A stream of [`schedule`]: the caller's or the comm stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum On {
    Main,
    Comm,
}

/// 2026-10-14: An event of [`schedule`]: the section's start on the caller's stream, the comm
/// stream's end, piece `j` received (its reduce-scatter done) and computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ev {
    In,
    Out,
    Received(usize),
    Computed(usize),
}

/// 2026-10-14: The events a section uses: `In`, `Out`, and two per piece.
pub const EVENTS: usize = 2 + 2 * MAX_PIECES;

/// 2026-10-14: `ev`'s index among the [`EVENTS`] reserved events.
pub fn event_index(ev: Ev) -> Result<usize> {
    let i = match ev {
        Ev::In => 0,
        Ev::Out => 1,
        Ev::Received(j) if j < MAX_PIECES => 2 + j,
        Ev::Computed(j) if j < MAX_PIECES => 2 + MAX_PIECES + j,
        Ev::Received(j) | Ev::Computed(j) => {
            anyhow::bail!("SP overlap: piece {j} past the {MAX_PIECES} reserved")
        }
    };
    Ok(i)
}

/// 2026-10-14: One enqueue of a section, in host order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Record(Ev, On),
    Wait(On, Ev),
    ReduceScatter(usize, On),
    Compute(usize),
    AllGather(usize, On),
}

/// 2026-10-14: The enqueues of a section of `m` pieces with a reduce-scatter before the compute
/// (`rs`) and an all-gather after it (`ag`). Serial (`m == 1`, or no exchange): the
/// reduce-scatter, the compute and the all-gather on the caller's stream. Pipelined: the comm
/// stream waits for the caller's earlier work, reduce-scatters every piece and marks each
/// received; the caller's stream computes each piece once it is received and marks it
/// computed; the comm stream all-gathers each piece once computed; the caller's stream waits
/// for the comm stream's end. Every reduce-scatter precedes every all-gather on the comm
/// stream, so an all-gather into a buffer the reduce-scatter reads (DSA's output over its
/// normed input) never overtakes it.
pub fn schedule(m: usize, rs: bool, ag: bool) -> Vec<Op> {
    if m <= 1 || !(rs || ag) {
        let mut ops = Vec::with_capacity(3 * m);
        for j in 0..m {
            if rs {
                ops.push(Op::ReduceScatter(j, On::Main));
            }
            ops.push(Op::Compute(j));
            if ag {
                ops.push(Op::AllGather(j, On::Main));
            }
        }
        return ops;
    }
    let mut ops = vec![Op::Record(Ev::In, On::Main), Op::Wait(On::Comm, Ev::In)];
    if rs {
        for j in 0..m {
            ops.push(Op::ReduceScatter(j, On::Comm));
            ops.push(Op::Record(Ev::Received(j), On::Comm));
        }
    }
    for j in 0..m {
        if rs {
            ops.push(Op::Wait(On::Main, Ev::Received(j)));
        }
        ops.push(Op::Compute(j));
        if ag {
            ops.push(Op::Record(Ev::Computed(j), On::Main));
        }
    }
    if ag {
        for j in 0..m {
            ops.push(Op::Wait(On::Comm, Ev::Computed(j)));
            ops.push(Op::AllGather(j, On::Comm));
        }
    }
    ops.push(Op::Record(Ev::Out, On::Comm));
    ops.push(Op::Wait(On::Main, Ev::Out));
    ops
}

/// 2026-10-14: The comm stream (at the greatest priority, so its exchange kernels are not
/// queued behind the high-priority caller stream's mHC waves) and the [`EVENTS`] events.
#[derive(Clone, Copy, Debug)]
pub struct OverlapStreams {
    pub comm: u64,
    pub events: [u64; EVENTS],
}

fn streams() -> &'static Mutex<Option<OverlapStreams>> {
    static S: OnceLock<Mutex<Option<OverlapStreams>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(None))
}

/// 2026-10-14: Create the comm stream and events once when the lever is on; a no-op otherwise.
/// Called from `glm5next_sp_mhc::reserve` at load.
pub fn reserve(gpu: &dyn GpuBackend) -> Result<()> {
    if !enabled()? {
        return Ok(());
    }
    let mut g = streams().lock().unwrap_or_else(|e| e.into_inner());
    if g.is_some() {
        return Ok(());
    }
    let comm = gpu
        .create_stream_greatest_priority()
        .context("SP overlap: comm stream")?;
    let mut events = [0u64; EVENTS];
    for e in &mut events {
        *e = gpu.create_event().context("SP overlap: event")?;
    }
    *g = Some(OverlapStreams { comm, events });
    Ok(())
}

/// 2026-10-14: The comm stream and events reserved at load, or an error when there are none.
pub fn streams_get() -> Result<OverlapStreams> {
    streams()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .context("SP overlap: no comm stream was reserved at load")
}

/// 2026-10-14: One mHC section of a layer: its row pieces (`pieces[j][rank]`), the partial
/// reduce-scattered before the compute and the buffer all-gathered after it (each with the
/// profile bucket of its exchange), and the row width.
pub struct Section<'a> {
    pub pieces: &'a [Vec<(usize, usize)>],
    pub rs: Option<(DevicePtr, Option<usize>)>,
    pub ag: Option<(DevicePtr, Option<usize>)>,
    pub row_bytes: usize,
}

/// 2026-10-14: Enqueue `sec` by [`schedule`]: `compute(lo, hi, partials)` enqueues the mHC
/// launches of this rank's rows `[lo, hi)` on `main`, with the peers' partials of those rows
/// when the section reduce-scatters. `overlap` is required when the section pipelines.
pub fn run_section(
    gpu: &dyn GpuBackend,
    comm: &dyn CommBackend,
    ws: &sp::SpWorkspace,
    overlap: Option<&OverlapStreams>,
    main: u64,
    sec: &Section<'_>,
    compute: &mut dyn FnMut(usize, usize, Option<&PeerPartials>) -> Result<()>,
) -> Result<()> {
    use crate::glm5next_layer::profile;
    let rank = comm.rank();
    let m = sec.pieces.len();
    ensure!(m >= 1, "SP overlap: a section of no pieces");
    let shard_lo = sec.pieces[0][rank].0;
    let on = |s: On| -> Result<u64> {
        match s {
            On::Main => Ok(main),
            On::Comm => overlap
                .map(|o| o.comm)
                .context("SP overlap: a pipelined section without a comm stream"),
        }
    };
    let event = |ev: Ev| -> Result<u64> {
        let o = overlap.context("SP overlap: a pipelined section without events")?;
        Ok(o.events[event_index(ev)?])
    };
    let mut partials: Vec<Option<PeerPartials>> = vec![None; m];
    for op in schedule(m, sec.rs.is_some(), sec.ag.is_some()) {
        match op {
            Op::Record(ev, s) => gpu.record_event(event(ev)?, on(s)?)?,
            Op::Wait(s, ev) => gpu.stream_wait_event(on(s)?, event(ev)?)?,
            Op::ReduceScatter(j, s) => {
                let (src, bucket) = sec.rs.context("SP overlap: no partial to reduce")?;
                let (stream, lo) = (on(s)?, sec.pieces[j][rank].0);
                let t = bucket.and_then(|_| profile::start_hot());
                partials[j] = Some(sp::reduce_scatter_rows(
                    comm,
                    ws,
                    src,
                    &sec.pieces[j],
                    lo - shard_lo,
                    sec.row_bytes,
                    stream,
                )?);
                if let Some(b) = bucket {
                    profile::end(b, t, gpu, stream);
                }
            }
            Op::Compute(j) => {
                let (lo, hi) = sec.pieces[j][rank];
                compute(lo, hi, partials[j].as_ref())?;
            }
            Op::AllGather(j, s) => {
                let (buf, bucket) = sec.ag.context("SP overlap: no buffer to gather")?;
                let stream = on(s)?;
                let t = bucket.and_then(|_| profile::start_hot());
                sp::all_gather_rows(comm, buf, &sec.pieces[j], sec.row_bytes, stream)?;
                if let Some(b) = bucket {
                    profile::end(b, t, gpu, stream);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "glm5next_sp_overlap_tests.rs"]
mod tests;
