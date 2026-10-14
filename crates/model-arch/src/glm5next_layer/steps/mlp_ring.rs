// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `METRALE_GLM_MOE_COMBINE_RING=1`: the SP-mHC MLP section's reduce-scatter as
//! ring steps under the MoE combine (`forward_rows_sp`). The combine runs one launch per row
//! shard in ring order, [`order`], recording an event after each; the comm stream runs ring step
//! `s` (`glm5next_sp_mhc::reduce_scatter_step`) once shard `(rank + s) % world` is combined, so
//! every exchange but the last overlaps the later shards' combines. The post-sum then reads the
//! partials the last step returns, whole, on the caller's stream.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Bytes: each shard's combine writes the one launch's rows (`topk_combine::CombineRing`) and
//!   the steps move the bytes `reduce_scatter_rows` moves into the same slots, so the post-sum
//!   reads the serial body's partials.
//! - Every rank enqueues steps `1..world` in order on the one comm stream
//!   (`glm5next_sp_overlap::streams_get`, reserved under `METRALE_GLM_SP_OVERLAP=1`); step `s`
//!   waits for its shard's event only, and the caller's stream waits for the last step before
//!   the post-sum reads the partials.
//! - Off, absent a reduce, or on a dense layer: no plan, the serial section.

use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};
use metrale_comm::{CommBackend, PeerPartials};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use crate::glm5next_layer::profile;
use crate::glm5next_mlp::forward::CombineRing;
use crate::glm5next_sp_mhc as sp;

/// 2026-10-14: The shard of each combine launch at `rank` of `world`: ring step `s`'s shard
/// `(rank + s) % world` for `s = 1..world`, then this rank's own, which never leaves it.
pub fn order(rank: usize, world: usize) -> Vec<usize> {
    (1..=world).map(|s| (rank + s) % world).collect()
}

/// 2026-10-14: One MLP section's ring: the shards in launch order, an event per shard and the
/// comm stream's end.
pub struct MlpRing {
    shards: Vec<(usize, usize)>,
    events: Vec<u64>,
    done: u64,
    comm_stream: u64,
}

/// 2026-10-14: `world` shard events and an end event, created on first use and shared by every
/// layer (each layer's waits are enqueued before the next layer records them again).
fn events(gpu: &dyn GpuBackend, world: usize) -> Result<Vec<u64>> {
    static E: OnceLock<Mutex<Vec<u64>>> = OnceLock::new();
    let mut e = E
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    while e.len() < world + 1 {
        e.push(gpu.create_event().context("MoE combine ring: event")?);
    }
    Ok(e[..=world].to_vec())
}

impl MlpRing {
    /// 2026-10-14: The ring of a section whose `k` rows are sharded as `shards` (rank order), or
    /// `None` when `on` is off or the section reduces nothing.
    pub fn plan(
        on: bool,
        reduce: bool,
        gpu: &dyn GpuBackend,
        rank: usize,
        shards: &[(usize, usize)],
    ) -> Result<Option<Self>> {
        if !(on && reduce) {
            return Ok(None);
        }
        let world = shards.len();
        let comm_stream = crate::glm5next_sp_overlap::streams_get()
            .context("METRALE_GLM_MOE_COMBINE_RING=1 needs METRALE_GLM_SP_OVERLAP=1")?
            .comm;
        let mut events = events(gpu, world)?;
        let done = events.pop().context("MoE combine ring: no end event")?;
        // 2026-10-14: The serve log's proof that the lever reached a plan (lever_readback).
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::info!(
                "GLM MoE: combine ring planned at {world} ranks (METRALE_GLM_MOE_COMBINE_RING=1)"
            )
        });
        Ok(Some(Self {
            shards: order(rank, world).iter().map(|&r| shards[r]).collect(),
            events,
            done,
            comm_stream,
        }))
    }

    /// 2026-10-14: The combine's launches (`forward_moe_pieces`).
    pub fn combine(&self) -> CombineRing<'_> {
        CombineRing {
            shards: &self.shards,
            events: &self.events,
        }
    }

    /// 2026-10-14: Enqueue the ring steps of the partial at `partial` (rank-ordered `shards`)
    /// on the comm stream behind the combine's events, and make `main` wait for the last; the
    /// peers' partials of this rank's shard.
    #[allow(clippy::too_many_arguments)]
    pub fn exchange(
        &self,
        gpu: &dyn GpuBackend,
        comm: &dyn CommBackend,
        ws: &sp::SpWorkspace,
        partial: DevicePtr,
        shards: &[(usize, usize)],
        row_bytes: usize,
        main: u64,
    ) -> Result<PeerPartials> {
        let t = profile::start_hot();
        let mut got = None;
        for s in 1..shards.len() {
            gpu.stream_wait_event(self.comm_stream, self.events[s - 1])?;
            got =
                sp::reduce_scatter_step(comm, ws, partial, shards, s, row_bytes, self.comm_stream)?;
        }
        profile::end(profile::REDUCE_MLP, t, gpu, self.comm_stream);
        gpu.record_event(self.done, self.comm_stream)?;
        gpu.stream_wait_event(main, self.done)?;
        got.context("MoE combine ring: the last step returned no partials")
    }
}

#[cfg(test)]
#[path = "mlp_ring_tests.rs"]
mod tests;
