// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `METRALE_GLM_SP_PIPELINE=1`: pipeline a layer's SP-mHC prefill sub-chunks
//! (`glm5next_sp_mhc`) so each sub-chunk's exchanges run on the comm stream behind another
//! sub-chunk's mixer or MLP. One compute stream (the caller's) and the SP comm stream
//! (`glm5next_sp_overlap::reserve`); the host enqueues the stages of `n >= 2` sub-chunks of a
//! segment in [`order`]: after a prologue of `Head(0) Mixer(0) Head(1)`, step `s` enqueues
//! `Mid(s) Mixer(s+1) Tail(s-1) Head(s+2) Mlp(s)`, and the epilogue `Tail(n-1)`. So the
//! reduce-scatter of a mixer output and the all-gather of a later head run behind the MLP
//! before them, the all-gather after the post-mix behind the next mixer, and the MLP's
//! reduce-scatter behind the next post-mix and mixer. [`program`] lowers that order to the
//! enqueues the driver (`glm5next_layer/steps/forward_sp_pipe.rs`) runs and the tests model.
//!
//! Stages of sub-chunk `s` (the serial body's three sections around its mixer and MLP):
//! `Head` is section 1 (`hc_expand`, the attention pre-norm) and its all-gather of the normed
//! rows; `Mixer` the attention sublayer and the reduce-scatter of its output; `Mid` section 2
//! (the post-mix, the FFN pre-norm) and its all-gather; `Mlp` the MLP and, when its output is
//! a partial, its reduce-scatter; `Tail` section 3 (the post-mix, `hc_head_mean` on the last
//! layer) and, on the last layer, its all-gather of `hidden`.
//!
//! Numerics: the same launches with the same shapes and inputs as the serial SP-mHC body
//! (`METRALE_GLM_SP_OVERLAP`'s pieces included): only their host order across sub-chunks and
//! the buffers of the normed rows and the MLP's received partials move. A whole-shard
//! reduce-scatter moves the bytes its row pieces move to the same place (the piece tests of
//! `glm5next_sp_overlap`), so the output is the serial body's bit for bit.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - The normed rows of sub-chunk `s` live in slot [`normed_slot`]`(s)` of [`NORMED_SLOTS`]:
//!   sub-chunk `s + 2`'s head writes its slot while sub-chunk `s`'s MLP still reads its own.
//! - The mixer's and the MLP's received partials live in two receive workspaces
//!   ([`recv_slot`]): the MLP's reduce-scatter of `s` runs while sub-chunk `s + 1`'s post-mix
//!   reads the mixer's.
//! - One KDA `final_out` and one MLP output suffice: each is rewritten only after the main
//!   stream has waited for the reduce-scatter that read it (the tests prove it on a two-stream
//!   happens-before model, with negative controls).
//! - Every deferred event is waited before it is recorded again, in host order.
//! - The pass ends with the main stream waiting for the comm stream (`Join`), so nothing
//!   overlaps across layers or segments.
//! - Refused at the startup resolve (`glm5next_mhc_levers`) without `METRALE_GLM_SP_MHC=1`, and
//!   under the layer dump (whose taps read buffers by the serial order).

use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use crate::glm5next_sp_mhc as sp;
use crate::glm5next_sp_overlap::On;

/// 2026-10-14: Slots of the normed rows: a head runs two sub-chunks ahead of the MLP.
pub const NORMED_SLOTS: usize = 3;

/// 2026-10-14: The fewest sub-chunks a pipeline runs: one sub-chunk has nothing to overlap.
pub const MIN_SUBS: usize = 2;

/// 2026-10-14: The sub-chunk sizes of `len` rows at `rows` per sub-chunk, in order: what
/// `prefill_segments` runs one `forward_k` each.
pub fn sub_chunks(len: usize, rows: usize) -> Vec<usize> {
    let rows = rows.max(1);
    (0..len.div_ceil(rows))
        .map(|i| rows.min(len - i * rows))
        .collect()
}

/// 2026-10-14: How many leading sub-chunks a pipeline runs, `takes[s]` whether sub-chunk `s`
/// takes the SP body: the leading run of those that do, when it has at least [`MIN_SUBS`]; 0
/// otherwise.
pub fn pipelined(takes: &[bool]) -> usize {
    let n = takes.iter().take_while(|&&t| t).count();
    if n >= MIN_SUBS { n } else { 0 }
}

/// 2026-10-14: A stage of a sub-chunk (see the module header).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    Head,
    Mixer,
    Mid,
    Mlp,
    Tail,
}

/// 2026-10-14: The stages of `n` sub-chunks in host order (see the module header).
pub fn order(n: usize) -> Vec<(Stage, usize)> {
    if n == 0 {
        return Vec::new();
    }
    let mut out = vec![(Stage::Head, 0), (Stage::Mixer, 0)];
    if n > 1 {
        out.push((Stage::Head, 1));
    }
    for s in 0..n {
        out.push((Stage::Mid, s));
        if s + 1 < n {
            out.push((Stage::Mixer, s + 1));
        }
        if s >= 1 {
            out.push((Stage::Tail, s - 1));
        }
        if s + 2 < n {
            out.push((Stage::Head, s + 2));
        }
        out.push((Stage::Mlp, s));
    }
    out.push((Stage::Tail, n - 1));
    out
}

/// 2026-10-14: The pipeline's events: the comm end of each kind of deferred exchange, the
/// main stream's mark before a reduce-scatter, and the pass's join.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Event {
    Head,
    MixerRs,
    Mid,
    MlpRs,
    Tail,
    RsIn,
    Join,
}

/// 2026-10-14: The events reserved at load, one per [`Event`].
pub const EVENTS: usize = 7;

/// 2026-10-14: `ev`'s index among the [`EVENTS`] reserved events.
pub fn event_index(ev: Event) -> usize {
    match ev {
        Event::Head => 0,
        Event::MixerRs => 1,
        Event::Mid => 2,
        Event::MlpRs => 3,
        Event::Tail => 4,
        Event::RsIn => 5,
        Event::Join => 6,
    }
}

/// 2026-10-14: Whose partial a reduce-scatter carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RsKind {
    Mixer,
    Mlp,
}

/// 2026-10-14: One enqueue of the pipeline, in host order. `Section` runs a section of
/// sub-chunk `sub` by `glm5next_sp_overlap::run_section` deferred into `done`'s event (`None`:
/// a section without an all-gather, its computes alone); `Rs` is a whole-shard reduce-scatter
/// on the comm stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prim {
    Record(Event, On),
    Wait(On, Event),
    Section {
        stage: Stage,
        sub: usize,
        done: Option<Event>,
    },
    Mixer(usize),
    Mlp(usize),
    Rs(RsKind, usize),
}

/// 2026-10-14: The enqueues of a layer's `n` sub-chunks in [`order`]: `reduce` when the MLP's
/// output is a partial, `last` on the last layer (its tail all-gathers `hidden`). Every
/// consumer waits for its exchange's event; every reduce-scatter starts after its producer
/// (`RsIn`); the pass ends joined.
pub fn program(n: usize, reduce: bool, last: bool) -> Vec<Prim> {
    let rs = |p: &mut Vec<Prim>, kind, sub, done| {
        p.extend([
            Prim::Record(Event::RsIn, On::Main),
            Prim::Wait(On::Comm, Event::RsIn),
            Prim::Rs(kind, sub),
            Prim::Record(done, On::Comm),
        ]);
    };
    let mut p = Vec::new();
    for (stage, sub) in order(n) {
        match stage {
            Stage::Head => p.push(Prim::Section {
                stage,
                sub,
                done: Some(Event::Head),
            }),
            Stage::Mixer => {
                p.push(Prim::Wait(On::Main, Event::Head));
                p.push(Prim::Mixer(sub));
                rs(&mut p, RsKind::Mixer, sub, Event::MixerRs);
            }
            Stage::Mid => {
                p.push(Prim::Wait(On::Main, Event::MixerRs));
                p.push(Prim::Section {
                    stage,
                    sub,
                    done: Some(Event::Mid),
                });
            }
            Stage::Mlp => {
                p.push(Prim::Wait(On::Main, Event::Mid));
                p.push(Prim::Mlp(sub));
                if reduce {
                    rs(&mut p, RsKind::Mlp, sub, Event::MlpRs);
                }
            }
            Stage::Tail => {
                if reduce {
                    p.push(Prim::Wait(On::Main, Event::MlpRs));
                }
                p.push(Prim::Section {
                    stage,
                    sub,
                    done: last.then_some(Event::Tail),
                });
            }
        }
    }
    if n > 0 {
        p.push(Prim::Record(Event::Join, On::Comm));
        p.push(Prim::Wait(On::Main, Event::Join));
    }
    p
}

/// 2026-10-14: The normed slot of sub-chunk `s`.
pub fn normed_slot(s: usize) -> usize {
    s % NORMED_SLOTS
}

/// 2026-10-14: The receive workspace of a `kind` reduce-scatter: 0 is SP-mHC's own, 1 the
/// pipeline's.
pub fn recv_slot(kind: RsKind) -> usize {
    match kind {
        RsKind::Mixer => 0,
        RsKind::Mlp => 1,
    }
}

/// 2026-10-14: The pipeline's device memory: the normed slots past the first (whose slot 0 is
/// the layer's own normed buffer), the second receive workspace and the events.
#[derive(Clone, Copy, Debug)]
pub struct PipelineWorkspace {
    pub normed: [DevicePtr; NORMED_SLOTS - 1],
    pub normed_bytes: usize,
    pub recv: DevicePtr,
    pub slot_bytes: usize,
    pub world: usize,
    pub events: [u64; EVENTS],
}

impl PipelineWorkspace {
    /// 2026-10-14: The normed rows of sub-chunk `s` holding `bytes`: `base` (the layer's own
    /// buffer) for slot 0. Errors past the reserved size.
    pub fn normed_at(&self, base: DevicePtr, s: usize, bytes: usize) -> Result<DevicePtr> {
        ensure!(
            bytes <= self.normed_bytes,
            "SP pipeline: {bytes} B of normed rows exceed the {} B slots reserved at load",
            self.normed_bytes
        );
        Ok(match normed_slot(s) {
            0 => base,
            i => self.normed[i - 1],
        })
    }

    /// 2026-10-14: The receive workspace of a `kind` reduce-scatter (`own` is SP-mHC's).
    pub fn recv_for(&self, own: &sp::SpWorkspace, kind: RsKind) -> Result<sp::SpWorkspace> {
        if recv_slot(kind) == 0 {
            return Ok(*own);
        }
        ensure!(
            own.world == self.world && own.slot_bytes <= self.slot_bytes,
            "SP pipeline: the receive workspace was reserved for {} B slots at {} ranks",
            self.slot_bytes,
            self.world
        );
        Ok(sp::SpWorkspace {
            recv: self.recv,
            slot_bytes: own.slot_bytes,
            world: own.world,
        })
    }

    /// 2026-10-14: `ev`'s reserved event.
    pub fn event(&self, ev: Event) -> u64 {
        self.events[event_index(ev)]
    }
}

/// 2026-10-14: The bytes [`reserve`] allocates for normed slots of `normed_bytes` and receive
/// slots of `slot_bytes` at `world` ranks.
pub fn extra_bytes(normed_bytes: usize, slot_bytes: usize, world: usize) -> usize {
    (NORMED_SLOTS - 1) * normed_bytes + slot_bytes * (world - 1)
}

fn workspace() -> &'static Mutex<Option<PipelineWorkspace>> {
    static S: OnceLock<Mutex<Option<PipelineWorkspace>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(None))
}

/// 2026-10-14: Reserve the pipeline's buffers once when the lever is on (a no-op otherwise or
/// when they already hold the sizes): normed slots of `normed_bytes` (the widest sub-chunk's
/// rows) and a receive workspace of `world - 1` slots of `slot_bytes`. Called from
/// `glm5next_sp_mhc::reserve` at load, before the KV budget reads the allocator, so the bytes
/// are charged to the load's memory ledger.
pub fn reserve(
    gpu: &dyn GpuBackend,
    normed_bytes: usize,
    slot_bytes: usize,
    world: usize,
    levers: &crate::glm5next_mhc_levers::MhcLevers,
) -> Result<()> {
    if !levers.sp_pipeline || world < 2 {
        return Ok(());
    }
    let mut g = workspace().lock().unwrap_or_else(|e| e.into_inner());
    if g.is_some_and(|w| {
        w.world == world && w.slot_bytes >= slot_bytes && w.normed_bytes >= normed_bytes
    }) {
        return Ok(());
    }
    let events = match g.take() {
        Some(old) => {
            for p in old.normed {
                gpu.free(p)?;
            }
            gpu.free(old.recv)?;
            old.events
        }
        None => {
            let mut events = [0u64; EVENTS];
            for e in &mut events {
                *e = gpu.create_event().context("SP pipeline: event")?;
            }
            events
        }
    };
    let mut normed = [DevicePtr(0); NORMED_SLOTS - 1];
    for p in &mut normed {
        *p = gpu
            .alloc(normed_bytes)
            .with_context(|| format!("SP pipeline: a normed slot of {normed_bytes} B"))?;
    }
    let recv = gpu.alloc(slot_bytes * (world - 1)).with_context(|| {
        format!(
            "SP pipeline: receive workspace of {} x {slot_bytes} B",
            world - 1
        )
    })?;
    tracing::info!(
        bytes = extra_bytes(normed_bytes, slot_bytes, world),
        "METRALE_GLM_SP_PIPELINE=1: reserved the pipeline's normed slots and receive workspace"
    );
    *g = Some(PipelineWorkspace {
        normed,
        normed_bytes,
        recv,
        slot_bytes,
        world,
        events,
    });
    Ok(())
}

/// 2026-10-14: The buffers reserved at load, or an error when there are none.
pub fn workspace_get() -> Result<PipelineWorkspace> {
    workspace()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .context("SP pipeline: no buffers were reserved at load")
}

#[cfg(test)]
#[path = "glm5next_sp_pipeline_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "glm5next_sp_pipeline_hb_tests.rs"]
mod hb_tests;
