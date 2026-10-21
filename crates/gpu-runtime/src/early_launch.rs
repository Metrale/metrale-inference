// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Early graph launch (`METRALE_GLM_GRAPH_EARLY_LAUNCH`): a captured graph launched on
//! its own idle side stream, its kernel path gated on the compute stream's work by an external
//! event wait that is the graph's first captured node, so the graph's other roots start at launch.
//!
//! Why: a collective NCCL captures over a network transport waits for a host-callback node (the
//! proxy upload, `enqueue.cc` `hostStreamPlanCallback`). Those nodes chain from the graph's start
//! (`strongstream.cc`), and a graph launched behind other work on the compute stream starts only
//! when that work ends. The first host callback of a graph then runs on the critical path: on
//! GB10 the first collective of every K=2 verify graph started p50 392 us (p90 804) after its
//! producer, every later one ~5 us (rank-0 nsys `ctrldevA_p1k`, 220 steps). A one-GPU emulation
//! (one host node before the first of 8 kernels, behind 2 ms of prior work) reproduces it: p50
//! 276-420 us launched behind the work, 2 us launched early on an idle stream.
//!
//! Ordering: the gate is recorded on the compute stream before the launch, so every kernel of the
//! graph runs after the work queued there before it; the compute stream waits for `done`, so work
//! queued after the launch runs after the graph. NCCL keeps its host tasks and kernels in launch
//! order across graphs (`serialEvent`: an external wait honours a not yet executed record node of
//! a graph launched before it; verified on GB10, 0 of 200 out of order, 200 of 200 with the wait
//! removed).
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - A graph begun with [`EarlyLaunch::begin_capture`] is launched only with
//!   [`EarlyLaunch::launch`] of the same instance. (A plain launch stays correct: its gate wait
//!   then sees an older, finished record, and stream order gates it.)
//! - One graph runs on the side stream at a time: the compute stream waits for each before any
//!   later work, so the next launch finds the side stream idle.

use std::sync::OnceLock;

use anyhow::Result;

use crate::gpu::{GpuBackend, GraphHandle};

/// 2026-10-21: The device operations of an early launch (SBIO: the order is tested against a
/// recording double; production is any [`GpuBackend`]).
pub trait LaunchDevice {
    fn begin_capture(&self, stream: u64) -> Result<()>;
    fn abort_capture(&self, stream: u64);
    /// 2026-10-21: Under capture, an external wait node on `event`.
    fn wait_external(&self, stream: u64, event: u64) -> Result<()>;
    fn record(&self, event: u64, stream: u64) -> Result<()>;
    fn wait(&self, stream: u64, event: u64) -> Result<()>;
    fn launch(&self, graph: GraphHandle, stream: u64) -> Result<()>;
}

impl<G: GpuBackend + ?Sized> LaunchDevice for G {
    fn begin_capture(&self, stream: u64) -> Result<()> {
        GpuBackend::begin_capture(self, stream)
    }
    fn abort_capture(&self, stream: u64) {
        self.abort_capture_if_active(stream);
    }
    fn wait_external(&self, stream: u64, event: u64) -> Result<()> {
        self.stream_wait_event_external(stream, event)
    }
    fn record(&self, event: u64, stream: u64) -> Result<()> {
        self.record_event(event, stream)
    }
    fn wait(&self, stream: u64, event: u64) -> Result<()> {
        self.stream_wait_event(stream, event)
    }
    fn launch(&self, graph: GraphHandle, stream: u64) -> Result<()> {
        self.launch_graph(graph, stream)
    }
}

/// 2026-10-21: One early-launched graph family: its side stream and its two events.
#[derive(Debug)]
pub struct EarlyLaunch {
    side: u64,
    gate: u64,
    done: u64,
}

impl EarlyLaunch {
    /// 2026-10-21: A side stream at `compute`'s priority (a graph keeps the priority it had on
    /// the compute stream) and the gate and done events.
    ///
    /// # Errors
    /// A stream or event cannot be created.
    pub fn new(gpu: &dyn GpuBackend, compute: u64) -> Result<Self> {
        Ok(Self {
            side: gpu.create_stream_like(compute)?,
            gate: gpu.create_event()?,
            done: gpu.create_event()?,
        })
    }

    /// 2026-10-21: The instance in `cell`: resolved once, on the first call, from `on` (an
    /// instance at `compute`'s priority when it says so, none otherwise); later calls return it.
    ///
    /// # Errors
    /// `on` fails or the instance cannot be created.
    pub fn shared<'a>(
        cell: &'a OnceLock<Option<Self>>,
        on: impl FnOnce() -> Result<bool>,
        gpu: &dyn GpuBackend,
        compute: u64,
    ) -> Result<Option<&'a Self>> {
        if let Some(e) = cell.get() {
            return Ok(e.as_ref());
        }
        let e = on()?.then(|| Self::new(gpu, compute)).transpose()?;
        Ok(cell.get_or_init(|| e).as_ref())
    }

    /// 2026-10-21: [`Self::begin_capture`] with `early`, a plain capture without it.
    ///
    /// # Errors
    /// As [`Self::begin_capture`] or `GpuBackend::begin_capture`.
    pub fn begin_capture_with(
        early: Option<&Self>,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        match early {
            Some(e) => e.begin_capture(gpu, stream),
            None => gpu.begin_capture(stream),
        }
    }

    /// 2026-10-21: [`Self::launch`] with `early`, a plain launch on `compute` without it.
    ///
    /// # Errors
    /// As [`Self::launch`] or `GpuBackend::launch_graph`.
    pub fn launch_with(
        early: Option<&Self>,
        gpu: &dyn GpuBackend,
        graph: GraphHandle,
        compute: u64,
    ) -> Result<()> {
        match early {
            Some(e) => e.launch(gpu, graph, compute),
            None => gpu.launch_graph(graph, compute),
        }
    }

    /// 2026-10-21: Begin capturing on `stream` a graph for [`Self::launch`]: its first node waits
    /// for the gate's record at launch.
    ///
    /// # Errors
    /// The capture cannot begin or the wait cannot be captured; the capture is then ended.
    pub fn begin_capture<D: LaunchDevice + ?Sized>(&self, dev: &D, stream: u64) -> Result<()> {
        dev.begin_capture(stream)?;
        if let Err(e) = dev.wait_external(stream, self.gate) {
            dev.abort_capture(stream);
            return Err(e.context("early launch: capturing the gate wait"));
        }
        Ok(())
    }

    /// 2026-10-21: Launch `graph` on the side stream, after the work queued on `compute` so far
    /// and before any work queued there later.
    ///
    /// # Errors
    /// A record, launch or wait fails.
    pub fn launch<D: LaunchDevice + ?Sized>(
        &self,
        dev: &D,
        graph: GraphHandle,
        compute: u64,
    ) -> Result<()> {
        dev.record(self.gate, compute)?;
        dev.launch(graph, self.side)?;
        dev.record(self.done, self.side)?;
        dev.wait(compute, self.done)
    }
}

#[cfg(test)]
#[path = "early_launch_tests.rs"]
mod tests;
