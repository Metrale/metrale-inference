// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: On-device step timing for the paired in-serve A/B instrument: CUDA event pairs
//! recorded around one forward on its stream, read back when that forward has completed.
//!
//! Owner: gpu-runtime (telemetry).
//! Invariants:
//! - `elapsed_us` never blocks: an unfinished pair reads `Ok(None)`.
//! - Every event the timer creates is destroyed once, with the timer (`CudaEvents`'s `Drop`).

use anyhow::Result;

/// 2026-10-23: `slots` timed event pairs; slot `i` times one span at a time.
pub struct StepTimer {
    #[cfg(feature = "cuda")]
    events: super::cuda_events::CudaEvents,
    slots: usize,
}

impl StepTimer {
    /// 2026-10-23: `slots` pairs in the current CUDA context; an error without CUDA.
    pub fn new(slots: usize) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            Ok(Self {
                events: super::cuda_events::CudaEvents::create(slots)?,
                slots,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = slots;
            anyhow::bail!("StepTimer needs the cuda feature")
        }
    }

    /// 2026-10-23: Slots the timer holds.
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// 2026-10-23: Record slot `slot`'s start on `stream`.
    pub fn start(&self, slot: usize, stream: u64) -> Result<()> {
        self.record(slot, false, stream)
    }

    /// 2026-10-23: Record slot `slot`'s end on `stream`.
    pub fn end(&self, slot: usize, stream: u64) -> Result<()> {
        self.record(slot, true, stream)
    }

    fn record(&self, slot: usize, end: bool, stream: u64) -> Result<()> {
        anyhow::ensure!(
            slot < self.slots,
            "StepTimer: slot {slot} of {}",
            self.slots
        );
        #[cfg(feature = "cuda")]
        {
            use metrale_telemetry::kernel::{Edge, EventOps};
            let edge = if end { Edge::End } else { Edge::Start };
            anyhow::ensure!(
                self.events.record(slot, edge, stream),
                "cuEventRecord failed"
            );
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (end, stream);
            anyhow::bail!("StepTimer needs the cuda feature")
        }
    }

    /// 2026-10-23: Slot `slot`'s span in microseconds once its end completed, `None` before.
    pub fn elapsed_us(&self, slot: usize) -> Result<Option<f64>> {
        anyhow::ensure!(
            slot < self.slots,
            "StepTimer: slot {slot} of {}",
            self.slots
        );
        #[cfg(feature = "cuda")]
        {
            use metrale_telemetry::kernel::{EventOps, SpanPoll};
            match self.events.poll(slot) {
                SpanPoll::Ready(ns) => Ok(Some(ns as f64 / 1e3)),
                SpanPoll::Pending => Ok(None),
                SpanPoll::Failed => anyhow::bail!("StepTimer: slot {slot}'s events failed"),
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            anyhow::bail!("StepTimer needs the cuda feature")
        }
    }
}
