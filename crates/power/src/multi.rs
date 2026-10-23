// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Sampling every available meter concurrently (owner decision 1): a [`MeterSet`]
//! holds heterogeneous meters as a tuple (or a `Vec` of one type) — generic, no `dyn` (NDIC) —
//! and a [`Sampler`] tick issues a read on every meter with none in flight and polls every one
//! that has, never waiting on any (FILL, ACNB). A slow meter delays only its own samples.
//!
//! Owner: metrale-power.
//! Invariants: at most one read in flight per meter; every completed sample enters the trace
//! through its source's capability check; [`SamplerStats::idle_ticks`] counts ticks where nothing
//! was ready (the measured idle the FILL rule asks for).

use alloc::vec;
use alloc::vec::Vec;
use core::task::Poll;

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{PowerCapabilities, PowerMeter, PowerSample, SampleTicket};
use metrale_error::ErrorCode;

use crate::Result;
use crate::trace::PowerTrace;

/// 2026-10-22: A fixed set of meters addressed by index.
pub trait MeterSet {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn caps(&self, i: usize) -> &PowerCapabilities;
    fn issue(&mut self, i: usize, at: MonoNs) -> Result<SampleTicket>;
    fn complete(&mut self, i: usize, t: SampleTicket) -> Poll<Result<PowerSample>>;
}

impl<M: PowerMeter> MeterSet for Vec<M> {
    fn len(&self) -> usize {
        <[M]>::len(self)
    }
    fn caps(&self, i: usize) -> &PowerCapabilities {
        self[i].capabilities()
    }
    fn issue(&mut self, i: usize, at: MonoNs) -> Result<SampleTicket> {
        self[i].issue(at)
    }
    fn complete(&mut self, i: usize, t: SampleTicket) -> Poll<Result<PowerSample>> {
        self[i].complete(t)
    }
}

macro_rules! tuple_set {
    ($n:expr; $($t:ident $i:tt),+) => {
        impl<$($t: PowerMeter),+> MeterSet for ($($t,)+) {
            fn len(&self) -> usize {
                $n
            }
            fn caps(&self, i: usize) -> &PowerCapabilities {
                match i {
                    $($i => self.$i.capabilities(),)+
                    _ => panic!("meter index {i} out of {}", $n),
                }
            }
            fn issue(&mut self, i: usize, at: MonoNs) -> Result<SampleTicket> {
                match i {
                    $($i => self.$i.issue(at),)+
                    _ => panic!("meter index {i} out of {}", $n),
                }
            }
            fn complete(&mut self, i: usize, t: SampleTicket) -> Poll<Result<PowerSample>> {
                match i {
                    $($i => self.$i.complete(t),)+
                    _ => panic!("meter index {i} out of {}", $n),
                }
            }
        }
    };
}

tuple_set!(1; A 0);
tuple_set!(2; A 0, B 1);
tuple_set!(3; A 0, B 1, C 2);
tuple_set!(4; A 0, B 1, C 2, D 3);
tuple_set!(5; A 0, B 1, C 2, D 3, E 4);
tuple_set!(6; A 0, B 1, C 2, D 3, E 4, F 5);
tuple_set!(7; A 0, B 1, C 2, D 3, E 4, F 5, G 6);
tuple_set!(8; A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7);

/// 2026-10-22: What the sampler did so far.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SamplerStats {
    pub ticks: u64,
    /// 2026-10-22: Ticks on which no meter had a sample ready.
    pub idle_ticks: u64,
    pub issued: u64,
    pub completed: u64,
    /// 2026-10-22: Reads that failed (counted, and the meter re-issued next tick).
    pub read_errors: u64,
    /// 2026-10-22: Issues a meter refused because its queue was full.
    pub refused: u64,
}

/// 2026-10-22: Drives a [`MeterSet`] into a [`PowerTrace`].
#[derive(Clone, Debug)]
pub struct Sampler {
    in_flight: Vec<Option<SampleTicket>>,
    trace_index: Vec<usize>,
    pub stats: SamplerStats,
}

impl Sampler {
    /// 2026-10-22: Register every meter of `set` as a source of `trace`.
    pub fn new<S: MeterSet>(set: &S, trace: &mut PowerTrace) -> Self {
        let trace_index = (0..set.len())
            .map(|i| trace.add_source(set.caps(i).clone()))
            .collect();
        Self {
            in_flight: vec![None; set.len()],
            trace_index,
            stats: SamplerStats::default(),
        }
    }

    /// 2026-10-22: One tick at `at`: issue on every free meter, poll every busy one. Returns the
    /// number of samples recorded. Never waits. A sample a meter's capabilities reject is an
    /// error (the meter broke its declaration).
    pub fn tick<S: MeterSet>(
        &mut self,
        set: &mut S,
        at: MonoNs,
        trace: &mut PowerTrace,
    ) -> Result<usize> {
        self.stats.ticks += 1;
        let mut done = 0;
        for i in 0..set.len() {
            if self.in_flight[i].is_none() {
                match set.issue(i, at) {
                    Ok(t) => {
                        self.in_flight[i] = Some(t);
                        self.stats.issued += 1;
                    }
                    Err(e) if e.code() == ErrorCode::PowerQueueFull => self.stats.refused += 1,
                    Err(e) => return Err(e),
                }
            }
            let Some(t) = self.in_flight[i] else { continue };
            match set.complete(i, t) {
                Poll::Pending => {}
                Poll::Ready(Ok(s)) => {
                    self.in_flight[i] = None;
                    trace.push(self.trace_index[i], s)?;
                    self.stats.completed += 1;
                    done += 1;
                }
                Poll::Ready(Err(e)) if e.code() == ErrorCode::PowerReadFailed => {
                    self.in_flight[i] = None;
                    self.stats.read_errors += 1;
                }
                Poll::Ready(Err(e)) => return Err(e),
            }
        }
        if done == 0 {
            self.stats.idle_ticks += 1;
        }
        Ok(done)
    }

    /// 2026-10-22: Reads still in flight.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.in_flight.iter().filter(|t| t.is_some()).count()
    }
}
