// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Two pure [`PowerMeter`]s:
//!
//! - [`ReplayPowerMeter`] plays one source of a recorded trace back, sample by sample, with the
//!   recorded capabilities and timestamps: the same trace through any host gives the same rows
//!   (HPAR). A replayed external meter is this, over its recorded trace.
//! - [`ScriptedPowerMeter`] answers from a script: each read is a result and the number of
//!   `complete` polls it stays pending, so a test drives latency, failure and ordering exactly.
//!
//! Owner: metrale-power.
//! Invariants: a ticket completes once; an unknown or completed ticket is `PowerTicketUnknown`;
//! neither meter ever blocks.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::string::ToString;
use alloc::vec::Vec;
use core::task::Poll;

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{PowerCapabilities, PowerMeter, PowerSample, SampleTicket};
use metrale_error::{ErrorCode, error};

use crate::Result;
use crate::trace::SourceTrace;

fn unknown(caps: &PowerCapabilities, t: SampleTicket) -> metrale_error::Error {
    error!(ErrorCode::PowerTicketUnknown, caps.source().0.clone(), t.0)
}

/// 2026-10-22: One recorded source, played back in order.
#[derive(Clone, Debug)]
pub struct ReplayPowerMeter {
    caps: PowerCapabilities,
    samples: VecDeque<PowerSample>,
    next: u64,
    issued: BTreeMap<u64, PowerSample>,
}

impl ReplayPowerMeter {
    /// 2026-10-22: Replay `src`.
    #[must_use]
    pub fn new(src: &SourceTrace) -> Self {
        Self {
            caps: src.caps.clone(),
            samples: src.samples.iter().cloned().collect(),
            next: 0,
            issued: BTreeMap::new(),
        }
    }

    /// 2026-10-22: Samples not yet issued.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.samples.len()
    }
}

impl PowerMeter for ReplayPowerMeter {
    fn capabilities(&self) -> &PowerCapabilities {
        &self.caps
    }

    /// 2026-10-22: Issues the next recorded sample; `at` is ignored (the recorded stamp is the
    /// virtual clock). An exhausted trace is `PowerSourceUnavailable`.
    fn issue(&mut self, _at: MonoNs) -> Result<SampleTicket> {
        let s = self.samples.pop_front().ok_or_else(|| {
            error!(
                ErrorCode::PowerSourceUnavailable,
                self.caps.source().0.clone(),
                "the replayed trace is exhausted".to_string()
            )
        })?;
        let t = self.next;
        self.next += 1;
        self.issued.insert(t, s);
        Ok(SampleTicket(t))
    }

    fn complete(&mut self, t: SampleTicket) -> Poll<Result<PowerSample>> {
        Poll::Ready(
            self.issued
                .remove(&t.0)
                .ok_or_else(|| unknown(&self.caps, t)),
        )
    }
}

/// 2026-10-22: One scripted read: what it returns and after how many pending polls.
#[derive(Clone, Debug)]
pub struct Scripted {
    pub result: Result<PowerSample>,
    pub pending_polls: u32,
}

/// 2026-10-22: A meter that answers from a script; each issue consumes the next entry and stamps
/// the sample with the issue's `at`.
#[derive(Clone, Debug)]
pub struct ScriptedPowerMeter {
    caps: PowerCapabilities,
    script: VecDeque<Scripted>,
    next: u64,
    issued: BTreeMap<u64, Scripted>,
    max_in_flight: usize,
}

impl ScriptedPowerMeter {
    /// 2026-10-22: A meter over `script` that refuses more than `max_in_flight` open tickets.
    #[must_use]
    pub fn new(caps: PowerCapabilities, script: Vec<Scripted>, max_in_flight: usize) -> Self {
        Self {
            caps,
            script: script.into(),
            next: 0,
            issued: BTreeMap::new(),
            max_in_flight,
        }
    }
}

impl PowerMeter for ScriptedPowerMeter {
    fn capabilities(&self) -> &PowerCapabilities {
        &self.caps
    }

    fn issue(&mut self, at: MonoNs) -> Result<SampleTicket> {
        if self.issued.len() >= self.max_in_flight {
            return Err(error!(
                ErrorCode::PowerQueueFull,
                self.caps.source().0.clone(),
                self.max_in_flight
            ));
        }
        let mut s = self.script.pop_front().ok_or_else(|| {
            error!(
                ErrorCode::PowerSourceUnavailable,
                self.caps.source().0.clone(),
                "the script is exhausted".to_string()
            )
        })?;
        if let Ok(sample) = &mut s.result {
            sample.at = at;
        }
        let t = self.next;
        self.next += 1;
        self.issued.insert(t, s);
        Ok(SampleTicket(t))
    }

    fn complete(&mut self, t: SampleTicket) -> Poll<Result<PowerSample>> {
        let Some(s) = self.issued.get_mut(&t.0) else {
            return Poll::Ready(Err(unknown(&self.caps, t)));
        };
        if s.pending_polls > 0 {
            s.pending_polls -= 1;
            return Poll::Pending;
        }
        let s = self.issued.remove(&t.0).expect("present: checked above");
        Poll::Ready(s.result.and_then(|x| self.caps.check(&x).map(|()| x)))
    }
}
