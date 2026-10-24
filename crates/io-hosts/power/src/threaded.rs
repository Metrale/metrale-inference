// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: [`ThreadedMeter`]: any blocking [`PowerProbe`] as an issue/complete
//! [`PowerMeter`]. The probe lives on its own thread; `issue` hands it a request over a bounded
//! channel and returns, `complete` polls the answer channel. Neither ever waits (ACNB, FILL), so
//! one caller samples many meters concurrently.
//!
//! Owner: metrale-io-host-power.
//! Invariants:
//! - At most `depth` tickets are outstanding (issued, not yet completed); one more is
//!   `PowerQueueFull`.
//! - A ticket completes once; a second `complete`, or a ticket never issued, is
//!   `PowerTicketUnknown`.
//! - Every sample is checked against the probe's capabilities before it is returned.
//! - Dropping the meter closes the request channel and joins the thread (after the read in
//!   progress, if any, returns).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, TrySendError, sync_channel};
use std::task::Poll;
use std::thread::JoinHandle;

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{
    PowerCapabilities, PowerMeter, PowerProbe, PowerSample, SampleTicket,
};
use metrale_error::{ErrorCode, error};

type Answer = (u64, metrale_abstract_io::Result<PowerSample>);

/// 2026-10-22: A probe on its own thread, read through tickets.
pub struct ThreadedMeter {
    caps: PowerCapabilities,
    depth: usize,
    requests: Option<SyncSender<(u64, MonoNs)>>,
    answers: Receiver<Answer>,
    next: u64,
    in_flight: BTreeSet<u64>,
    done: BTreeMap<u64, metrale_abstract_io::Result<PowerSample>>,
    worker: Option<JoinHandle<()>>,
}

impl ThreadedMeter {
    /// 2026-10-22: Move `probe` onto a new thread allowing `depth` outstanding reads (at least 1;
    /// there is no default depth).
    pub fn spawn<P: PowerProbe + Send + 'static>(
        mut probe: P,
        depth: usize,
    ) -> metrale_abstract_io::Result<Self> {
        let caps = probe.capabilities().clone();
        if depth == 0 {
            return Err(error!(
                ErrorCode::InvalidArgument,
                "depth", "a threaded meter needs room for at least one outstanding read"
            ));
        }
        let (req_tx, req_rx) = sync_channel::<(u64, MonoNs)>(depth);
        let (ans_tx, answers) = sync_channel::<Answer>(depth);
        let worker = std::thread::Builder::new()
            .name(format!("power:{}", caps.source().0))
            .spawn(move || {
                for (id, at) in req_rx {
                    if ans_tx.send((id, probe.read(at))).is_err() {
                        break;
                    }
                }
            })
            .map_err(|e| {
                error!(
                    ErrorCode::PowerSourceUnavailable,
                    caps.source().0.clone(),
                    format!("cannot start its reader thread: {e}")
                )
            })?;
        Ok(Self {
            caps,
            depth,
            requests: Some(req_tx),
            answers,
            next: 0,
            in_flight: BTreeSet::new(),
            done: BTreeMap::new(),
            worker: Some(worker),
        })
    }

    fn name(&self) -> String {
        self.caps.source().0.clone()
    }

    /// 2026-10-22: Move every finished answer from the channel into `done`; whether the worker is
    /// gone.
    fn drain(&mut self) -> bool {
        loop {
            match self.answers.try_recv() {
                Ok((id, r)) => {
                    self.in_flight.remove(&id);
                    self.done.insert(id, r);
                }
                Err(TryRecvError::Empty) => return false,
                Err(TryRecvError::Disconnected) => return true,
            }
        }
    }
}

impl PowerMeter for ThreadedMeter {
    fn capabilities(&self) -> &PowerCapabilities {
        &self.caps
    }

    fn issue(&mut self, at: MonoNs) -> metrale_abstract_io::Result<SampleTicket> {
        if self.in_flight.len() + self.done.len() >= self.depth {
            return Err(error!(ErrorCode::PowerQueueFull, self.name(), self.depth));
        }
        let id = self.next;
        let tx = self.requests.as_ref().expect("requests is Some until drop");
        match tx.try_send((id, at)) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                return Err(error!(ErrorCode::PowerQueueFull, self.name(), self.depth));
            }
            Err(TrySendError::Disconnected(_)) => {
                return Err(error!(
                    ErrorCode::PowerReadFailed,
                    self.name(),
                    "its reader thread has ended"
                ));
            }
        }
        self.next += 1;
        self.in_flight.insert(id);
        Ok(SampleTicket(id))
    }

    fn complete(&mut self, ticket: SampleTicket) -> Poll<metrale_abstract_io::Result<PowerSample>> {
        let id = ticket.0;
        if !self.in_flight.contains(&id) && !self.done.contains_key(&id) {
            return Poll::Ready(Err(error!(ErrorCode::PowerTicketUnknown, self.name(), id)));
        }
        let gone = self.drain();
        if let Some(r) = self.done.remove(&id) {
            return Poll::Ready(r.and_then(|s| self.caps.check(&s).map(|()| s)));
        }
        if gone {
            self.in_flight.remove(&id);
            return Poll::Ready(Err(error!(
                ErrorCode::PowerReadFailed,
                self.name(),
                "its reader thread ended before answering"
            )));
        }
        Poll::Pending
    }
}

impl Drop for ThreadedMeter {
    fn drop(&mut self) {
        self.requests = None;
        // 2026-10-22: Unblock a worker parked on a full answer channel.
        while self.answers.try_recv().is_ok() {}
        if let Some(w) = self.worker.take() {
            // 2026-10-22: A panicked probe has nothing left to release; its panic was the
            // reader's, already reported as a disconnected channel to any pending ticket.
            let _ = w.join();
        }
    }
}

#[cfg(test)]
#[path = "threaded_tests.rs"]
mod tests;
