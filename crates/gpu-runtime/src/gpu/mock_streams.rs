// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: The mock backend's cross-stream log: event records, stream waits and event
//! destroys in call order, and an opt-in switch that makes `create_stream` and `create_event`
//! hand out distinct ids, so a test can tell a side stream from the step's.
//!
//! Owner: gpu-runtime (mock backend).
//! Invariants: without the switch, both creators return 0, as the trait's defaults do.

use super::MockGpuBackend;

/// 2026-10-04: One cross-stream call the mock saw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamOp {
    /// 2026-10-04: `record_event(event, stream)`.
    Record { event: u64, stream: u64 },
    /// 2026-10-04: `stream_wait_event(stream, event)`.
    Wait { stream: u64, event: u64 },
    /// 2026-10-04: `destroy_event(event)`.
    Destroy(u64),
}

impl MockGpuBackend {
    /// 2026-10-04: From now on, `create_stream` and `create_event` return distinct non-zero ids.
    pub fn set_distinct_streams(&self) {
        self.distinct.lock().get_or_insert(100);
    }

    /// 2026-10-04: The event records, stream waits and event destroys so far.
    pub fn stream_ops(&self) -> Vec<StreamOp> {
        self.stream_ops.lock().clone()
    }

    pub(super) fn next_distinct(&self) -> u64 {
        let mut d = self.distinct.lock();
        match d.as_mut() {
            Some(n) => {
                *n += 1;
                *n
            }
            None => 0,
        }
    }
}
