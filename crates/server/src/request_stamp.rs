// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The identity and receive time of one `/v1/*` request (investor-mvp #49).
//! The observability middleware (`main_modules/trace_layer.rs`) builds it before any other
//! layer runs and stores it in the request extensions; inference handlers take it as a
//! required extractor, and the per-request usage record (`request_accounting.rs`) is
//! anchored at `received_at`.
//!
//! Owner: server (HTTP layer).
//! Invariants: one stamp per request, built once; the ids never become metric labels.

use std::time::Instant;

use metrale_qci_dump::trace::RequestTrace;

#[derive(Debug, Clone, Copy)]
pub(crate) struct RequestStamp {
    pub(crate) trace: RequestTrace,
    /// 2026-10-11: Monotonic instant the HTTP layer received the request; the scheduler's
    /// prefill instant is compared against it for the queue time.
    pub(crate) received_at: Instant,
    pub(crate) received_unix_ms: i64,
}

impl RequestStamp {
    /// 2026-10-11: The ids of this request, as the usage record keys them.
    pub(crate) fn ids(
        &self,
    ) -> (
        metrale_qci_dump::contract::ids::RequestId,
        metrale_qci_dump::contract::ids::TraceId,
    ) {
        (self.trace.request_id, self.trace.trace_id)
    }
}
