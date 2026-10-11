// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The `StreamEvent::Error(msg)` arm of the streaming chat handler.
//!
//! Owner: server streaming API.
//! Invariants: none beyond the types.

use crate::ir::StreamDelta;

use super::ctx::StreamCtx;

type DeltaVec = Vec<StreamDelta>;

pub(super) fn handle_error(ctx: &StreamCtx, msg: String) -> DeltaVec {
    // 2026-09-26: A failed stream refunds its whole rate-limit reservation.
    // REQUESTS_ACTIVE is released by the `ActiveRequestGuard` in `StreamCtx` on drop.
    if let Some(ref rctx) = ctx.req_ctx {
        ctx.state
            .rate_limiter
            .refund_tokens(&rctx.identity, rctx.reserved_tokens);
    }
    // 2026-10-11: A stream that fails after dispatch is an engine error.
    ctx.pending
        .fail(metrale_qci_dump::contract::usage::EngineErrorKind::Internal);
    // 2026-09-26: An OpenAI error envelope. The OpenAI encoder forwards it unchanged as
    // the SSE data; other surfaces translate it.
    // 2026-10-11: It carries the request and trace ids, as every JSON error body does.
    let mut err = serde_json::json!({
        "error": {"message": msg, "type": "server_error", "code": 500}
    });
    ctx.pending.trace().annotate_error(&mut err);
    vec![StreamDelta::Error {
        message: err.to_string(),
    }]
}
