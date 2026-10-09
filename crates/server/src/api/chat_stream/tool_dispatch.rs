// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: One dispatch for the streaming detector's tool-call outputs,
//! shared by `handle_token` and both `handle_done` loops; the handler for a
//! policy-checked call (`DetectorOutput::CheckedToolCall`); and the keep-alive
//! chunk for a call whose arguments are held.
//!
//! Owner: server streaming API.
//! Invariants:
//! - A keep-alive is an empty `ToolCallArgs` for a call whose `ToolCallStart`
//!   already went out, so it appends nothing to the call's arguments.

use std::time::{Duration, Instant};

use crate::ir::StreamDelta;
use crate::tool_parser::{self, DetectorOutput};

use super::ctx::StreamCtx;
use super::state::StreamState;
use super::tool_handlers::{
    handle_complete_tool_call, handle_tool_call_args_fragment, handle_tool_call_delta,
    handle_tool_call_end, handle_tool_call_start,
};

type DeltaVec = Vec<StreamDelta>;

/// 2026-10-08: Handle one detector output. A `Content` output is returned for
/// the caller, whose content handling differs per call site; every tool-call
/// output is handled here.
pub(super) fn dispatch_tool_output(
    state: &mut StreamState,
    ctx: &StreamCtx,
    output: DetectorOutput,
    deltas: &mut DeltaVec,
) -> Option<String> {
    match output {
        DetectorOutput::Content(text) => return Some(text),
        DetectorOutput::ToolCall(mut tc, idx) => {
            handle_complete_tool_call(state, ctx, &mut tc, idx, deltas);
        }
        DetectorOutput::ToolCallStart { id, name, idx } => {
            handle_tool_call_start(state, ctx, id, name, idx, deltas);
        }
        DetectorOutput::ToolCallDelta { args, idx } => {
            handle_tool_call_delta(state, ctx, args, idx, deltas);
        }
        DetectorOutput::ToolCallArgsFragment { fragment, idx } => {
            handle_tool_call_args_fragment(state, ctx, fragment, idx, deltas);
        }
        DetectorOutput::ToolCallEnd { idx } => handle_tool_call_end(state, ctx, idx),
        DetectorOutput::CheckedToolCall {
            call,
            idx,
            header_sent,
            refused,
        } => handle_checked_tool_call(state, ctx, call, idx, header_sent, refused, deltas),
    }
    None
}

/// 2026-10-08: Deliver a call the format's policy already checked: its header
/// unless one went out, then its whole arguments, then the end-of-call
/// bookkeeping (`handle_tool_call_end`). Nothing here rewrites or rejects it.
fn handle_checked_tool_call(
    state: &mut StreamState,
    ctx: &StreamCtx,
    call: tool_parser::ToolCall,
    idx: usize,
    header_sent: bool,
    refused: bool,
    deltas: &mut DeltaVec,
) {
    if !header_sent {
        handle_tool_call_start(
            state,
            ctx,
            call.id.clone(),
            call.function.name.clone(),
            idx,
            deltas,
        );
    }
    if refused {
        tracing::info!(tool = %call.function.name, "streaming a refused tool call");
    }
    if let Some(entry) = state.streaming_tool_args.get_mut(&idx) {
        entry.1.push_str(&call.function.arguments);
    }
    deltas.push(StreamDelta::ToolCallArgs {
        index: idx,
        fragment: call.function.arguments,
        token_ids: Vec::new(),
    });
    handle_tool_call_end(state, ctx, idx);
}

/// 2026-10-08: Under `CallPolicy::FailClosed`, push an empty argument chunk
/// for the held call when `keepalive_due` says one is due.
pub(super) fn push_keepalive_if_due(
    state: &mut StreamState,
    ctx: &StreamCtx,
    deltas: &mut DeltaVec,
) {
    let tool_parser::CallPolicy::FailClosed { keepalive } = ctx.call_policy else {
        return;
    };
    let held = state.detector.as_ref().and_then(|d| d.held_call_index());
    let sent_for_held = held.is_some_and(|i| deltas.iter().any(|d| delta_call_index(d) == Some(i)));
    if let Some(index) = held
        && keepalive_due(
            &mut state.keepalive_last,
            held,
            sent_for_held,
            Instant::now(),
            keepalive,
        )
    {
        deltas.push(StreamDelta::ToolCallArgs {
            index,
            fragment: String::new(),
            token_ids: Vec::new(),
        });
    }
}

fn delta_call_index(d: &StreamDelta) -> Option<usize> {
    match d {
        StreamDelta::ToolCallStart { index, .. } | StreamDelta::ToolCallArgs { index, .. } => {
            Some(*index)
        }
        _ => None,
    }
}

/// 2026-10-08: Whether a keep-alive for the held call is due at `now`. `last`
/// is the call and the time something last went out for it: it restarts when
/// the held call changes or `sent_for_held` (a chunk for it went out with this
/// token), and is cleared when no call is held. A keep-alive is due once
/// `interval` has passed since `last`, which then moves to `now`.
pub(super) fn keepalive_due(
    last: &mut Option<(usize, Instant)>,
    held: Option<usize>,
    sent_for_held: bool,
    now: Instant,
    interval: Duration,
) -> bool {
    let Some(idx) = held else {
        *last = None;
        return false;
    };
    match *last {
        Some((i, at)) if i == idx && !sent_for_held => {
            let due = now.saturating_duration_since(at) >= interval;
            if due {
                *last = Some((idx, now));
            }
            due
        }
        _ => {
            *last = Some((idx, now));
            false
        }
    }
}

#[cfg(test)]
mod keepalive_tests {
    use super::keepalive_due;
    use std::time::{Duration, Instant};

    const FIVE: Duration = Duration::from_secs(5);

    #[test]
    fn first_sight_of_a_held_call_starts_the_clock_without_a_chunk() {
        let t0 = Instant::now();
        let mut last = None;
        assert!(!keepalive_due(&mut last, Some(0), true, t0, FIVE));
        assert_eq!(last, Some((0, t0)));
    }

    #[test]
    fn a_chunk_goes_out_once_per_interval_while_the_call_is_held() {
        let t0 = Instant::now();
        let mut last = Some((0, t0));
        assert!(!keepalive_due(
            &mut last,
            Some(0),
            false,
            t0 + Duration::from_secs(4),
            FIVE
        ));
        assert!(keepalive_due(&mut last, Some(0), false, t0 + FIVE, FIVE));
        // 2026-10-08: The clock restarted at the chunk just sent.
        assert!(!keepalive_due(
            &mut last,
            Some(0),
            false,
            t0 + Duration::from_secs(9),
            FIVE
        ));
        assert!(keepalive_due(
            &mut last,
            Some(0),
            false,
            t0 + Duration::from_secs(10),
            FIVE
        ));
    }

    #[test]
    fn real_output_for_the_call_restarts_the_clock() {
        let t0 = Instant::now();
        let mut last = Some((0, t0));
        assert!(!keepalive_due(
            &mut last,
            Some(0),
            true,
            t0 + Duration::from_secs(6),
            FIVE
        ));
        assert!(!keepalive_due(
            &mut last,
            Some(0),
            false,
            t0 + Duration::from_secs(10),
            FIVE
        ));
        assert!(keepalive_due(
            &mut last,
            Some(0),
            false,
            t0 + Duration::from_secs(11),
            FIVE
        ));
    }

    #[test]
    fn a_new_call_or_no_call_never_inherits_the_old_clock() {
        let t0 = Instant::now();
        let mut last = Some((0, t0));
        assert!(!keepalive_due(
            &mut last,
            Some(1),
            false,
            t0 + Duration::from_secs(60),
            FIVE
        ));
        assert_eq!(last, Some((1, t0 + Duration::from_secs(60))));
        assert!(!keepalive_due(
            &mut last,
            None,
            false,
            t0 + Duration::from_secs(90),
            FIVE
        ));
        assert_eq!(last, None);
    }
}
