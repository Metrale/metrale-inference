// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Fork/join on a second stream. A rule may run its group on the side stream
//! (`stream = "side"`), where legacy overlaps that work with the main stream's (the MoE prefill's
//! shared expert beside the router, `adaptive_fp8.rs:48-55`). The fuser derives every event the
//! overlap needs from the plan; no rule writes one.
//!
//! - A fork (record on main after group `after`, or before the plan's first launch; the side
//!   stream waits before side group `before`) puts a side group after all the main work issued
//!   before it, as legacy records its ready event where it issues the side work. So the side
//!   stream never runs ahead of main's earlier work, this step's or a previous step's.
//! - A join (record on side after side group `after`, main waits before group `before`, or
//!   before the plan ends) orders a main group after every earlier side group it depends on:
//!   one that writes an edge it reads, or that touches a scratch region it touches. Every side
//!   group is joined before the plan ends, so a captured graph has no unjoined fork.
//! - The side stream runs its groups in plan order, so an event covers every earlier group of
//!   the stream it was recorded on; an event already implied by an earlier one is not added.
//! - A side group may run beside any main group between its fork and its join, so the buffer
//!   planner keeps every edge it touches live over that window ([`side_windows`]).
//!
//! Owner: metrale-circuit.
//! Invariants: every side group has a join after it; a plan with no side group has no event.

use std::collections::{BTreeMap, BTreeSet};

use crate::ir::EdgeIdx;

/// 2026-10-04: The stream a group runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum Stream {
    /// 2026-10-04: The step's stream.
    #[default]
    Main,
    /// 2026-10-04: The executor's second stream.
    Side,
}

impl Stream {
    /// 2026-10-04: The spelling in rules and plan text.
    pub fn name(self) -> &'static str {
        match self {
            Stream::Main => "main",
            Stream::Side => "side",
        }
    }

    /// 2026-10-04: The stream a rule names.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "main" => Some(Stream::Main),
            "side" => Some(Stream::Side),
            _ => None,
        }
    }
}

/// 2026-10-04: Which way an event orders the two streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EventKind {
    /// 2026-10-04: Recorded on main, waited on by the side stream.
    Fork,
    /// 2026-10-04: Recorded on the side stream, waited on by main.
    Join,
}

/// 2026-10-04: One cross-stream event of a plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamEvent {
    /// 2026-10-04: Fork or join.
    pub kind: EventKind,
    /// 2026-10-04: The group the event is recorded after, on its recording stream (a main group
    /// for a fork, a side group for a join); `None` for a fork before the plan's first launch.
    pub after: Option<usize>,
    /// 2026-10-04: The group the waiting stream waits before; `None` for a join at the end of
    /// the plan.
    pub before: Option<usize>,
}

/// 2026-10-04: One event as the plan text and the digest write it: `fork g0003 -> g0004`,
/// `join g0004 -> g0007`, with `start` and `end` for the plan's ends.
pub fn event_text(e: &StreamEvent) -> String {
    let at = |g: Option<usize>, end: &str| g.map_or(end.to_string(), |g| format!("g{g:04}"));
    match e.kind {
        EventKind::Fork => format!("fork {} -> {}", at(e.after, "start"), at(e.before, "end")),
        EventKind::Join => format!("join {} -> {}", at(e.after, "start"), at(e.before, "end")),
    }
}

/// 2026-10-04: What one group touches, for the event derivation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Access {
    /// 2026-10-04: Its stream.
    pub stream: Stream,
    /// 2026-10-04: Edges it reads from outside itself.
    pub reads: Vec<EdgeIdx>,
    /// 2026-10-04: Edges it writes.
    pub writes: Vec<EdgeIdx>,
    /// 2026-10-04: Scratch regions it reads and writes (its rule's `scratch`).
    pub scratch: Vec<String>,
}

/// 2026-10-04: Whether main group `later` must wait for side group `earlier`: it reads an edge
/// `earlier` writes, or they share a scratch region. (A later group never writes an edge an
/// earlier one reads: every edge has one producer, which precedes its readers.)
fn depends(later: &Access, earlier: &Access) -> bool {
    later.reads.iter().any(|e| earlier.writes.contains(e))
        || later.scratch.iter().any(|s| earlier.scratch.contains(s))
}

/// 2026-10-04: The events of groups `groups`, in plan order.
pub fn derive_events(groups: &[Access]) -> Vec<StreamEvent> {
    let mut out = Vec::new();
    // 2026-10-04: The main work the side stream has waited for (`Some(None)`: the plan's
    // start), the latest main group so far, and the latest side group main has waited for.
    let mut side_saw: Option<Option<usize>> = None;
    let (mut last_main, mut main_saw, mut last_side) = (None, None, None);
    for (g, a) in groups.iter().enumerate() {
        match a.stream {
            Stream::Side => {
                if side_saw != Some(last_main) {
                    out.push(StreamEvent {
                        kind: EventKind::Fork,
                        after: last_main,
                        before: Some(g),
                    });
                    side_saw = Some(last_main);
                }
                last_side = Some(g);
            }
            Stream::Main => {
                let dep = (0..g)
                    .rev()
                    .take_while(|&p| main_saw.is_none_or(|s| p > s))
                    .find(|&p| groups[p].stream == Stream::Side && depends(a, &groups[p]));
                if let Some(s) = dep {
                    out.push(StreamEvent {
                        kind: EventKind::Join,
                        after: Some(s),
                        before: Some(g),
                    });
                    main_saw = Some(s);
                }
                last_main = Some(g);
            }
        }
    }
    if let Some(s) = last_side.filter(|&s| main_saw.is_none_or(|m| m < s)) {
        out.push(StreamEvent {
            kind: EventKind::Join,
            after: Some(s),
            before: None,
        });
    }
    out
}

/// 2026-10-04: Per side group, the window of plan positions it may run beside: from the group
/// after its fork (or the plan's first group) to the group its join waits before (or the last
/// group). The buffer planner keeps every edge the group touches live over it.
pub fn side_windows(groups: &[Access], events: &[StreamEvent]) -> BTreeMap<usize, (usize, usize)> {
    let last = groups.len().saturating_sub(1);
    let side: BTreeSet<usize> = (0..groups.len())
        .filter(|&g| groups[g].stream == Stream::Side)
        .collect();
    side.iter()
        .map(|&g| {
            let start = events
                .iter()
                .filter(|e| e.kind == EventKind::Fork && e.before.is_some_and(|b| b <= g))
                .map(|e| e.after.map_or(0, |a| a + 1))
                .max()
                .unwrap_or(0);
            let end = events
                .iter()
                .filter(|e| e.kind == EventKind::Join && e.after.is_some_and(|a| a >= g))
                .map(|e| e.before.unwrap_or(last))
                .min()
                .unwrap_or(last);
            (g, (start, end))
        })
        .collect()
}

#[cfg(test)]
#[path = "streams_tests.rs"]
mod tests;
