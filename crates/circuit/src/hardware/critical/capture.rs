// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: An event capture for `met circuit whatif` (`critical-trace/v1`): every GPU
//! operation of a window of whole steps with its stream, times, launch correlation and launch
//! configuration, every API call of the launching thread, every event record and every
//! synchronization, as `scripts/critical_trace.py` exports them from an nsys SQLite trace.
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - Parsed, never guessed: the field layout is checked against [`FIELDS`]; a capture whose
//!   times run backwards, whose anchors are not whole steps or whose group ids dangle is refused
//!   with [`ErrorCode::CriticalCaptureInvalid`].
//! - Times are integer nanoseconds relative to the first anchor (negative: the fixed context
//!   before the window).

use crate::prelude::*;

use metrale_error::{ErrorCode, error};
use serde::Deserialize;

/// 2026-10-23: The record layouts this build reads, as the exporter states them.
pub const FIELDS: [(&str, &[&str]); 5] = [
    (
        "op",
        &[
            "kind", "name", "stream", "start", "end", "corr", "graph", "ctas", "threads", "regs",
            "smem", "bytes", "group",
        ],
    ),
    ("api", &["name", "start", "end", "corr"]),
    ("record", &["corr", "stream", "event"]),
    ("sync", &["corr", "kind", "stream", "event", "start", "end"]),
    ("group", &["kernel", "site", "label", "note"]),
];

/// 2026-10-23: What a GPU operation is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum OpKind {
    /// 2026-10-23: A kernel.
    #[serde(rename = "k")]
    Kernel,
    /// 2026-10-23: A memory copy.
    #[serde(rename = "m")]
    Memcpy,
    /// 2026-10-23: A memset.
    #[serde(rename = "s")]
    Memset,
}

/// 2026-10-23: One GPU operation.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Op(
    pub OpKind,
    pub String,
    pub u64,
    pub i64,
    pub i64,
    pub u64,
    pub bool,
    pub u32,
    pub u32,
    pub u32,
    pub u32,
    pub u64,
    pub u32,
);

impl Op {
    /// 2026-10-23: The function (or `memcpy` / `memset`).
    pub fn name(&self) -> &str {
        &self.1
    }
    /// 2026-10-23: The stream it ran on.
    pub fn stream(&self) -> u64 {
        self.2
    }
    /// 2026-10-23: Start, ns.
    pub fn start(&self) -> i64 {
        self.3
    }
    /// 2026-10-23: End, ns.
    pub fn end(&self) -> i64 {
        self.4
    }
    /// 2026-10-23: The launch correlation id (a graph's nodes share their launch's).
    pub fn corr(&self) -> u64 {
        self.5
    }
    /// 2026-10-23: Whether it is a CUDA-graph node.
    pub fn graph(&self) -> bool {
        self.6
    }
    /// 2026-10-23: CTAs, threads per CTA, registers per thread, shared memory per CTA.
    pub fn launch(&self) -> (u32, u32, u32, u32) {
        (self.7, self.8, self.9, self.10)
    }
    /// 2026-10-23: Bytes moved (copies and memsets).
    pub fn bytes(&self) -> u64 {
        self.11
    }
    /// 2026-10-23: Index into [`Capture::groups`].
    pub fn group(&self) -> u32 {
        self.12
    }
}

/// 2026-10-23: One API call of the launching thread: name, start, end, correlation id.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Api(pub String, pub i64, pub i64, pub u64);

/// 2026-10-23: An event record: correlation id, stream, event id.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Record(pub u64, pub u64, pub u64);

/// 2026-10-23: A synchronization: correlation id, CUPTI kind, stream, event id, start, end.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Sync(
    pub u64,
    pub String,
    pub u64,
    pub Option<u64>,
    pub i64,
    pub i64,
);

/// 2026-10-23: A launch group: kernel, attributed plan site, shape label, note (the keys of
/// `headroom_trace.py`'s rows, so the headroom attribution applies to each operation).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Group(
    pub String,
    pub Option<String>,
    pub Option<String>,
    pub Option<String>,
);

/// 2026-10-23: A capture.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capture {
    /// 2026-10-23: `critical-trace/v1`.
    pub schema: String,
    /// 2026-10-23: The fingerprint.
    pub source: String,
    /// 2026-10-23: The kernel launched once per step.
    pub step_kernel: String,
    /// 2026-10-23: Start of every launch of the step kernel, ns.
    pub anchors: Vec<i64>,
    /// 2026-10-23: The record layouts (checked against [`FIELDS`]).
    pub fields: alloc::collections::BTreeMap<String, Vec<String>>,
    /// 2026-10-23: Launch groups.
    pub groups: Vec<Group>,
    /// 2026-10-23: GPU operations, by start.
    pub ops: Vec<Op>,
    /// 2026-10-23: API calls, by start.
    pub api: Vec<Api>,
    /// 2026-10-23: Event records.
    pub record: Vec<Record>,
    /// 2026-10-23: Synchronizations.
    pub sync: Vec<Sync>,
}

impl Capture {
    fn bad(&self, why: String) -> metrale_error::Error {
        error!(ErrorCode::CriticalCaptureInvalid, self.source.clone(), why)
    }

    /// 2026-10-23: Refuse a capture the graph cannot be built from.
    pub fn check(&self) -> metrale_error::Result<()> {
        if self.schema != "critical-trace/v1" {
            return Err(self.bad(format!(
                "schema `{}` (this build reads critical-trace/v1)",
                self.schema
            )));
        }
        for (rec, names) in FIELDS {
            let got = self.fields.get(rec).map(Vec::as_slice).unwrap_or(&[]);
            if got.len() != names.len() || got.iter().zip(names).any(|(a, b)| a != b) {
                return Err(self.bad(format!("`{rec}` fields {got:?}, expected {names:?}")));
            }
        }
        if self.anchors.len() < 2 || self.anchors.windows(2).any(|w| w[1] <= w[0]) {
            return Err(self.bad(format!(
                "{} anchors of `{}`: a window needs two, ascending",
                self.anchors.len(),
                self.step_kernel
            )));
        }
        let n = self.groups.len();
        for (i, o) in self.ops.iter().enumerate() {
            if o.end() < o.start() || o.group() as usize >= n {
                return Err(self.bad(format!(
                    "op {i} `{}`: end {} before start {} or group {} of {n}",
                    o.name(),
                    o.end(),
                    o.start(),
                    o.group()
                )));
            }
        }
        if self.ops.windows(2).any(|w| w[1].start() < w[0].start()) {
            return Err(self.bad("ops are not ordered by start".into()));
        }
        if self.api.windows(2).any(|w| w[1].1 < w[0].1) || self.api.iter().any(|a| a.2 < a.1) {
            return Err(
                self.bad("API calls are not ordered by start, or one ends before it starts".into())
            );
        }
        let anchored = self
            .ops
            .iter()
            .filter(|o| o.name() == self.step_kernel && o.start() >= 0)
            .count();
        if anchored != self.anchors.len() {
            return Err(self.bad(format!(
                "{anchored} launches of `{}` in the window, {} anchors stated",
                self.step_kernel,
                self.anchors.len()
            )));
        }
        Ok(())
    }

    /// 2026-10-23: Whole steps in the window.
    pub fn steps(&self) -> usize {
        self.anchors.len() - 1
    }

    /// 2026-10-23: The measured mean step, microseconds.
    pub fn step_us(&self) -> f64 {
        let w = self.anchors[self.anchors.len() - 1] - self.anchors[0];
        w as f64 / 1e3 / self.steps() as f64
    }
}

/// 2026-10-23: Parse and check a capture.
pub fn parse_capture(text: &str) -> metrale_error::Result<Capture> {
    let c: Capture = serde_json::from_str(text).map_err(|e| {
        error!(
            ErrorCode::CriticalCaptureInvalid,
            "(unparsed)",
            format!("not critical-trace/v1 JSON: {e}")
        )
    })?;
    c.check()?;
    Ok(c)
}
