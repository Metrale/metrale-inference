// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: A measured capture for `met circuit headroom --trace`: per-kernel time over a
//! window of whole steps, as `scripts/headroom_trace.py` exports it from an nsys SQLite trace
//! (or any profiler whose per-launch start/end times give the same sums).
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - The text is parsed, never guessed: an unknown key, a missing fingerprint or a window of no
//!   whole step is an error.
//! - `exclusive_us` is the part of a kernel's time no earlier-launched kernel overlaps, clipped to
//!   the window, so the exclusive times of all kernels sum to the GPU's busy time and never
//!   exceed `window_us` ([`Trace::check`]).

use crate::prelude::*;

use alloc::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// 2026-10-22: A stated quantity and where it comes from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stated {
    /// 2026-10-22: The value.
    pub value: f64,
    /// 2026-10-22: How it was obtained (a measurement's fingerprint, or "stated by ...").
    pub basis: String,
}

/// 2026-10-22: What the capture ran, beyond the recipe: the shard one rank plans (dims that
/// differ from the instance's whole-model dims) and the KV length the decode attended.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TracePlan {
    /// 2026-10-22: Dims replaced on the instance (one rank's shard under tensor parallelism).
    #[serde(default)]
    pub dims: BTreeMap<String, u64>,
    /// 2026-10-22: Why those dims (the split rule they follow).
    pub dims_basis: Option<String>,
    /// 2026-10-22: KV tokens each sequence attends over in the window.
    pub kv_tokens: Option<u64>,
    /// 2026-10-22: Where the KV length comes from.
    pub kv_basis: Option<String>,
}

/// 2026-10-22: One kernel function's time in the window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceKernel {
    /// 2026-10-22: The function name as captured (nsys `shortName`), matched to the plan's
    /// `KernelId::func`.
    pub kernel: String,
    /// 2026-10-22: The plan site (`block.local`) the exporter attributed these launches to, from
    /// their launch shape; `None`: matched by name alone.
    pub site: Option<String>,
    /// 2026-10-22: The attribution's shape label (what told these launches apart).
    pub label: Option<String>,
    /// 2026-10-22: What the exporter knows these launches do differently from the plan's site
    /// (another lowering, work of a neighbouring site folded in).
    pub note: Option<String>,
    /// 2026-10-22: Launches in the window.
    pub calls: u64,
    /// 2026-10-22: Sum of the launches' durations, microseconds.
    pub total_us: f64,
    /// 2026-10-22: Sum of the parts no earlier-launched kernel overlaps, microseconds.
    pub exclusive_us: f64,
    /// 2026-10-22: Energy of these launches, microjoules, when a power capture attributed it.
    pub energy_uj: Option<f64>,
}

/// 2026-10-22: A capture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Trace {
    /// 2026-10-22: Schema version (1).
    pub schema: u32,
    /// 2026-10-22: The fingerprint: trace file, build, binary, recipe and flags, box, date.
    pub source: String,
    /// 2026-10-22: The kernel launched once per step that delimits the window.
    pub step_kernel: String,
    /// 2026-10-22: Whole steps in the window.
    pub steps: u64,
    /// 2026-10-22: From the first step's start to the last step's end, microseconds.
    pub window_us: f64,
    /// 2026-10-22: Tokens a step emits on average in the capture.
    pub tokens_per_step: Option<Stated>,
    /// 2026-10-22: The time one launch takes when it does no work on this device.
    pub launch_floor_us: Option<Stated>,
    /// 2026-10-22: What the capture ran beyond the recipe.
    #[serde(default)]
    pub plan: TracePlan,
    /// 2026-10-22: Per kernel (and attributed site).
    #[serde(rename = "kernel")]
    pub kernels: Vec<TraceKernel>,
}

impl Trace {
    /// 2026-10-22: The measured step, microseconds.
    pub fn step_us(&self) -> f64 {
        self.window_us / self.steps as f64
    }

    /// 2026-10-22: Refuse a capture whose numbers cannot be a window of whole steps.
    pub fn check(&self) -> Result<(), String> {
        if self.schema != 1 {
            return Err(format!(
                "trace schema {} (this build reads 1)",
                self.schema
            ));
        }
        if self.source.trim().is_empty() {
            return Err("trace has no `source` fingerprint".into());
        }
        if self.steps == 0 || self.window_us.is_nan() || self.window_us <= 0.0 {
            return Err(format!(
                "trace window holds no whole step ({} steps over {} us)",
                self.steps, self.window_us
            ));
        }
        let mut busy = 0.0;
        for k in &self.kernels {
            let bad = !(k.total_us >= 0.0 && k.exclusive_us >= 0.0);
            if bad || k.exclusive_us > k.total_us * (1.0 + 1e-9) {
                return Err(format!(
                    "trace kernel `{}`: exclusive {} us must lie in [0, total {} us]",
                    k.kernel, k.exclusive_us, k.total_us
                ));
            }
            busy += k.exclusive_us;
        }
        if busy > self.window_us * (1.0 + 1e-9) {
            return Err(format!(
                "trace exclusive time {busy} us exceeds its window {} us: exclusive times must be \
                 clipped to the window and must not overlap",
                self.window_us
            ));
        }
        Ok(())
    }
}

/// 2026-10-22: Parse and check a trace file.
pub fn parse_trace(text: &str) -> Result<Trace, String> {
    let t: Trace = toml::from_str(text).map_err(|e| format!("trace: {e}"))?;
    t.check()?;
    Ok(t)
}
