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
    /// 2026-10-23: How the captured serve launched each run (`mode n=rows`): from the plan or by
    /// the hand-written layer code. Required by [`Trace::check`]: the plan-executed share is a
    /// fact about the serve, never assumed.
    #[serde(default)]
    pub execution: BTreeMap<String, RunExecution>,
}

impl TracePlan {
    /// 2026-10-23: What makes two captures one configuration: the dims, the KV length and who
    /// executed each run. The bases and reasons are provenance (they name each capture's own usage
    /// file), so they may differ between captures that pool.
    fn configuration(
        &self,
    ) -> (
        &BTreeMap<String, u64>,
        Option<u64>,
        Vec<(&String, ExecutedBy, &[String])>,
    ) {
        let exec = self
            .execution
            .iter()
            .map(|(r, e)| (r, e.by, e.sites.as_slice()))
            .collect();
        (&self.dims, self.kv_tokens, exec)
    }
}

/// 2026-10-23: Who issued a run's launches in the captured serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutedBy {
    /// 2026-10-23: The circuit executor, from the fused plan.
    Plan,
    /// 2026-10-23: The model's own layer code (its hand-written dispatch).
    Legacy,
}

impl ExecutedBy {
    /// 2026-10-23: The spelling in reports.
    pub fn name(self) -> &'static str {
        match self {
            ExecutedBy::Plan => "plan",
            ExecutedBy::Legacy => "legacy",
        }
    }
}

/// 2026-10-23: How one run was executed: by whom, which of its sites the executor issued when
/// it issued only some (`["*"]` is every site), and why (the mode flag, or the reason no
/// executor exists).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunExecution {
    /// 2026-10-23: Plan or legacy.
    pub by: ExecutedBy,
    /// 2026-10-23: The sites (`block [locals]` as the headroom names them) the executor issued:
    /// non-empty under `plan`, empty under `legacy`.
    #[serde(default)]
    pub sites: Vec<String>,
    /// 2026-10-23: Why: the serve's declared mode, or the missing capability.
    pub why: String,
}

impl RunExecution {
    /// 2026-10-23: Refuse a statement that cannot be true of a serve.
    fn check(&self, run: &str) -> Result<(), String> {
        if self.why.trim().is_empty() {
            return Err(format!("trace execution of `{run}` states no `why`"));
        }
        match (self.by, self.sites.is_empty()) {
            (ExecutedBy::Plan, true) => Err(format!(
                "trace execution of `{run}` is `plan` but names no sites (`[\"*\"]` is every site)"
            )),
            (ExecutedBy::Legacy, false) => Err(format!(
                "trace execution of `{run}` is `legacy` but names executed sites {:?}",
                self.sites
            )),
            _ => Ok(()),
        }
    }
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
    /// 2026-10-22: Per step: the durations of the launches that start in it (empty: no series).
    #[serde(default)]
    pub step_total_us: Vec<f64>,
    /// 2026-10-22: Per step: the exclusive time inside it.
    #[serde(default)]
    pub step_exclusive_us: Vec<f64>,
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
    /// 2026-10-22: Per step: its duration, anchor to anchor (empty: the capture has no series and
    /// the headroom states no interval).
    #[serde(default)]
    pub step_us: Vec<f64>,
    /// 2026-10-22: The steps of each pooled capture, in order (empty: one capture).
    #[serde(default)]
    pub segments: Vec<u64>,
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
            return Err(format!("trace schema {} (this build reads 1)", self.schema));
        }
        if self.source.trim().is_empty() {
            return Err("trace has no `source` fingerprint".into());
        }
        if self.plan.execution.is_empty() {
            return Err(
                "trace states no `[plan.execution]`: which runs the captured serve executed from \
                 the plan (scripts/headroom_trace.py --execution)"
                    .into(),
            );
        }
        for (run, e) in &self.plan.execution {
            e.check(run)?;
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
        if !self.step_us.is_empty() {
            self.check_series(busy)?;
        }
        if !self.segments.is_empty() && self.segments.iter().sum::<u64>() != self.steps {
            return Err(format!(
                "trace segments {:?} do not add up to its {} steps",
                self.segments, self.steps
            ));
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

impl Trace {
    /// 2026-10-22: The per-step series exist for every kernel and add up to the totals (to the
    /// exporter's rounding, 0.5 ns per step).
    fn check_series(&self, _busy: f64) -> Result<(), String> {
        let n = usize::try_from(self.steps).map_err(|e| e.to_string())?;
        let tol = |x: f64| 1e-9 * x.abs() + 5e-4 * n as f64;
        let sum = |v: &[f64]| v.iter().sum::<f64>();
        if self.step_us.len() != n
            || (sum(&self.step_us) - self.window_us).abs() > tol(self.window_us)
        {
            return Err(format!(
                "trace step_us holds {} steps summing to {} us; the window is {} steps, {} us",
                self.step_us.len(),
                sum(&self.step_us),
                n,
                self.window_us
            ));
        }
        for k in &self.kernels {
            let ok = k.step_total_us.len() == n
                && k.step_exclusive_us.len() == n
                && (sum(&k.step_total_us) - k.total_us).abs() <= tol(k.total_us)
                && (sum(&k.step_exclusive_us) - k.exclusive_us).abs() <= tol(k.exclusive_us);
            if !ok {
                return Err(format!(
                    "trace kernel `{}`: its per-step series must have {n} steps and add up to its totals",
                    k.kernel
                ));
            }
        }
        Ok(())
    }

    /// 2026-10-22: The steps of each capture.
    pub fn segment_lengths(&self) -> Vec<usize> {
        let segs: Vec<u64> = if self.segments.is_empty() {
            vec![self.steps]
        } else {
            self.segments.clone()
        };
        segs.iter()
            .map(|&x| usize::try_from(x).unwrap_or(0))
            .collect()
    }

    /// 2026-10-22: Captures pooled into this one.
    pub fn captures(&self) -> usize {
        self.segments.len().max(1)
    }
}

/// 2026-10-22: Pool captures of one configuration: same step kernel, plan (shard dims, KV) and
/// tokens per step, or refused. Kernels are joined by (kernel, site, label, note); a kernel a
/// capture lacks has zero time in its steps. The launch floor is the median of the captures'.
pub fn pool(traces: &[Trace]) -> Result<Trace, String> {
    let Some(first) = traces.first() else {
        return Err("no capture to pool".into());
    };
    if traces.len() == 1 {
        return Ok(first.clone());
    }
    for t in traces {
        t.check()?;
        let same = t.step_kernel == first.step_kernel
            && t.plan.configuration() == first.plan.configuration()
            && t.tokens_per_step.as_ref().map(|x| x.value)
                == first.tokens_per_step.as_ref().map(|x| x.value);
        if !same {
            return Err(format!(
                "captures `{}` and `{}` differ in step kernel, plan or tokens per step: pooling them \
                 would mix configurations",
                first.source, t.source
            ));
        }
        if t.step_us.is_empty() {
            return Err(format!(
                "capture `{}` has no per-step series to pool",
                t.source
            ));
        }
    }
    type Key = (String, Option<String>, Option<String>, Option<String>);
    let mut keys: Vec<Key> = Vec::new();
    for t in traces {
        for k in &t.kernels {
            let key = (
                k.kernel.clone(),
                k.site.clone(),
                k.label.clone(),
                k.note.clone(),
            );
            if !keys.contains(&key) {
                keys.push(key);
            }
        }
    }
    let mut kernels = Vec::new();
    for key in keys {
        let mut row = TraceKernel {
            kernel: key.0.clone(),
            site: key.1.clone(),
            label: key.2.clone(),
            note: key.3.clone(),
            calls: 0,
            total_us: 0.0,
            exclusive_us: 0.0,
            energy_uj: Some(0.0),
            step_total_us: Vec::new(),
            step_exclusive_us: Vec::new(),
        };
        for t in traces {
            let n = usize::try_from(t.steps).map_err(|e| e.to_string())?;
            match t.kernels.iter().find(|k| {
                (&k.kernel, &k.site, &k.label, &k.note) == (&key.0, &key.1, &key.2, &key.3)
            }) {
                Some(k) => {
                    row.calls += k.calls;
                    row.total_us += k.total_us;
                    row.exclusive_us += k.exclusive_us;
                    row.energy_uj = row.energy_uj.zip(k.energy_uj).map(|(a, b)| a + b);
                    row.step_total_us.extend(&k.step_total_us);
                    row.step_exclusive_us.extend(&k.step_exclusive_us);
                }
                None => {
                    row.energy_uj = None;
                    row.step_total_us.extend(core::iter::repeat_n(0.0, n));
                    row.step_exclusive_us.extend(core::iter::repeat_n(0.0, n));
                }
            }
        }
        kernels.push(row);
    }
    let floors: Vec<f64> = traces
        .iter()
        .filter_map(|t| t.launch_floor_us.as_ref().map(|l| l.value))
        .collect();
    let pooled = Trace {
        schema: 1,
        source: traces
            .iter()
            .map(|t| t.source.as_str())
            .collect::<Vec<_>>()
            .join(" | "),
        step_kernel: first.step_kernel.clone(),
        steps: traces.iter().map(|t| t.steps).sum(),
        window_us: traces.iter().map(|t| t.window_us).sum(),
        step_us: traces
            .iter()
            .flat_map(|t| t.step_us.iter().copied())
            .collect(),
        segments: traces.iter().map(|t| t.steps).collect(),
        tokens_per_step: first.tokens_per_step.clone(),
        launch_floor_us: (floors.len() == traces.len())
            .then(|| super::stats::median(&floors))
            .flatten()
            .map(|v| Stated {
                value: v,
                basis: format!("median of the {} captures' launch floors", floors.len()),
            }),
        plan: first.plan.clone(),
        kernels,
    };
    pooled.check()?;
    Ok(pooled)
}

/// 2026-10-22: Parse and check a trace file.
pub fn parse_trace(text: &str) -> Result<Trace, String> {
    let t: Trace = toml::from_str(text).map_err(|e| format!("trace: {e}"))?;
    t.check()?;
    Ok(t)
}
