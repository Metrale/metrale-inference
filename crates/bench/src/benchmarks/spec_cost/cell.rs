// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: The cost of one speculative scheduler step at one batch width `n`, from two
//! `/metrics` snapshots and the GPU-rail energy of the window between them. Pure: no I/O.
//!
//! The arithmetic, per width `n` at draft depth `k`, over a window of `window_ms`:
//! - `k > 0`: `steps` = Δcount of phase `step_mtp` (one per MTP step, serial or batched);
//!   `step_ms` = 1000 · Δsum(`step_mtp`) / `steps`; `draft_ms` = 1000 · Δsum(`propose`) /
//!   `steps`.
//! - `k = 0` (no speculation, one token per sequence per step): `steps` = Δtokens / `n`;
//!   `step_ms` = `window_ms` / `steps`; `draft_ms` = 0.
//! - `verify_ms` = `step_ms` − `draft_ms`; `tok_per_step` = Δtokens / (`n` · `steps`);
//!   `wall_ms` = `window_ms` / `steps`.
//! - `step_j` = the bench's GPU-rail integral (nvidia-smi, `EnergyMeter`) over the window /
//!   `steps`. ASSUMPTION: energy splits between draft and verify in proportion to time, so
//!   `draft_j` = `step_j` · `draft_ms` / `step_ms` and `verify_j` = `step_j` − `draft_j`. The
//!   rail is not sampled per phase, so the split is not measured.
//! - `nvml_j` = the change in the serve's NVML energy counter
//!   (`metrale_gpu_energy_millijoules_total`) between the two snapshots / `steps`: recorded as
//!   a cross-check, never used for the split. Measured 2026-10-04 on the dense 27B (16 cells,
//!   two interleaved rounds, dgx2): this delta repeated within 8.2 % (median, max 9.8 %) between
//!   rounds while the integral repeated within 0.7 % (max 2.1 %), so the integral is the cost.
//!
//! `step_j` covers the whole window, including the scheduler loop's work between steps, while
//! `step_ms` is the time inside `step_mtp` only; at `k > 0`, `wall_ms` − `step_ms` is that
//! loop share, and its energy lands in `verify_j` and `draft_j` by the same proportion.
//!
//! Owner: bench, spec-cost.
//! Invariants:
//! - A vacuous cell carries no numbers, only its reason.
//! - A measured cell has `steps` >= `MIN_STEPS` and 0 <= `draft_ms` <= `step_ms`.

use std::collections::BTreeMap;

use anyhow::{Result, bail};

use super::prom::Scrape;

/// 2026-10-04: The live per-token counter. `metrale_generation_tokens_total` is added once
/// per request at completion, so it does not move while every stream is still running
/// (`crates/server/src/metrics.rs`).
pub(crate) const TOKENS_METRIC: &str = "metrale_decoded_tokens_total";
/// 2026-10-04: The scheduler phase histogram (`crates/telemetry/src/export/
/// prometheus_layers.rs`); it is rendered only at `--telemetry basic` or `kernel`, and a
/// phase only once it has fired.
pub(crate) const PHASE_COUNT: &str = "metrale_sched_phase_seconds_count";
pub(crate) const PHASE_SUM: &str = "metrale_sched_phase_seconds_sum";
/// 2026-10-04: Phase labels from `crates/server/src/scheduler/mtp_timing.rs` `NAMES`.
pub(crate) const STEP_PHASE: &str = "step_mtp";
pub(crate) const PROPOSE_PHASE: &str = "propose";
/// 2026-10-04: Fewer steps than this in a window make the cell vacuous.
pub(crate) const MIN_STEPS: f64 = 20.0;

/// 2026-10-04: The serve's NVML energy counter (`telemetry export::prometheus`), reset- and
/// wrap-corrected; rendered at `--telemetry basic` once the device sampler has read the GPU.
pub(crate) const ENERGY_METRIC: &str = "metrale_gpu_energy_millijoules_total";

/// 2026-10-04: One phase's histogram totals.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct PhaseTotals {
    pub(crate) count: f64,
    pub(crate) sum_s: f64,
}

/// 2026-10-04: The counters one snapshot reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Counters {
    pub(crate) tokens: f64,
    /// 2026-10-04: `None` when the page has no `step_mtp` series.
    pub(crate) step: Option<PhaseTotals>,
    /// 2026-10-04: Zero when the page has no `propose` series (it has not fired).
    pub(crate) propose: PhaseTotals,
    /// 2026-10-04: `None` when the page has no [`ENERGY_METRIC`] (no NVML on the serve).
    pub(crate) energy_mj: Option<f64>,
}

impl Counters {
    pub(crate) fn read(scrape: &Scrape) -> Result<Self> {
        let Some(tokens) = scrape.value(TOKENS_METRIC, &[])? else {
            bail!("/metrics has no {TOKENS_METRIC}: the serve does not count decoded tokens");
        };
        Ok(Self {
            tokens,
            step: phase(scrape, STEP_PHASE)?,
            propose: phase(scrape, PROPOSE_PHASE)?.unwrap_or_default(),
            energy_mj: scrape.value(ENERGY_METRIC, &[])?,
        })
    }
}

fn phase(scrape: &Scrape, name: &str) -> Result<Option<PhaseTotals>> {
    let labels = [("phase", name)];
    match (
        scrape.value(PHASE_COUNT, &labels)?,
        scrape.value(PHASE_SUM, &labels)?,
    ) {
        (None, None) => Ok(None),
        (Some(count), Some(sum_s)) => Ok(Some(PhaseTotals { count, sum_s })),
        _ => bail!("/metrics carries only one of {PHASE_COUNT} and {PHASE_SUM} for {name:?}"),
    }
}

/// 2026-10-04: Refuse a speculative run whose serve does not time its steps.
pub(crate) fn require_step_series(k: u8, snapshot: &Counters) -> Result<()> {
    if k > 0 && snapshot.step.is_none() {
        bail!(
            "/metrics has no {PHASE_COUNT}{{phase=\"{STEP_PHASE}\"}} series while k = {k}: the \
             serve needs `--telemetry basic` (or `kernel`) to time its scheduler phases, and must \
             be running speculative decoding (k = 0 measures a serve without --speculative)"
        );
    }
    Ok(())
}

/// 2026-10-04: What one width's window observed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Window {
    pub(crate) n: usize,
    pub(crate) k: u8,
    pub(crate) before: Counters,
    pub(crate) after: Counters,
    pub(crate) window_s: f64,
    /// 2026-10-04: The bench's GPU-rail integral over the window; `None` when the rail was not
    /// sampled.
    pub(crate) energy_j: Option<f64>,
    /// 2026-10-04: Streams that had finished or failed when the window closed.
    pub(crate) ended_early: usize,
}

/// 2026-10-04: The per-step cost at one width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StepCost {
    pub(crate) steps: f64,
    pub(crate) wall_ms: f64,
    pub(crate) step_ms: f64,
    pub(crate) verify_ms: f64,
    pub(crate) draft_ms: f64,
    pub(crate) verify_j: Option<f64>,
    pub(crate) draft_j: Option<f64>,
    pub(crate) nvml_j: Option<f64>,
    pub(crate) tok_per_step: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CellVerdict {
    Vacuous(String),
    Measured(StepCost),
}

pub(crate) fn evaluate(w: &Window) -> CellVerdict {
    let vacuous = CellVerdict::Vacuous;
    if w.ended_early > 0 {
        return vacuous(format!(
            "{} of {} streams finished or failed before the window closed",
            w.ended_early, w.n
        ));
    }
    let d_tokens = w.after.tokens - w.before.tokens;
    if d_tokens < 0.0 {
        return vacuous(format!(
            "{TOKENS_METRIC} went backwards: the serve restarted"
        ));
    }
    let window_ms = w.window_s * 1000.0;
    let n = w.n as f64;
    let (steps, step_ms, draft_ms) = if w.k == 0 {
        let steps = d_tokens / n;
        (steps, window_ms / steps, 0.0)
    } else {
        let (Some(a), Some(b)) = (w.before.step, w.after.step) else {
            return vacuous(format!("no {STEP_PHASE} series in one of the snapshots"));
        };
        let steps = b.count - a.count;
        let step_s = b.sum_s - a.sum_s;
        let propose_count = w.after.propose.count - w.before.propose.count;
        let propose_s = w.after.propose.sum_s - w.before.propose.sum_s;
        if steps < 0.0 || step_s < 0.0 || propose_count < 0.0 || propose_s < 0.0 {
            return vacuous("a phase histogram went backwards: the serve restarted".to_string());
        }
        if steps >= MIN_STEPS && propose_count == 0.0 {
            return vacuous(format!(
                "no {PROPOSE_PHASE} phase was timed in {steps} steps: this width's verify path \
                 does not mark it (the serial K3 and K4 arms do not), so the draft share cannot \
                 be separated"
            ));
        }
        (steps, 1000.0 * step_s / steps, 1000.0 * propose_s / steps)
    };
    if steps.is_nan() || steps < MIN_STEPS {
        return vacuous(format!(
            "{steps:.1} steps in the window, fewer than {MIN_STEPS}"
        ));
    }
    if !step_ms.is_finite() || step_ms <= 0.0 {
        return vacuous(format!("step time {step_ms} ms is not positive"));
    }
    if draft_ms > step_ms {
        return vacuous(format!(
            "propose time {draft_ms:.3} ms per step exceeds the step's {step_ms:.3} ms"
        ));
    }
    let counter_j = match (w.before.energy_mj, w.after.energy_mj) {
        (Some(a), Some(b)) if b < a => {
            return vacuous(format!(
                "{ENERGY_METRIC} went backwards: the serve restarted"
            ));
        }
        (Some(a), Some(b)) => Some((b - a) / 1000.0),
        _ => None,
    };
    let step_j = w.energy_j.map(|j| j / steps);
    let draft_j = step_j.map(|j| j * draft_ms / step_ms);
    CellVerdict::Measured(StepCost {
        steps,
        wall_ms: window_ms / steps,
        step_ms,
        verify_ms: step_ms - draft_ms,
        draft_ms,
        verify_j: step_j.zip(draft_j).map(|(s, d)| s - d),
        draft_j,
        nvml_j: counter_j.map(|j| j / steps),
        tok_per_step: d_tokens / (n * steps),
    })
}

/// 2026-10-04: The record keys for one width: `n{n}_vacuous` always, the costs only for a
/// measured cell, and the joule keys only when the rail was sampled.
pub(crate) fn record(n: usize, verdict: &CellVerdict, m: &mut BTreeMap<String, f64>) {
    let mut put = |key: &str, v: f64| {
        m.insert(record_key(n, key), v);
    };
    match verdict {
        CellVerdict::Vacuous(_) => put(KEY_VACUOUS, 1.0),
        CellVerdict::Measured(c) => {
            put(KEY_VACUOUS, 0.0);
            put(KEY_STEPS, c.steps);
            put(KEY_WALL_MS, c.wall_ms);
            put(KEY_VERIFY_MS, c.verify_ms);
            put(KEY_DRAFT_MS, c.draft_ms);
            put(KEY_TOK_PER_STEP, c.tok_per_step);
            if let Some(j) = c.verify_j {
                put(KEY_VERIFY_J, j);
            }
            if let Some(j) = c.draft_j {
                put(KEY_DRAFT_J, j);
            }
            if let Some(j) = c.nvml_j {
                put(KEY_NVML_J, j);
            }
        }
    }
}

/// 2026-10-04: The record key of field `field` at width `n`; `table_input` reads with it.
pub(crate) fn record_key(n: usize, field: &str) -> String {
    format!("n{n}_{field}")
}

/// 2026-10-04: The record key of the draft depth the run measured.
pub(crate) const KEY_K: &str = "k";
// 2026-10-04: Per-width record fields, written by `record` and read by `table_input`.
pub(crate) const KEY_VACUOUS: &str = "vacuous";
pub(crate) const KEY_STEPS: &str = "steps";
pub(crate) const KEY_WALL_MS: &str = "wall_ms";
pub(crate) const KEY_VERIFY_MS: &str = "verify_ms";
pub(crate) const KEY_DRAFT_MS: &str = "draft_ms";
pub(crate) const KEY_TOK_PER_STEP: &str = "tok_per_step";
pub(crate) const KEY_VERIFY_J: &str = "verify_j";
pub(crate) const KEY_DRAFT_J: &str = "draft_j";
pub(crate) const KEY_NVML_J: &str = "nvml_j";

#[cfg(test)]
#[path = "cell_tests.rs"]
mod tests;
