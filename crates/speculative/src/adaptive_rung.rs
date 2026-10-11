// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Adaptive MTP draft count for batch widths 9..=16: one or two
//! drafts, chosen from the observed accept statistics, with the static ladder
//! (`metrale_model_layers::speculative::mtp_ladder_drafts`) as the floor.
//!
//! Owner: speculative.
//! Invariants:
//! - `drafts_for` never returns fewer drafts than the static ladder, and
//!   returns exactly the ladder's count outside 9..=16, when
//!   `RungParams::disabled`, when `METRALE_NO_MTP_K_LADDER` is set, or when
//!   `num_drafts < 2`.
//! - `observe` changes no state for widths outside 9..=16 or when disabled.
//!
//! With `p1` the first-draft accept rate and `p2_cond` the second-draft accept
//! rate given an accepted first draft, two drafts yield `token_ratio(p1,
//! p2_cond) = 1 + p1*p2_cond/(1 + p1)` times the tokens per verify step of one
//! draft. The controller moves to two drafts when the smoothed ratio reaches
//! `enter` and back to one when it falls below `leave` (`next_state`). The
//! estimates are EWMAs seeded with their first sample. With the default
//! tiered verify pools the scheduler clamps the lift back to one draft
//! (`spec_capacity::clamp_drafts_to_slot_capacity`; see that module).
//!
//! 2026-10-10: The decision is the speculation controller's (`crate::spec_ctl`). The two
//! thresholds are a cost model in disguise: "two drafts when `E(2)/E(1) >= enter`" is
//! `E(2)/cost(2) >= E(1)/cost(1)` with `cost(2)/cost(1) = enter`, and `leave` is a hysteresis
//! margin `enter/leave - 1` against leaving depth ([`next_state`]). The EWMAs are
//! `spec_ctl::accept::AcceptCounts` in rate form with a steady-state seed (equal to a seeded
//! EWMA), and the probe triggers are `spec_ctl::reprobe::DeepProbe`.
//!
//! At one draft nothing proposes a second token, so `p2_cond` cannot be
//! observed there. The controller then runs two drafts for a flush (a probe)
//! only on evidence or after a long backstop; see `drafts_for`. Measured
//! 2026-08-01 on dgx1 at C=16 from flush timestamps: a steady flush took
//! 1.16 s and the flush that entered a probe 2.90 s, so a probe cost about
//! 1.74 s.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use parking_lot::Mutex;

use crate::spec_ctl::accept::{AcceptCounts, AcceptParams, SeedWeight};
use crate::spec_ctl::chain::expected_tokens;
use crate::spec_ctl::cost::StepCost;
use crate::spec_ctl::decide::{Candidate, Margins, Objective, choose};
use crate::spec_ctl::reprobe::{DeepProbe, DeepProbeState};

/// 2026-09-25: The batch widths the controller adapts. Every other width
/// keeps the static ladder's count.
const BAND: std::ops::RangeInclusive<usize> = 9..=16;

/// 2026-09-25: Move to two drafts when the smoothed token ratio is at or
/// above this. Overridden by `METRALE_MTP_RUNG_ENTER`.
const ENTER: f64 = 1.32;
/// 2026-09-25: Move back to one draft when the smoothed token ratio falls
/// below this. Overridden by `METRALE_MTP_RUNG_LEAVE`.
const LEAVE: f64 = 1.30;
/// 2026-09-25: EWMA weight of the `p1` and `p2_cond` estimates the decision
/// reads (effective window `1/ALPHA` = 2 flushes). Overridden by
/// `METRALE_MTP_RUNG_ALPHA`, capped at 1.
const ALPHA: f64 = 0.5;
/// 2026-09-25: EWMA weight of the `p1` estimate the probe trigger reads
/// (effective window about 7 flushes). It is slower than [`ALPHA`] so that
/// flush-to-flush noise in `p1` does not buy probes. Overridden by
/// `METRALE_MTP_RUNG_ALPHA_SLOW`, capped at 1.
const ALPHA_SLOW: f64 = 0.15;
/// 2026-09-25: Backstop probe interval in flushes, for a `p2_cond` drift at
/// constant `p1` that [`P1_TRIGGER`] cannot see. With the probe and flush
/// times measured 2026-08-01 (1.74 s and 1.16 s, module doc), 2048 flushes is
/// one probe per about 40 minutes, about 0.07% of the time. Overridden by
/// `METRALE_MTP_RUNG_PROBE_TICKS`.
const PROBE_TICKS: u64 = 2048;
/// 2026-09-25: Probe when the slow `p1` EWMA has risen this far above its
/// value at the last depth flush. Overridden by `METRALE_MTP_RUNG_P1_TRIGGER`.
const P1_TRIGGER: f64 = 0.08;

/// 2026-09-25: The controller's thresholds, fixed for the life of one
/// `AdaptiveRung`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RungParams {
    /// 2026-09-25: Pins the static ladder. `from_env` sets it when
    /// `METRALE_MTP_STATIC_RUNG` is present (any value) or the serve runs an explicit
    /// ladder (`speculative::mtp_ladder_pinned`: `--mtp-k-ladder` or
    /// `METRALE_MTP_K_LADDER`): an operator who spells out the rungs gets exactly those
    /// rungs.
    pub disabled: bool,
    pub enter: f64,
    pub leave: f64,
    pub alpha: f64,
    pub alpha_slow: f64,
    pub probe_ticks: u64,
    pub p1_trigger: f64,
}

impl RungParams {
    /// 2026-09-25: The compiled constants, with adaptation on.
    pub const DEFAULTS: Self = Self {
        disabled: false,
        enter: ENTER,
        leave: LEAVE,
        alpha: ALPHA,
        alpha_slow: ALPHA_SLOW,
        probe_ticks: PROBE_TICKS,
        p1_trigger: P1_TRIGGER,
    };

    /// 2026-09-25: Reads the environment. A tunable whose value does not
    /// parse as a finite positive number keeps its constant (`tunable`).
    pub fn from_env() -> Self {
        Self {
            disabled: std::env::var_os("METRALE_MTP_STATIC_RUNG").is_some()
                || metrale_model_layers::speculative::mtp_ladder_pinned(),
            enter: tunable("METRALE_MTP_RUNG_ENTER", ENTER),
            leave: tunable("METRALE_MTP_RUNG_LEAVE", LEAVE),
            alpha: tunable("METRALE_MTP_RUNG_ALPHA", ALPHA).min(1.0),
            alpha_slow: tunable("METRALE_MTP_RUNG_ALPHA_SLOW", ALPHA_SLOW).min(1.0),
            probe_ticks: tunable("METRALE_MTP_RUNG_PROBE_TICKS", PROBE_TICKS as f64) as u64,
            p1_trigger: tunable("METRALE_MTP_RUNG_P1_TRIGGER", P1_TRIGGER),
        }
    }
}

/// 2026-09-25: `var` parsed as `f64` when it is finite and positive,
/// otherwise `default`.
fn tunable(var: &str, default: f64) -> f64 {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(default)
}

/// 2026-09-25: Expected tokens per verify step at two drafts relative to one:
/// `(1 + p1 + p1*p2_cond) / (1 + p1)`. Returns 1.0 when `p1 <= 0` or either
/// input is not finite.
pub fn token_ratio(p1: f64, p2_cond: f64) -> f64 {
    if p1 <= 0.0 || !p1.is_finite() || !p2_cond.is_finite() {
        return 1.0;
    }
    let c = [p1, p2_cond];
    expected_tokens(&c, 2) / expected_tokens(&c, 1)
}

/// 2026-09-25: The second-draft conditional accept implied by a flush at two
/// drafts, where `mean_na = p1 + p1*p2_cond`: `(mean_na - p1) / p1`, clamped
/// to `[0, 1]`. Deeper drafts are not modelled. `None` when `p1 <= 0`.
pub fn p2_cond_from(p1: f64, mean_na: f64) -> Option<f64> {
    (p1 > 0.0).then(|| ((mean_na - p1) / p1).clamp(0.0, 1.0))
}

/// 2026-09-25: The next state from the current one and the smoothed token
/// ratio; `true` means two drafts.
pub fn next_state(at_depth: bool, tr: f64, p: &RungParams) -> bool {
    let one = Candidate {
        k: 1,
        tokens: 1.0,
        slowest: 1.0,
        cost: StepCost { ms: 1.0, j: None },
    };
    let two = Candidate {
        k: 2,
        tokens: tr,
        slowest: tr,
        cost: StepCost {
            ms: p.enter,
            j: None,
        },
    };
    let margins = Margins {
        deeper: 0.0,
        shallower: p.enter / p.leave - 1.0,
        suspend: 0.0,
    };
    let incumbent = Some(if at_depth { 2 } else { 1 });
    choose(&Objective::Throughput, &margins, incumbent, &[one, two]) == Some(2)
}

impl RungParams {
    fn fast(&self) -> AcceptParams {
        AcceptParams {
            decay: 1.0 - self.alpha,
            prior_weight: 0.0,
            cold_weight: 0.0,
            seed: SeedWeight::SteadyState,
        }
    }

    fn probe(&self) -> DeepProbe {
        DeepProbe {
            backstop: self.probe_ticks,
            rise: self.p1_trigger,
            slow: AcceptParams {
                decay: 1.0 - self.alpha_slow,
                prior_weight: 0.0,
                cold_weight: 0.0,
                seed: SeedWeight::SteadyState,
            },
        }
    }
}

/// 2026-10-10: The rung's state: the fast estimates the decision reads (position 1 = `p1`,
/// position 2 = `p2_cond`), the probe trigger, the state and its flip count.
#[derive(Default)]
struct Ctl {
    fast: AcceptCounts,
    probe: DeepProbeState,
    at_depth: bool,
    flips: u64,
}

/// 2026-09-25: The controller. Each scheduler context owns one
/// (`SchedCtx::rung`), so accept history is not shared between contexts.
pub struct AdaptiveRung {
    params: RungParams,
    ctl: Mutex<Ctl>,
    /// 2026-09-25: The last `engaged` passed to `note_width_regime`; starts
    /// `true`.
    width_engaged: AtomicBool,
    /// 2026-09-25: Number of changes of `width_engaged`.
    width_flips: AtomicU64,
}

impl AdaptiveRung {
    pub fn new(params: RungParams) -> Self {
        Self {
            params,
            ctl: Mutex::new(Ctl::default()),
            width_engaged: AtomicBool::new(true),
            width_flips: AtomicU64::new(0),
        }
    }

    pub fn from_env() -> Self {
        Self::new(RungParams::from_env())
    }

    pub fn params(&self) -> &RungParams {
        &self.params
    }

    /// 2026-09-25: `true` while the controller's state is two drafts.
    pub fn at_depth(&self) -> bool {
        self.ctl.lock().at_depth
    }

    /// 2026-09-25: Number of state changes so far.
    pub fn flips(&self) -> u64 {
        self.ctl.lock().flips
    }

    pub fn width_engaged(&self) -> bool {
        self.width_engaged.load(Ordering::Relaxed)
    }

    pub fn width_flips(&self) -> u64 {
        self.width_flips.load(Ordering::Relaxed)
    }

    /// 2026-09-25: Feeds one accept-statistics flush at batch width `n`:
    /// `k_drafts` is the flush's largest draft depth, `p1` its first-draft
    /// accept rate, `mean_na` its mean accepted drafts per verify. The
    /// scheduler's `AcceptBuckets::record` is the caller; this type keeps no
    /// accept counters of its own.
    pub fn observe(&self, n: usize, k_drafts: usize, p1: f64, mean_na: f64) {
        let p = &self.params;
        if p.disabled || !BAND.contains(&n) {
            return;
        }
        let (fast, probe) = (p.fast(), p.probe());
        let mut c = self.ctl.lock();
        c.fast.observe_rate(&fast, 1, p1);
        // 2026-09-25: A depth flush observes p2_cond and resets both probe
        // triggers read by `drafts_for`.
        let deep = k_drafts >= 2;
        c.probe.observe(&probe, p1, deep);
        if !deep {
            return;
        }
        let Some(p2) = p2_cond_from(p1, mean_na) else {
            return;
        };
        c.fast.observe_rate(&fast, 2, p2);
        let (p1_e, p2_e) = (c.fast.rate(1).unwrap_or(0.0), c.fast.rate(2).unwrap_or(0.0));
        let tr = token_ratio(p1_e, p2_e);
        let was = c.at_depth;
        let now = next_state(was, tr, p);
        if now != was {
            c.at_depth = now;
            c.flips += 1;
            tracing::info!(
                "MTP rung n={n} -> k_drafts={} (token_ratio={tr:.4} p1={p1_e:.3} \
                 p2_cond={p2_e:.3} enter={:.3} leave={:.3} flips={})",
                if now { 2 } else { 1 },
                p.enter,
                p.leave,
                c.flips,
            );
        }
    }

    /// 2026-09-25: The draft count for a batch of `n_active`: the static
    /// ladder's count, raised to `min(2, num_drafts)` inside the adapted band
    /// while the state is two drafts or a probe is due.
    pub fn drafts_for(&self, n_active: usize, num_drafts: usize) -> usize {
        let p = &self.params;
        let base = metrale_model_layers::speculative::mtp_ladder_drafts(n_active, num_drafts);
        if p.disabled
            || metrale_model_layers::speculative::mtp_ladder_disabled()
            || num_drafts < 2
            || !BAND.contains(&n_active)
        {
            return base;
        }
        // 2026-09-25: A probe is due when p2_cond has never been observed,
        // when the slow p1 has risen by `p1_trigger` since the last depth
        // flush, or when `probe_ticks` flushes have passed since it
        // (`DeepProbeState::due`).
        let c = self.ctl.lock();
        if c.at_depth || c.probe.due(&p.probe(), c.fast.observed(2)) {
            return 2.min(num_drafts).max(base);
        }
        base
    }

    /// 2026-09-25: Records the width decision the scheduler already took:
    /// `engaged` is `n_active <= cap`, where `cap` is the run's
    /// `mtp_max_seqs` lever (32 unless overridden). It logs one INFO line per
    /// change and counts changes in `width_flips`; it decides nothing.
    pub fn note_width_regime(&self, n_active: usize, engaged: bool, cap: usize) {
        if self.width_engaged.swap(engaged, Ordering::Relaxed) == engaged {
            return;
        }
        let flips = self.width_flips.fetch_add(1, Ordering::Relaxed) + 1;
        if engaged {
            tracing::info!(
                "speculation ENGAGED at width n={n_active} (dispatch cap {cap}) — flips={flips}"
            );
        } else {
            tracing::info!(
                "speculation DISENGAGED at width n={n_active} > dispatch cap {cap}: this width \
                 plain-decodes (--mtp-max-seqs or MODEL.toml mtp_max_seqs raises the cap; the verify pools grow with it) \
                 — flips={flips}"
            );
        }
    }
}

#[cfg(test)]
#[path = "adaptive_rung_tests.rs"]
mod tests;
