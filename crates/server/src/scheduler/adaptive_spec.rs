// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Adaptive DFlash speculation, on when `METRALE_DFLASH_ADAPTIVE=1`.
//!
//! Policy: each sequence keeps the accept counts of its last [`WINDOW`] K=γ
//! verify steps. When the window is full and its mean is below
//! `METRALE_DFLASH_ADAPTIVE_MIN` (default 2.0), speculation is suspended for
//! that sequence. After `METRALE_DFLASH_ADAPTIVE_REPROBE` (default 256)
//! serially decoded tokens, [`spec_allowed`] lifts the suspension with an
//! empty window, so it can only re-trigger after [`WINDOW`] more verify steps.
//! The state is reset when a sequence is requeued or restored from spill.
//!
//! 2026-10-10: The single-stream adaptive draft count (`--dflash-adaptive-k`,
//! `metrale_speculative::dflash_adaptive_k`) keeps its per-sequence rates here (`k`) and
//! suspends through the same state ([`suspend`]), re-probing after the same
//! `METRALE_DFLASH_ADAPTIVE_REPROBE` tokens; the re-probe softens the rates
//! ([`RESUME_SOFTEN`]) so the probe's steps count against the old evidence.
//!
//! Owner: scheduler.
//! Invariants:
//! - `window` never holds more than [`WINDOW`] entries.
//! - Suspension is read and lifted when `METRALE_DFLASH_ADAPTIVE=1` or the adaptive draft
//!   count is installed ([`enabled`]); only the former trips it from the accept window.

use crate::scheduler::ActiveSeq;

/// 2026-09-25: Rolling accept window + suspend state, embedded in [`ActiveSeq`].
#[derive(Default)]
pub(crate) struct AdaptState {
    window: Vec<u32>,
    suspended: bool,
    serial_tokens: u32,
    /// 2026-10-10: The adaptive draft count's per-sequence rates and step counter.
    pub(crate) k: metrale_speculative::dflash_adaptive_k::SeqAdaptiveK,
}

const WINDOW: usize = 12;

/// 2026-10-10: Factor a re-probe applies to the adaptive draft count's rates.
pub(crate) const RESUME_SOFTEN: f64 = 0.25;

/// 2026-10-10: Suspension is in force: `METRALE_DFLASH_ADAPTIVE=1` or the adaptive draft
/// count installed.
fn enabled(sched: &crate::scheduler::sched_ctx::SchedCtx) -> bool {
    sched.levers.dflash_adaptive || sched.dflash_rung.adaptive_k_on()
}

/// 2026-10-10: Suspend this sequence's speculation until the re-probe (the adaptive draft
/// count chose plain decode). A no-op when suspension is not in force.
pub(crate) fn suspend(a: &mut ActiveSeq, sched: &crate::scheduler::sched_ctx::SchedCtx) {
    if !enabled(sched) {
        return;
    }
    let st = &mut a.spec_adapt;
    st.suspended = true;
    st.serial_tokens = 0;
    st.window.clear();
    tracing::info!("adaptive spec: SUSPENDED (adaptive draft count chose plain decode)");
}

/// 2026-09-25: Record one K=γ verify step's accept count; may trip suspension.
pub(crate) fn record_verify(
    a: &mut ActiveSeq,
    num_accepted: usize,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) {
    if !sched.levers.dflash_adaptive {
        return;
    }
    let st = &mut a.spec_adapt;
    st.window.push(num_accepted as u32);
    if st.window.len() > WINDOW {
        st.window.remove(0);
    }
    if st.window.len() == WINDOW {
        let mean = st.window.iter().sum::<u32>() as f32 / WINDOW as f32;
        if mean < sched.levers.dflash_adaptive_min {
            st.suspended = true;
            st.serial_tokens = 0;
            st.window.clear();
            tracing::info!(
                "adaptive spec: SUSPENDED (mean accepted {mean:.2} < {} over {WINDOW} steps) — \
                 serial decode until re-probe",
                sched.levers.dflash_adaptive_min,
            );
        }
    }
}

/// 2026-09-25: May this sequence propose/speculate right now? Un-suspends (re-probe)
/// once enough serial tokens have passed.
pub(crate) fn spec_allowed(
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) -> bool {
    if !enabled(sched) {
        return true;
    }
    let st = &mut a.spec_adapt;
    if !st.suspended {
        return true;
    }
    if st.serial_tokens >= sched.levers.dflash_adaptive_reprobe {
        st.suspended = false;
        st.serial_tokens = 0;
        st.window.clear();
        st.k.rates.soften(RESUME_SOFTEN);
        tracing::info!(
            "adaptive spec: RE-PROBING after {} serial tokens",
            sched.levers.dflash_adaptive_reprobe
        );
        return true;
    }
    false
}

/// 2026-09-25: Is this sequence currently in the adaptive-suspended (serial) regime?
/// Read-only peek — unlike `spec_allowed`, never mutates re-probe state.
pub(crate) fn is_suspended(a: &ActiveSeq, sched: &crate::scheduler::sched_ctx::SchedCtx) -> bool {
    enabled(sched) && a.spec_adapt.suspended
}

/// 2026-09-25: Count a serially-decoded token toward the re-probe interval.
pub(crate) fn tick_serial(a: &mut ActiveSeq, sched: &crate::scheduler::sched_ctx::SchedCtx) {
    if enabled(sched) && a.spec_adapt.suspended {
        a.spec_adapt.serial_tokens = a.spec_adapt.serial_tokens.saturating_add(1);
    }
}
