// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Re-probes: how a controller keeps measuring what its choice stopped measuring.
//! A depth the choice does not verify produces no acceptance data, so without re-probes a
//! stream whose acceptance changed would never be re-explored.
//!
//! - [`ReprobePolicy::explore_every`]: a stream's N-th decision verifies one draft more than the
//!   choice (within the cap); the next exploration comes N decisions later, doubling up to
//!   [`ReprobePolicy::explore_max`] while the choice it explored from stays the same, and back to
//!   N when it changes (a stationary stream pays less and less for exploring; a changing one is
//!   re-explored at once). `explore_max == explore_every` is a fixed cadence.
//! - [`ReprobePolicy::resume_after_tokens`] / [`ReprobePolicy::soften`]: a stream suspended to
//!   plain decode re-probes after that many plain tokens, its counts scaled by `soften` so the
//!   probe's steps outweigh the old evidence; the probe lasts
//!   [`ReprobePolicy::probe_steps`] speculative steps before plain decode may win again.
//! - [`DeepProbe`]: a serve-wide trigger for one deeper position: due when it was never
//!   observed, not observed for `backstop` observations, or when a slow estimate of the first
//!   position rose by `rise` since it was last observed (acceptance only rising can make depth
//!   newly worthwhile).
//!
//! Owner: speculative.
//! Invariants: pure state machines; the host calls them where it observes and decides.

use super::accept::{AcceptCounts, AcceptParams};

/// 2026-10-10: Per-stream re-probe settings. No defaults (PCND): each controller states them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReprobePolicy {
    pub explore_every: Option<u32>,
    /// 2026-10-10: The widest exploration interval the back-off reaches.
    pub explore_max: u32,
    pub resume_after_tokens: Option<u32>,
    /// 2026-10-10: Steps a resumed stream speculates before it may suspend again, so a
    /// re-probe measures a window, not one step.
    pub probe_steps: u32,
    pub soften: f64,
}

/// 2026-10-10: A stream's exploration schedule.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExploreState {
    /// 2026-10-10: The decision count at which the next exploration is due (`None`: the
    /// first, at `explore_every - 1`).
    next: Option<u32>,
    gap: u32,
    last_base: Option<usize>,
    /// 2026-10-10: The last decision was a cost probe, not a choice: running plain decode for
    /// it does not suspend the stream.
    pub(crate) probing: bool,
}

impl ReprobePolicy {
    /// 2026-10-10: `k` raised by one (within `cap`) when an exploration is due on the stream's
    /// `steps`-th decision and `k` is non-zero; advances the schedule when it explores.
    pub fn explore(&self, st: &mut ExploreState, k: usize, steps: u32, cap: usize) -> usize {
        let Some(n) = self.explore_every.filter(|&n| n > 0) else {
            return k;
        };
        if k == 0 || steps < st.next.unwrap_or(n - 1) {
            return k;
        }
        st.gap = if st.last_base == Some(k) {
            (st.gap.max(n) * 2).min(self.explore_max.max(n))
        } else {
            n
        };
        st.last_base = Some(k);
        st.next = Some(steps.saturating_add(st.gap));
        (k + 1).min(cap)
    }
}

/// 2026-10-10: The deeper-position probe trigger's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepProbe {
    /// 2026-10-10: Observations without a deep one after which a probe is due.
    pub backstop: u64,
    /// 2026-10-10: Rise of the slow first-position estimate that makes a probe due.
    pub rise: f64,
    /// 2026-10-10: The slow estimator.
    pub slow: AcceptParams,
}

/// 2026-10-10: [`DeepProbe`]'s state.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeepProbeState {
    tick: u64,
    last_deep: u64,
    slow: AcceptCounts,
    slow_at_deep: f64,
}

impl DeepProbeState {
    /// 2026-10-10: One observation whose first-position rate was `p1`; `deep` says it also
    /// verified the deeper position (which resets both triggers).
    pub fn observe(&mut self, cfg: &DeepProbe, p1: f64, deep: bool) {
        self.slow.observe_rate(&cfg.slow, 1, p1);
        self.tick += 1;
        if deep {
            self.last_deep = self.tick;
            self.slow_at_deep = self.slow.rate(1).unwrap_or(0.0);
        }
    }

    /// 2026-10-10: Whether a probe of the deeper position is due; `deep_observed` says the
    /// estimator of that position has any data.
    pub fn due(&self, cfg: &DeepProbe, deep_observed: bool) -> bool {
        let slow = self.slow.rate(1).unwrap_or(0.0);
        !deep_observed
            || slow - self.slow_at_deep >= cfg.rise
            || self.tick.saturating_sub(self.last_deep) >= cfg.backstop
    }
}

#[cfg(test)]
mod tests {
    use super::super::accept::SeedWeight;
    use super::*;

    #[test]
    fn exploration_adds_one_draft_on_its_cadence_within_the_cap() {
        let p = ReprobePolicy {
            explore_every: Some(4),
            explore_max: 4,
            resume_after_tokens: None,
            probe_steps: 0,
            soften: 1.0,
        };
        let mut st = ExploreState::default();
        let got: Vec<usize> = (0..12).map(|s| p.explore(&mut st, 2, s, 5)).collect();
        assert_eq!(
            got,
            vec![2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3],
            "fixed cadence"
        );
        assert_eq!(
            p.explore(&mut ExploreState::default(), 5, 3, 5),
            5,
            "capped"
        );
        assert_eq!(
            p.explore(&mut ExploreState::default(), 0, 3, 5),
            0,
            "plain never explores"
        );
        let off = ReprobePolicy {
            explore_every: None,
            ..p
        };
        assert_eq!(off.explore(&mut ExploreState::default(), 2, 3, 5), 2);
    }

    /// 2026-10-10: With back-off the interval doubles while the explored-from choice holds
    /// (4, 8, 16, capped at 16) and returns to 4 when it changes.
    #[test]
    fn exploration_backs_off_on_a_stable_choice_and_resets_on_a_change() {
        let p = ReprobePolicy {
            explore_every: Some(4),
            explore_max: 16,
            resume_after_tokens: None,
            probe_steps: 0,
            soften: 1.0,
        };
        let mut st = ExploreState::default();
        let at: Vec<u32> = (0..60)
            .filter(|&s| p.explore(&mut st, 2, s, 5) == 3)
            .collect();
        assert_eq!(at, vec![3, 7, 15, 31, 47]);
        let mut st = ExploreState::default();
        let mut at = Vec::new();
        for s in 0..40 {
            let k = if s < 20 { 2 } else { 1 };
            if p.explore(&mut st, k, s, 5) > k {
                at.push(s);
            }
        }
        assert_eq!(
            at,
            vec![3, 7, 15, 31, 35],
            "the change at 20 resets the gap at 31"
        );
    }

    /// 2026-10-10: Due while the deep position is unobserved; after a deep observation, due
    /// again only after `backstop` observations or a rise of `rise` in the slow estimate.
    #[test]
    fn a_deep_probe_is_due_when_unseen_stale_or_after_a_rise() {
        let cfg = DeepProbe {
            backstop: 10,
            rise: 0.08,
            slow: AcceptParams {
                decay: 0.85,
                prior_weight: 0.0,
                cold_weight: 0.0,
                seed: SeedWeight::SteadyState,
            },
        };
        let mut s = DeepProbeState::default();
        assert!(s.due(&cfg, false));
        s.observe(&cfg, 0.7, true);
        assert!(!s.due(&cfg, true));
        for _ in 0..9 {
            s.observe(&cfg, 0.7, false);
            assert!(!s.due(&cfg, true));
        }
        s.observe(&cfg, 0.7, false);
        assert!(s.due(&cfg, true), "backstop");
        s.observe(&cfg, 0.7, true);
        let mut n = 0;
        while !s.due(&cfg, true) {
            s.observe(&cfg, 0.95, false);
            n += 1;
        }
        assert!(
            n <= 3,
            "a rise of 0.25 is seen within a few observations, took {n}"
        );
        s.observe(&cfg, 0.95, true);
        s.observe(&cfg, 0.5, false);
        assert!(
            !s.due(&cfg, true),
            "a fall never makes depth newly worthwhile"
        );
    }
}
