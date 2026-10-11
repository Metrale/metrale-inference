// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: Per-position confidence stop for MTP draft chains
//! (`--draft-confidence-stop <tau>`): the drafter keeps extending a chain only
//! while its last draft's top-1 probability is at least `tau`. The serve
//! publishes `tau` once; the drafter (single and batched propose) and the
//! scheduler's depth planner read it here and apply the one rule
//! [`chain_depth`].
//!
//! Owner: model-layers (speculative).
//! 2026-10-11: Two forms of the one flag ([`DraftStopSpec`]): a plain `tau` is
//! the per-position rule above, unchanged; `cum:tau` stops a chain once the
//! product of its drafts' top-1 probabilities (the sum of their log-probabilities)
//! falls below `tau`. Callers apply the published [`StopRule`] through
//! [`StopRule::chain`] (drafting) and [`StopRule::depth`] (planning), so both
//! forms have one implementation each.
//!
//! Invariants:
//! - A published `tau` is in `(0, 1)`; a second publication of a different
//!   value or form is refused.
//! - Under the per-position form `StopRule` answers exactly what
//!   [`chain_continues`] and [`chain_depth`] answer.
//! - [`chain_depth`] never returns more than the drafts it is given, and
//!   returns at least 1 for a non-empty chain: the first draft is always
//!   verified, as without the stop.

static PUBLISHED: std::sync::OnceLock<DraftStopSpec> = std::sync::OnceLock::new();

/// 2026-10-11: How a chain's confidence is judged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopMode {
    /// 2026-10-11: Each draft's own top-1 probability against `tau` (the plain form).
    PerPosition,
    /// 2026-10-11: The product of the chain's top-1 probabilities so far against `tau`
    /// (`cum:tau`).
    Cumulative,
}

/// 2026-10-11: A `--draft-confidence-stop` value: `tau`, or `cum:tau`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DraftStopSpec {
    pub tau: f32,
    pub mode: StopMode,
}

impl std::str::FromStr for DraftStopSpec {
    type Err = anyhow::Error;

    /// 2026-10-11: `0.6` is per-position, `cum:0.35` cumulative; `tau` must be in `(0, 1)`.
    fn from_str(s: &str) -> anyhow::Result<Self> {
        let (mode, t) = match s.trim().strip_prefix("cum:") {
            Some(t) => (StopMode::Cumulative, t),
            None => (StopMode::PerPosition, s.trim()),
        };
        let tau: f32 = t.trim().parse().map_err(|_| {
            anyhow::anyhow!("--draft-confidence-stop {s:?}: expected TAU or cum:TAU")
        })?;
        anyhow::ensure!(
            tau > 0.0 && tau < 1.0,
            "--draft-confidence-stop must be in (0, 1), got {tau}"
        );
        Ok(Self { tau, mode })
    }
}

impl std::fmt::Display for DraftStopSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.mode {
            StopMode::PerPosition => write!(f, "{}", self.tau),
            StopMode::Cumulative => write!(f, "cum:{}", self.tau),
        }
    }
}

/// 2026-10-11: The published stop, as drafters and the planner apply it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StopRule {
    pub ln_tau: f32,
    pub mode: StopMode,
}

impl StopRule {
    /// 2026-10-11: The per-position rule at `ln_tau`.
    pub fn per_position(ln_tau: f32) -> Self {
        Self {
            ln_tau,
            mode: StopMode::PerPosition,
        }
    }

    /// 2026-10-11: The cumulative rule at `ln_tau`.
    pub fn cumulative(ln_tau: f32) -> Self {
        Self {
            ln_tau,
            mode: StopMode::Cumulative,
        }
    }

    /// 2026-10-11: A fresh chain's stop state.
    pub fn chain(&self) -> ChainStop {
        ChainStop {
            rule: *self,
            sum: 0.0,
        }
    }

    /// 2026-10-11: How many leading drafts of a chain with log-probabilities `lps` to verify:
    /// up to and including the first draft where the chain stops; all of them if it never
    /// does. Per-position, exactly [`chain_depth`]. A NaN never stops a chain and, under the
    /// cumulative form, adds nothing.
    pub fn depth(&self, lps: &[f32]) -> usize {
        match self.mode {
            StopMode::PerPosition => chain_depth(lps, self.ln_tau),
            StopMode::Cumulative => {
                let mut c = self.chain();
                lps.iter()
                    .position(|&lp| !lp.is_nan() && !c.continues(lp))
                    .map_or(lps.len(), |j| j + 1)
            }
        }
    }
}

/// 2026-10-11: One chain's stop state, fed each draft's log-probability in order.
#[derive(Clone, Copy, Debug)]
pub struct ChainStop {
    rule: StopRule,
    sum: f32,
}

impl ChainStop {
    /// 2026-10-11: Whether the chain continues past a draft with log-probability `lp`.
    /// Per-position, exactly [`chain_continues`]; cumulative, whether the chain's summed
    /// log-probability is still at least `ln tau`. Callers skip a NaN (an unmeasured draft).
    pub fn continues(&mut self, lp: f32) -> bool {
        match self.rule.mode {
            StopMode::PerPosition => chain_continues(lp, self.rule.ln_tau),
            StopMode::Cumulative => {
                self.sum += lp;
                self.sum >= self.rule.ln_tau
            }
        }
    }
}

/// 2026-10-02: Publish the serve's stop threshold, before the model builds and
/// the scheduler starts. 2026-10-11: The per-position form ([`set_draft_stop`]).
pub fn set_draft_confidence_stop(tau: f32) -> anyhow::Result<()> {
    set_draft_stop(DraftStopSpec {
        tau,
        mode: StopMode::PerPosition,
    })
}

/// 2026-10-11: Publish the serve's stop, either form.
pub fn set_draft_stop(spec: DraftStopSpec) -> anyhow::Result<()> {
    let tau = spec.tau;
    anyhow::ensure!(
        tau > 0.0 && tau < 1.0,
        "--draft-confidence-stop must be in (0, 1), got {tau}"
    );
    let got = *PUBLISHED.get_or_init(|| spec);
    anyhow::ensure!(
        got.tau.to_bits() == tau.to_bits() && got.mode == spec.mode,
        "draft confidence stop already published as {got}, refusing {spec}"
    );
    Ok(())
}

/// 2026-10-02: The published threshold as a log-probability, `ln tau`; `None`
/// when the serve did not ask for the stop (and in unit tests and tools).
pub fn draft_stop_logprob() -> Option<f32> {
    PUBLISHED.get().map(|s| s.tau.ln())
}

/// 2026-10-11: The published stop as a rule; `None` without the flag.
pub fn draft_stop_rule() -> Option<StopRule> {
    PUBLISHED.get().map(|s| StopRule {
        ln_tau: s.tau.ln(),
        mode: s.mode,
    })
}

/// 2026-10-02: How many leading drafts of a chain to verify, from each draft's
/// top-1 log-probability `lps[j]` (in draft order): drafting continues past
/// draft `j` only while `lps[j] >= ln_tau`, so the chain ends with the first
/// draft below the threshold, which is still verified. An empty chain is 0.
pub fn chain_depth(lps: &[f32], ln_tau: f32) -> usize {
    lps.iter()
        .position(|&lp| lp < ln_tau)
        .map_or(lps.len(), |j| j + 1)
}

/// 2026-10-02: Whether a chain whose last draft scored `lp` continues.
pub fn chain_continues(lp: f32, ln_tau: f32) -> bool {
    lp >= ln_tau
}

#[cfg(test)]
mod tests {
    use super::*;

    const LN_HALF: f32 = -std::f32::consts::LN_2;

    #[test]
    fn a_confident_chain_keeps_every_draft() {
        assert_eq!(chain_depth(&[-0.1, -0.2, -0.3], LN_HALF), 3);
    }

    #[test]
    fn the_chain_ends_with_its_first_unconfident_draft() {
        assert_eq!(chain_depth(&[-0.1, -2.0, -0.1], LN_HALF), 2);
        assert_eq!(chain_depth(&[-2.0, -0.1, -0.1], LN_HALF), 1);
    }

    #[test]
    fn the_threshold_itself_continues() {
        assert!(chain_continues(LN_HALF, LN_HALF));
        assert!(!chain_continues(LN_HALF - 1e-6, LN_HALF));
        assert_eq!(chain_depth(&[LN_HALF, LN_HALF], LN_HALF), 2);
    }

    #[test]
    fn an_empty_chain_is_zero_and_a_lone_draft_is_one() {
        assert_eq!(chain_depth(&[], LN_HALF), 0);
        assert_eq!(chain_depth(&[-9.0], LN_HALF), 1);
    }

    #[test]
    fn depth_agrees_with_continuation_at_every_position() {
        // 2026-10-02: The single-sequence drafter stops with `chain_continues`,
        // the batched planner truncates with `chain_depth`: the same rule.
        let lps = [-0.05, -0.4, -0.3, -1.5, -0.01];
        let ln_tau = (0.6f32).ln();
        let mut drafted = 0;
        for &lp in &lps {
            drafted += 1;
            if !chain_continues(lp, ln_tau) {
                break;
            }
        }
        assert_eq!(drafted, chain_depth(&lps, ln_tau));
    }

    /// 2026-10-11: The per-position rule answers exactly what `chain_continues` and
    /// `chain_depth` answer (today's bits for every plain-number serve), over every chain of
    /// up to four drafts from a grid of log-probabilities around the threshold, NaN included.
    #[test]
    fn the_per_position_rule_is_the_plain_functions() {
        let ln_tau = (0.6f32).ln();
        let rule = StopRule::per_position(ln_tau);
        let grid = [-0.01f32, ln_tau, ln_tau - 1e-6, -0.9, -3.0, f32::NAN];
        let mut chains: Vec<Vec<f32>> = vec![vec![]];
        for _ in 0..4 {
            let next: Vec<Vec<f32>> = chains
                .iter()
                .flat_map(|c| {
                    grid.iter().map(move |&g| {
                        let mut c = c.clone();
                        c.push(g);
                        c
                    })
                })
                .collect();
            chains.extend(next.clone());
            chains.retain(|c| c.len() <= 4);
        }
        for c in &chains {
            assert_eq!(rule.depth(c), chain_depth(c, ln_tau), "{c:?}");
            let mut s = rule.chain();
            for &lp in c {
                assert_eq!(s.continues(lp), chain_continues(lp, ln_tau), "{c:?}");
            }
        }
    }

    /// 2026-10-11: The cumulative rule stops where the product of the probabilities falls
    /// below tau: drafts at 0.8 each make 0.8, 0.64, 0.512, then 0.41; at tau 0.5 the chain
    /// keeps three and verifies the fourth; a NaN neither stops nor counts.
    #[test]
    fn the_cumulative_rule_stops_on_the_product() {
        let p = (0.8f32).ln();
        let rule = StopRule::cumulative((0.5f32).ln());
        assert_eq!(rule.depth(&[p, p, p, p, p]), 4);
        assert_eq!(rule.depth(&[p, p]), 2);
        assert_eq!(rule.depth(&[p, f32::NAN, p, p, p, p]), 5);
        assert_eq!(rule.depth(&[(0.4f32).ln(), p]), 1);
        let mut c = rule.chain();
        assert!(c.continues(p));
        assert!(c.continues(p));
        assert!(c.continues(p));
        assert!(!c.continues(p));
        // 2026-10-11: The per-position rule at the same tau keeps all five 0.8 drafts.
        assert_eq!(
            StopRule::per_position((0.5f32).ln()).depth(&[p, p, p, p, p]),
            5
        );
    }

    /// 2026-10-11: The flag's two forms parse; anything else, or a tau outside (0, 1), is
    /// refused; Display round-trips.
    #[test]
    fn the_flag_parses_both_forms() {
        let a: DraftStopSpec = "0.6".parse().unwrap();
        assert_eq!(a.mode, StopMode::PerPosition);
        assert_eq!(a.tau, 0.6);
        let b: DraftStopSpec = "cum:0.35".parse().unwrap();
        assert_eq!(b.mode, StopMode::Cumulative);
        assert_eq!(b.tau, 0.35);
        for bad in ["", "cum:", "cum:1.0", "0", "x:0.5", "cum:abc", "1.5"] {
            assert!(bad.parse::<DraftStopSpec>().is_err(), "{bad:?} accepted");
        }
        for s in ["0.6", "cum:0.35"] {
            assert_eq!(s.parse::<DraftStopSpec>().unwrap().to_string(), s);
        }
    }

    #[test]
    fn publication_refuses_out_of_range_values() {
        for bad in [0.0, 1.0, -0.5, 1.5, f32::NAN] {
            assert!(set_draft_confidence_stop(bad).is_err(), "{bad} accepted");
        }
    }
}
