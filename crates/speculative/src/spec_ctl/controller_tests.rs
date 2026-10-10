// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The controller configured as the per-stream DFlash draft count of the GLM
//! campaign (`--dflash-adaptive-k`: a `k:ms` step table, Latency, decay 0.95, prior weight 4,
//! explore every 16, suspend margin 3%): its decisions on that controller's own test inputs,
//! then suspension, re-probe, calibration and the batch decision.
//!
//! Owner: speculative.
//! Invariants: none beyond the types.

use super::super::accept::{ColdPrior, SeedWeight};
use super::super::calib::Calibration;
use super::super::chain::{MAX_POSITIONS, conditional_from_marginal};
use super::super::cost::{CostSource, StepTable};
use super::super::reprobe::ExploreState;
use super::super::source::{DraftCost, DraftKind};
use super::*;

const EXPLORE: u32 = 16;

fn cfg(resume: Option<u32>) -> ControllerConfig {
    ControllerConfig {
        source: DraftSource {
            kind: DraftKind::DFlash,
            max_k: 15,
            prefix_stable: true,
            draft_cost: DraftCost::PerBlock { ms: 10.0, j: 0.0 },
        },
        accept: AcceptParams {
            decay: 0.95,
            prior_weight: 4.0,
            cold_weight: 0.0,
            seed: SeedWeight::Unit,
        },
        objective: Objective::Latency,
        margins: Margins {
            deeper: 0.0,
            shallower: 0.0,
            suspend: 0.03,
        },
        reprobe: ReprobePolicy {
            explore_every: Some(EXPLORE),
            explore_max: EXPLORE,
            resume_after_tokens: resume,
            probe_steps: 0,
            soften: 0.25,
        },
    }
}

fn ctl(table: &str) -> SpecController {
    let cost = CostModel {
        source: CostSource::StepTable(StepTable::parse(table).unwrap()),
        calib: Calibration::new(0.1),
    };
    SpecController::new(cfg(Some(256)), cost, ColdPrior::None)
}

/// 2026-10-10: A stream whose counts reproduce conditional `rates` exactly at positions
/// `1..=rates.len()` (one decayed observation per position does it: rate = acc / obs).
fn stream_at(rates: &[f64]) -> SeqState {
    let p = AcceptParams {
        decay: 0.5,
        prior_weight: 0.0,
        cold_weight: 0.0,
        seed: SeedWeight::Unit,
    };
    let mut s = SeqState::default();
    for (i, &r) in rates.iter().enumerate() {
        s.rates.observe_rate(&p, i + 1, r);
    }
    s
}

/// 2026-10-10: At the reference acceptance (0.611 / 0.312 / 0.117 / 0.028 cumulative) and a
/// 6 ms per-row step slope two drafts win; at a flat 0.2 plain decode wins; without a plain
/// entry the best count is kept; the cap and the widest priced count bound the choice.
#[test]
fn the_choice_maximises_tokens_per_ms_and_suspends_at_a_loss() {
    let c = ctl("0:32,1:44,2:50,3:56,4:62");
    let reference = conditional_from_marginal(&[0.611, 0.312, 0.117, 0.028]);
    let s = stream_at(&reference);
    assert_eq!(c.drafts(&mut s.clone(), 1, 7), 2);
    assert_eq!(c.drafts(&mut s.clone(), 1, 1), 1);
    assert_eq!(c.drafts(&mut stream_at(&[0.2]), 1, 7), 0);
    assert_eq!(ctl("1:44,2:50").drafts(&mut stream_at(&[0.2]), 1, 7), 1);
    assert_eq!(
        c.drafts(&mut stream_at(&[0.95]), 1, 7),
        4,
        "capped by the widest priced count"
    );
}

/// 2026-10-10: Plain decode must win by more than 3%: one draft at 2% below the plain rate
/// keeps speculating, at 4% it suspends.
#[test]
fn a_near_tie_keeps_speculating() {
    let c = ctl("0:32,1:44");
    let tie = |gap: f64| {
        let tput = (1.0 - gap) / 32.0;
        c.drafts(&mut stream_at(&[tput * 44.0 - 1.0]), 1, 1)
    };
    assert_eq!(tie(0.02), 1);
    assert_eq!(tie(0.04), 0);
}

/// 2026-10-10: With nothing measured the cap stands; afterwards every 16th decision verifies
/// one draft more than the choice, within the cap; one stream's counts equal the prior's.
#[test]
fn the_controller_starts_at_the_cap_and_explores_one_deeper() {
    let mut c = ctl("0:32,1:44,2:50,3:56,4:62");
    let mut seq = SeqState::default();
    assert_eq!(c.drafts(&mut seq, 1, 5), 5);
    let mut seen = Vec::new();
    for _ in 0..EXPLORE {
        c.observe_step(&mut seq, 3, 1);
        seen.push(c.drafts(&mut seq, 1, 3));
    }
    let mut want = vec![1usize; EXPLORE as usize];
    want[EXPLORE as usize - 2] = 2;
    assert_eq!(seen, want);
    assert_eq!(seq.steps, EXPLORE);
    assert_eq!(
        c.global, seq.rates,
        "one stream: the prior holds the same counts"
    );
}

/// 2026-10-10: Committing 0 suspends; the stream plain-decodes for `resume_after_tokens`
/// tokens, then resumes with softened counts and decides again.
#[test]
fn a_suspended_stream_resumes_after_its_plain_tokens_with_softened_counts() {
    let mut c = ctl("0:32,1:44");
    c.cfg.reprobe.resume_after_tokens = Some(4);
    let mut s = stream_at(&[0.1]);
    assert_eq!(c.drafts(&mut s.clone(), 1, 1), 0);
    c.commit(&mut s, 0);
    assert!(s.suspended());
    assert_eq!(
        c.drafts(&mut stream_at(&[0.99]), 1, 1),
        1,
        "another stream is not suspended"
    );
    let before = s.rates.clone();
    let resumed: Vec<bool> = (0..4).map(|_| c.note_plain_token(&mut s)).collect();
    assert_eq!(resumed, vec![false, false, false, true]);
    assert!(!s.suspended());
    assert_eq!(s.rates.rate(1), before.rate(1), "softening keeps the rate");
    let mut probe = s.clone();
    c.observe_step(&mut probe, 1, 1);
    let mut unsoftened = SeqState {
        rates: before,
        ..SeqState::default()
    };
    c.observe_step(&mut unsoftened, 1, 1);
    assert!(
        probe.rates.rate(1).unwrap() > unsoftened.rates.rate(1).unwrap(),
        "after a re-probe one accept weighs more against the old evidence"
    );
}

/// 2026-10-10: A measured step moves the bucket's scale `alpha` of the way to its ratio, and
/// every priced count of that width follows it.
#[test]
fn a_measured_step_rescales_every_count() {
    let mut c = ctl("0:30,1:40");
    c.observe_cost(1, 1, 60.0, None);
    assert!((c.cost.calib.ms_scale(1) - 1.05).abs() < 1e-12);
    assert!((c.cost.cost(1, 0).unwrap().ms - 30.0 * 1.05).abs() < 1e-12);
    c.observe_cost(1, 5, 1e9, None);
    c.observe_cost(1, 1, f64::NAN, None);
    assert!(
        (c.cost.calib.ms_scale(1) - 1.05).abs() < 1e-12,
        "unpriced or invalid ignored"
    );
}

/// 2026-10-10: A batch decision sums the streams: two streams that each prefer a different
/// depth alone get one depth that maximises their total.
#[test]
fn a_batch_decision_maximises_the_sum_over_streams() {
    let mut c = ctl("0:32,1:44,2:50,3:56,4:62");
    c.cfg.objective = Objective::Throughput;
    let (rich, poor) = (stream_at(&[0.95]), stream_at(&[0.3]));
    assert_eq!(c.drafts(&mut rich.clone(), 1, 4), 4);
    assert_eq!(c.drafts(&mut poor.clone(), 1, 4), 0);
    let k = c.batch_drafts(&[&rich, &poor], &mut ExploreState::default(), 4);
    let total = |k: usize| {
        [0.95f64, 0.3]
            .iter()
            .map(|&p| super::super::chain::expected_tokens(&[p], k))
            .sum::<f64>()
            / c.cost.cost(2, k).unwrap().ms
    };
    assert!(
        (0..=4).all(|j| total(k) >= total(j)),
        "k {k} is not the batch optimum"
    );
    assert_eq!(
        c.batch_drafts(&[&SeqState::default()], &mut ExploreState::default(), 3),
        3,
        "no data: the cap"
    );
    assert_eq!(MAX_POSITIONS, 16);
}

/// 2026-10-10: A resumed stream speculates for `probe_steps` steps even where plain decode
/// still looks better, then may suspend again.
#[test]
fn a_re_probe_lasts_its_window() {
    let mut c = ctl("0:32,1:44");
    c.cfg.reprobe.resume_after_tokens = Some(1);
    c.cfg.reprobe.probe_steps = 3;
    let mut s = stream_at(&[0.1]);
    c.commit(&mut s, 0);
    assert!(c.note_plain_token(&mut s));
    let mut ks = Vec::new();
    for _ in 0..4 {
        let k = c.drafts(&mut s, 1, 1);
        ks.push(k);
        c.commit(&mut s, k);
        if k > 0 {
            c.observe_step(&mut s, 1, 0);
        }
    }
    assert_eq!(ks, vec![1, 1, 1, 0]);
}

/// 2026-10-10: With measured-only costs every depth is run once before anything is planned
/// from it, shallowest first; running plain decode as a cost probe does not suspend.
#[test]
fn an_online_source_probes_each_depth_and_a_plain_probe_does_not_suspend() {
    let mut c = ctl("0:32,1:44");
    c.cost.source = CostSource::Online(super::super::online::OnlineTable::new(0.3, 64, 1, 2.0));
    let mut s = stream_at(&[0.9]);
    let mut ks = Vec::new();
    for ms in [32.0, 44.0, 0.0] {
        let k = c.drafts(&mut s, 1, 1);
        c.commit(&mut s, k);
        assert!(!s.suspended(), "a probe of depth {k} suspended the stream");
        ks.push(k);
        if ms > 0.0 {
            c.observe_cost(1, k, ms, None);
        }
    }
    assert_eq!(
        ks,
        vec![0, 1, 1],
        "probe 0, probe 1, then choose 1 (1.9/44 > 1/32)"
    );
}
