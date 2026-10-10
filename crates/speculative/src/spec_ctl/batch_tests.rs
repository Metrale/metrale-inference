// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The batch decision that replaced the MTP gate and the DFlash gamma resolver,
//! driven through the calls the scheduler makes, with synthetic step walls: what the gate's
//! tests pinned (plain decode wins only by its margin, a width change re-measures, plain decode
//! is re-probed) and what the resolver's did (a single stream goes deep when its drafts are
//! accepted and shallow when they are not).
//!
//! Owner: speculative.
//! Invariants: none beyond the types.

use super::super::accept::{AcceptParams, ColdPrior, SeedWeight};
use super::super::calib::Calibration;
use super::super::controller::ControllerConfig;
use super::super::cost::{CostModel, CostSource};
use super::super::decide::{Margins, Objective};
use super::super::online::OnlineTable;
use super::super::reprobe::ReprobePolicy;
use super::super::source::{DraftCost, DraftKind, DraftSource};
use super::*;

fn host(max_k: usize) -> BatchSpec {
    let cfg = ControllerConfig {
        source: DraftSource {
            kind: DraftKind::Mtp,
            max_k,
            prefix_stable: true,
            draft_cost: DraftCost::PerDraft { ms: 0.0, j: 0.0 },
        },
        accept: AcceptParams {
            decay: 0.95,
            prior_weight: 4.0,
            cold_weight: 8.0,
            seed: SeedWeight::Unit,
        },
        objective: Objective::Throughput,
        margins: Margins {
            deeper: 0.0,
            shallower: 0.0,
            suspend: 0.05,
        },
        reprobe: ReprobePolicy {
            explore_every: Some(32),
            explore_max: 512,
            resume_after_tokens: Some(64),
            probe_steps: 8,
            soften: 0.25,
        },
    };
    let cost = CostModel {
        source: CostSource::Online(OnlineTable::new(0.3, 256)),
        calib: Calibration::new(0.0),
    };
    BatchSpec::new(SpecController::new(cfg, cost, ColdPrior::Chained))
}

/// 2026-10-10: Runs `steps` steps of `streams` among `allowed`, each verified stream accepting
/// `accept(k)` of `k` drafts and each step costing `ms(k)`. Returns the depths run.
fn drive(
    h: &mut BatchSpec,
    streams: &mut [SeqState],
    allowed: &[usize],
    steps: usize,
    mut accept: impl FnMut(usize) -> usize,
    ms: impl Fn(usize) -> f64,
) -> Vec<usize> {
    let mut ran = Vec::new();
    for _ in 0..steps {
        let refs: Vec<&SeqState> = streams.iter().collect();
        let d = h.decide(&refs, allowed);
        h.settle(streams.iter_mut(), d.k);
        let mut emitted = 0;
        for s in streams.iter_mut() {
            if d.k == 0 {
                h.observe_plain(s);
                emitted += 1;
            } else {
                let a = accept(d.k);
                h.observe_stream(s, d.k, a);
                emitted += 1 + a;
            }
        }
        h.observe_step(streams.len(), d.k, ms(d.k), None, emitted);
        ran.push(d.k);
    }
    ran
}

/// 2026-10-10: Both depths are measured first (plain decode, then speculation); with drafts
/// that pay, the batch speculates.
#[test]
fn both_depths_are_measured_then_the_paying_one_runs() {
    let mut h = host(3);
    let mut s = vec![SeqState::default(); 2];
    let ran = drive(
        &mut h,
        &mut s,
        &[0, 3],
        40,
        |_| 2,
        |k| 30.0 + 4.0 * k as f64,
    );
    assert_eq!(
        &ran[..2],
        &[0, 3],
        "probe plain, then the speculative depth"
    );
    assert!(ran[2..].iter().all(|&k| k == 3), "{ran:?}");
    assert_eq!(h.probes(), 2);
}

/// 2026-10-10: Drafts that never pay: plain decode wins, the switch is reported once, and
/// plain decode is re-probed after `resume_after_tokens` plain tokens for a re-probe window.
#[test]
fn plain_decode_wins_when_drafts_do_not_pay_and_is_re_probed() {
    let mut h = host(3);
    let mut s = [SeqState::default()];
    let mut entered = 0;
    let mut ran = Vec::new();
    for _ in 0..200 {
        let refs: Vec<&SeqState> = s.iter().collect();
        let d = h.decide(&refs, &[0, 3]);
        entered += usize::from(d.entered_plain);
        h.settle(s.iter_mut(), d.k);
        if d.k == 0 {
            h.observe_plain(&mut s[0]);
        } else {
            h.observe_stream(&mut s[0], 3, 0);
        }
        h.observe_step(1, d.k, if d.k == 0 { 30.0 } else { 45.0 }, None, 1);
        ran.push(d.k);
    }
    let spec = ran.iter().filter(|&&k| k > 0).count();
    assert!(
        spec < 50,
        "speculated {spec} of 200 steps on drafts that never pay"
    );
    assert!(spec >= 8, "never re-probed: {ran:?}");
    assert!(
        entered >= 2,
        "each return to plain decode is reported: {entered}"
    );
    let first_plain = ran.iter().skip(2).position(|&k| k == 0).unwrap() + 2;
    let window: Vec<usize> = ran[first_plain..first_plain + 64].to_vec();
    assert!(
        window.iter().all(|&k| k == 0),
        "suspended for 64 plain tokens: {window:?}"
    );
}

/// 2026-10-10: A near tie keeps speculating: plain decode 3% faster per token does not beat
/// the 5% suspend margin; 10% does.
#[test]
fn plain_decode_must_win_by_its_margin() {
    for (gap, want_spec) in [(0.03, true), (0.10, false)] {
        let mut h = host(1);
        let mut s = vec![SeqState::default()];
        // 2026-10-10: Every draft accepted: 2 tokens per speculative step; its wall is set so
        // plain decode is `gap` faster per token.
        let spec_ms = 2.0 * 30.0 * (1.0 + gap);
        let ran = drive(
            &mut h,
            &mut s,
            &[0, 1],
            60,
            |k| k,
            |k| if k == 0 { 30.0 } else { spec_ms },
        );
        let tail_spec = ran[10..].iter().all(|&k| k == 1);
        assert_eq!(tail_spec, want_spec, "gap {gap}: {ran:?}");
    }
}

/// 2026-10-10: Costs are per power-of-two width: a batch that grows from 2 to 4 streams
/// re-measures both depths before planning.
#[test]
fn a_new_width_bucket_is_measured_again() {
    let mut h = host(3);
    let mut s = vec![SeqState::default(); 2];
    drive(&mut h, &mut s, &[0, 3], 10, |_| 2, |k| 30.0 + k as f64);
    let before = h.probes();
    let mut s4 = vec![SeqState::default(); 4];
    let ran = drive(&mut h, &mut s4, &[0, 3], 3, |_| 2, |k| 30.0 + k as f64);
    assert_eq!(h.probes() - before, 2, "{ran:?}");
    assert_eq!(&ran[..2], &[0, 3]);
}

/// 2026-10-10: The resolver's case: one stream, depths 0..=7. Drafts that are all accepted
/// (code) go deep; drafts accepted about one in four (prose) stay shallow.
#[test]
fn a_single_stream_goes_deep_on_code_and_shallow_on_prose() {
    let allowed: Vec<usize> = (0..=7).collect();
    let ms = |k: usize| if k == 0 { 28.0 } else { 38.0 + 5.0 * k as f64 };
    let mut code = host(7);
    let ran = drive(
        &mut code,
        &mut [SeqState::default()],
        &allowed,
        200,
        |k| k,
        ms,
    );
    let deep = ran[100..].iter().filter(|&&k| k >= 6).count();
    assert!(deep >= 90, "code: {:?}", &ran[100..]);
    let mut prose = host(7);
    let mut n = 0usize;
    let ran = drive(
        &mut prose,
        &mut [SeqState::default()],
        &allowed,
        200,
        |k| {
            n += 1;
            usize::from(n.is_multiple_of(4)).min(k)
        },
        ms,
    );
    assert!(
        ran[100..].iter().all(|&k| k <= 1),
        "prose: {:?}",
        &ran[100..]
    );
}

/// 2026-10-10: Speculation that pays (2 tokens per step at the plain step's wall) keeps plain
/// decode a small share over a 2048-token stream: only the cost re-measures run it.
#[test]
fn paying_speculation_keeps_plain_decode_a_small_share() {
    let mut h = host(1);
    let ran = drive(
        &mut h,
        &mut [SeqState::default()],
        &[0, 1],
        1024,
        |k| k,
        |_| 50.0,
    );
    let plain = ran.iter().filter(|&&k| k == 0).count();
    assert!(
        (plain as f64) < 0.02 * ran.len() as f64,
        "plain {plain} of {}",
        ran.len()
    );
}
