// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The paired A/B verdict on synthetic runs whose truth is known, shaped like the
//! measured serve (33 ms step, 1 ms of noise, a 2 ms drift over the request, 2.5% of steps +10..25
//! ms spikes, and an arm-blind 45 us speedup of each cycle's second step; cycles in the order or
//! its mirror at random, as the instrument runs them): an A/A run is equivalent at a 0.25% margin
//! and not superior, its half-width under 0.2%; a 100 us (0.3%) win is found and estimated
//! within the spikes; a cycle missing a step is dropped; a malformed line is an error.
//! Owner: metrale-bench tests.

use super::*;
use metrale_stats::rng::SplitMix64;

/// 2026-10-23: `requests` requests of `cycles` cycles each, shaped as the module doc says, minus
/// `win` us on arm 1.
fn run(requests: usize, cycles: usize, win: f64, seed: u64) -> Vec<Step> {
    let mut r = SplitMix64::new(seed);
    let mut out = Vec::new();
    let mut step = 0u64;
    let mut mirror = false;
    for q in 0..requests {
        for i in 0..cycles * 4 {
            if step.is_multiple_of(4) {
                mirror = r.next_f64() < 0.5;
            }
            let up = [0usize, 1, 1, 0][(step % 4) as usize];
            let arm = if mirror { 1 - up } else { up };
            let drift = 2000.0 * i as f64 / (cycles * 4) as f64;
            let spike = if r.next_f64() < 0.025 {
                10_000.0 + 15_000.0 * r.next_f64()
            } else {
                0.0
            };
            let position = if step % 4 == 1 { -45.0 } else { 0.0 };
            let us = 33_000.0 + drift + 1000.0 * r.normal() + spike + position
                - if arm == 1 { win } else { 0.0 };
            out.push(Step {
                step,
                arm,
                kv: 1000 + 100_000 * q + i,
                us,
            });
            step += 1;
        }
    }
    out
}

fn plan() -> Plan {
    Plan {
        arms: 2,
        warm: 2,
        margin: 0.0025,
        huber_k: 1.345,
        reps: 400,
        seed: 7,
    }
}

#[test]
fn an_a_a_run_is_equivalent_and_not_superior() {
    let v = judge(&run(2, 1500, 0.0, 1), plan()).unwrap();
    assert_eq!(v.len(), 2, "one verdict per request");
    for x in &v {
        assert!(x.d_lo < 0.0 && x.d_hi > 0.0, "the interval covers 0: {x:?}");
        assert!(x.equivalent, "equivalent at 0.25%: {x:?}");
        assert!(!x.superior_holm, "{x:?}");
        assert!(
            x.pct(x.d_hi - x.d_lo) / 2.0 < 0.2,
            "half-width under 0.2%: {x:?}"
        );
    }
}

#[test]
fn a_hundred_us_win_is_found_and_the_drift_cancels() {
    let v = judge(&run(2, 1500, 100.0, 2), plan()).unwrap();
    for x in &v {
        assert!(x.d_lo > 0.0, "B faster, interval above 0: {x:?}");
        assert!((x.d_us - 100.0).abs() < 60.0, "{x:?}");
        assert!(x.superior_holm, "{x:?}");
    }
}

#[test]
fn a_cycle_missing_a_step_is_dropped_and_warm_cycles_skipped() {
    let mut s = run(1, 40, 0.0, 3);
    s.remove(4 * 20 + 1);
    let v = judge(&s, plan()).unwrap();
    assert_eq!(v[0].cycles, 40 - 1 - 2);
}

#[test]
fn lines_are_parsed_and_a_malformed_one_is_an_error() {
    let text = "x INFO PAIRED_AB step=3 arm=1 levers=0x1 kv=1200 us=33012.500\n\
                x INFO PAIRED_AB on: levers [a], 2 arms\nnoise\n";
    assert_eq!(
        parse(text).unwrap(),
        vec![Step {
            step: 3,
            arm: 1,
            kv: 1200,
            us: 33012.5
        }]
    );
    assert!(parse("PAIRED_AB step=3 arm=1 kv=12").is_err());
    assert!(judge(&parse(text).unwrap(), Plan { arms: 1, ..plan() }).is_err());
}
