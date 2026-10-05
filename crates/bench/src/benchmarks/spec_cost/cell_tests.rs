// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Tests for the step-cost arithmetic and the snapshot reader.
//!
//! Owner: bench, spec-cost.
//! Invariants: none beyond the types.

use super::*;

fn counters(tokens: f64, step: Option<(f64, f64)>, propose: (f64, f64)) -> Counters {
    Counters {
        tokens,
        step: step.map(|(count, sum_s)| PhaseTotals { count, sum_s }),
        propose: PhaseTotals {
            count: propose.0,
            sum_s: propose.1,
        },
        energy_mj: None,
    }
}

/// 2026-10-04: `c` with the NVML energy counter at `mj`.
fn at_mj(c: Counters, mj: f64) -> Counters {
    Counters {
        energy_mj: Some(mj),
        ..c
    }
}

/// 2026-10-04: n = 4 at k = 3 over 8 s: 200 steps of 30 ms, of which 6 ms is propose,
/// 2000 tokens (2.5 per sequence per step), 800 J on the rail integral and 840 J on the
/// serve's energy counter.
fn speculative() -> Window {
    Window {
        n: 4,
        k: 3,
        before: at_mj(
            counters(1000.0, Some((100.0, 3.0)), (100.0, 0.6)),
            5_000_000.0,
        ),
        after: at_mj(
            counters(3000.0, Some((300.0, 9.0)), (300.0, 1.8)),
            5_840_000.0,
        ),
        window_s: 8.0,
        energy_j: Some(800.0),
        ended_early: 0,
    }
}

fn measured(w: &Window) -> StepCost {
    match evaluate(w) {
        CellVerdict::Measured(c) => c,
        CellVerdict::Vacuous(why) => panic!("vacuous: {why}"),
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

#[test]
fn speculative_cell_splits_time_from_the_phase_histograms() {
    let c = measured(&speculative());
    assert!(close(c.steps, 200.0));
    assert!(close(c.step_ms, 30.0));
    assert!(close(c.draft_ms, 6.0));
    assert!(close(c.verify_ms, 24.0));
    assert!(close(c.wall_ms, 40.0), "8000 ms over 200 steps");
    assert!(close(c.tok_per_step, 2.5));
}

#[test]
fn energy_splits_in_proportion_to_time() {
    let c = measured(&speculative());
    // 2026-10-04: 800 J / 200 steps = 4 J per step; propose is 6/30 of the step time.
    assert!(close(c.draft_j.unwrap(), 0.8));
    assert!(close(c.verify_j.unwrap(), 3.2));
    assert!(close(c.draft_j.unwrap() + c.verify_j.unwrap(), 4.0));
}

#[test]
fn without_a_rail_reading_the_joule_fields_are_absent_and_not_recorded() {
    let s = speculative();
    let w = Window {
        energy_j: None,
        ..s
    };
    let c = measured(&w);
    assert_eq!((c.draft_j, c.verify_j), (None, None));
    let mut m = BTreeMap::new();
    record(4, &CellVerdict::Measured(c), &mut m);
    assert!(m.contains_key("n4_verify_ms"));
    assert!(!m.contains_key("n4_verify_j") && !m.contains_key("n4_draft_j"));
}

#[test]
fn k0_counts_one_step_per_token_per_sequence_and_has_no_draft() {
    let w = Window {
        n: 8,
        k: 0,
        before: at_mj(counters(500.0, None, (0.0, 0.0)), 1_000.0),
        after: at_mj(counters(2100.0, None, (0.0, 0.0)), 401_000.0),
        window_s: 8.0,
        energy_j: Some(400.0),
        ended_early: 0,
    };
    let c = measured(&w);
    assert!(close(c.steps, 200.0), "1600 tokens over 8 sequences");
    assert!(close(c.step_ms, 40.0));
    assert!(close(c.verify_ms, 40.0));
    assert_eq!(c.draft_ms, 0.0);
    assert_eq!(c.draft_j, Some(0.0));
    assert!(close(c.verify_j.unwrap(), 2.0));
    assert!(close(c.tok_per_step, 1.0));
}

#[test]
fn fewer_than_min_steps_is_vacuous() {
    let mut w = speculative();
    w.after.step = Some(PhaseTotals {
        count: 119.0,
        sum_s: 3.5,
    });
    assert!(matches!(evaluate(&w), CellVerdict::Vacuous(why) if why.contains("fewer than")));
    w.after.step = Some(PhaseTotals {
        count: 120.0,
        sum_s: 3.6,
    });
    w.after.propose = PhaseTotals {
        count: 120.0,
        sum_s: 0.7,
    };
    assert!(matches!(evaluate(&w), CellVerdict::Measured(c) if close(c.steps, 20.0)));
    let k0 = Window {
        n: 2,
        k: 0,
        before: counters(0.0, None, (0.0, 0.0)),
        after: counters(39.0, None, (0.0, 0.0)),
        ..speculative()
    };
    assert!(
        matches!(evaluate(&k0), CellVerdict::Vacuous(_)),
        "19.5 steps"
    );
}

#[test]
fn a_stream_that_ended_inside_the_window_makes_the_cell_vacuous() {
    let w = Window {
        ended_early: 1,
        ..speculative()
    };
    let v = evaluate(&w);
    assert!(matches!(&v, CellVerdict::Vacuous(why) if why.contains("1 of 4")));
    let mut m = BTreeMap::new();
    record(4, &v, &mut m);
    assert_eq!(m, BTreeMap::from([("n4_vacuous".to_string(), 1.0)]));
}

#[test]
fn an_untimed_propose_phase_is_vacuous_not_a_zero_draft_cost() {
    let mut w = speculative();
    w.after.propose = w.before.propose;
    assert!(matches!(evaluate(&w), CellVerdict::Vacuous(why) if why.contains("propose")));
}

#[test]
fn propose_longer_than_the_step_is_vacuous() {
    let mut w = speculative();
    w.after.propose.sum_s = w.before.propose.sum_s + 6.1;
    assert!(matches!(evaluate(&w), CellVerdict::Vacuous(why) if why.contains("exceeds")));
}

#[test]
fn counters_that_go_backwards_are_vacuous() {
    let mut w = speculative();
    w.after.tokens = 10.0;
    assert!(matches!(evaluate(&w), CellVerdict::Vacuous(why) if why.contains("restarted")));
    let mut w = speculative();
    w.after.step = Some(PhaseTotals {
        count: 50.0,
        sum_s: 1.0,
    });
    assert!(matches!(evaluate(&w), CellVerdict::Vacuous(why) if why.contains("restarted")));
}

#[test]
fn measured_cell_records_every_key() {
    let mut m = BTreeMap::new();
    record(4, &evaluate(&speculative()), &mut m);
    let keys: Vec<&str> = m.keys().map(String::as_str).collect();
    assert_eq!(
        keys,
        [
            "n4_draft_j",
            "n4_draft_ms",
            "n4_nvml_j",
            "n4_steps",
            "n4_tok_per_step",
            "n4_vacuous",
            "n4_verify_j",
            "n4_verify_ms",
            "n4_wall_ms",
        ]
    );
    assert_eq!(m["n4_vacuous"], 0.0);
}

#[test]
fn snapshot_reads_live_tokens_and_both_phases() {
    let page = "\
metrale_generation_tokens_total 7
metrale_decoded_tokens_total 4000
metrale_sched_phase_seconds_sum{phase=\"step_mtp\"} 2.5
metrale_sched_phase_seconds_count{phase=\"step_mtp\"} 100
metrale_gpu_energy_counter_millijoules 999
metrale_gpu_energy_millijoules_total 123456
";
    let c = Counters::read(&Scrape::parse(page).unwrap()).unwrap();
    assert_eq!(
        c.tokens, 4000.0,
        "the live counter, not the per-request one"
    );
    assert_eq!(
        c.step,
        Some(PhaseTotals {
            count: 100.0,
            sum_s: 2.5
        })
    );
    assert_eq!(c.propose, PhaseTotals::default(), "absent propose is zero");
    assert_eq!(
        c.energy_mj,
        Some(123456.0),
        "the corrected total, not the raw counter"
    );
    require_step_series(3, &c).unwrap();
}

#[test]
fn a_speculative_run_without_phase_timing_is_refused() {
    let c = Counters::read(&Scrape::parse("metrale_decoded_tokens_total 1\n").unwrap()).unwrap();
    let err = require_step_series(2, &c).unwrap_err().to_string();
    assert!(err.contains("--telemetry basic"), "{err}");
    require_step_series(0, &c).unwrap();
}

#[test]
fn a_page_without_the_token_counter_or_with_half_a_histogram_is_an_error() {
    assert!(
        Counters::read(&Scrape::parse("metrale_generation_tokens_total 1\n").unwrap()).is_err()
    );
    let half = "metrale_decoded_tokens_total 1\n\
                metrale_sched_phase_seconds_count{phase=\"propose\"} 3\n";
    assert!(Counters::read(&Scrape::parse(half).unwrap()).is_err());
}

#[test]
fn joules_come_from_the_rail_integral_and_the_counter_is_only_recorded_beside_them() {
    let c = measured(&speculative());
    // 2026-10-04: The rail integral is 800 J over 200 steps; the serve's counter moved 840 J.
    assert!(close(c.draft_j.unwrap() + c.verify_j.unwrap(), 4.0));
    assert!(close(c.nvml_j.unwrap(), 4.2));
    let mut m = BTreeMap::new();
    record(4, &CellVerdict::Measured(c), &mut m);
    assert!(close(m["n4_nvml_j"], 4.2));
    assert!(close(m["n4_verify_j"] + m["n4_draft_j"], 4.0));
}

#[test]
fn an_energy_counter_that_went_backwards_makes_the_width_vacuous() {
    let s = speculative();
    let w = Window {
        after: Counters {
            energy_mj: Some(1.0),
            ..s.after
        },
        ..s
    };
    match evaluate(&w) {
        CellVerdict::Vacuous(why) => assert!(why.contains("went backwards"), "{why}"),
        CellVerdict::Measured(c) => panic!("measured {c:?}"),
    }
}
