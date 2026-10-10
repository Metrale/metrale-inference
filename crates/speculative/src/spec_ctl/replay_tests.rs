// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Replay of recorded traces: the controller against the mechanisms it replaces,
//! each on its own trace, by delivered tokens per ms (and per joule where joules were
//! measured). Fixtures: `tests/fixtures/spec_ctl/` (provenance in each file's header).
//!
//! - Qwen3.8-27B MTP, one GB10, widths 1/2/4/8: the static ladder, the MTP gate (ladder depth
//!   vs plain decode, by measured throughput; its result recorded in the fixture since the
//!   gate moved onto this controller) and `--spec-cost-model measured` (calibration fitted on
//!   the other run, as a serve plans from an earlier bench) against the controller with the
//!   measured table (Throughput, and Energy with the planner's floor). The adaptive rung adapts widths 9..=16 only, where no
//!   trace was recorded; at these widths it is the ladder.
//! - GLM-5.3 Flash DFlash2, one stream: the logged fixed depth, the gamma resolver and the gate
//!   (both recorded in the fixture), and the GLM branch's per-stream adaptive count (its exact port) against the controller
//!   with a cold-start prior and a re-probe window.
//! - Both also against the controller with measured-only costs (no table, no reference
//!   rates: a chained cold prior): the configuration that replaces the gate where no table exists.
//!
//! Owner: speculative.
//! Invariants: none beyond the types.

use serde::Deserialize;

use super::accept::{AcceptParams, ColdPrior, SeedWeight};
use super::calib::Calibration;
use super::chain::{MAX_POSITIONS, conditional_from_marginal};
use super::controller::{ControllerConfig, SpecController};
use super::cost::{CostModel, CostSource, StepCost, StepTable};
use super::decide::{FloorRef, Margins, Objective};
use super::online::OnlineTable;
use super::replay::{Controlled, Fixed, Outcome, Policy, replay};
use super::reprobe::ReprobePolicy;
use super::source::{DraftCost, DraftKind, DraftSource};
use crate::spec_cost::{AcceptanceCalibration, Cell, CostTable, SCHEMA, TableKey};

#[derive(Deserialize)]
struct QwenFile {
    k_logged: usize,
    run: Vec<QwenRun>,
}

#[derive(Deserialize)]
struct QwenRun {
    name: String,
    baseline_gate_tok_ms: [f64; 4],
    cells: Vec<[f64; 6]>,
    windows: Vec<[usize; 4]>,
}

#[derive(Deserialize)]
struct GlmFile {
    step_ms: Vec<f64>,
    trace: Vec<GlmTrace>,
}

#[derive(Deserialize)]
struct GlmTrace {
    name: String,
    baseline_resolver_tok_ms: f64,
    baseline_gate_tok_ms: f64,
    k_logged: usize,
    steps: Vec<String>,
}

fn qwen() -> QwenFile {
    toml::from_str(include_str!(
        "../../tests/fixtures/spec_ctl/qwen38_27b_mtp_gb10.toml"
    ))
    .expect("qwen fixture")
}

fn glm() -> GlmFile {
    toml::from_str(include_str!(
        "../../tests/fixtures/spec_ctl/glm53_flash_dflash2_gb10x3.toml"
    ))
    .expect("glm fixture")
}

/// 2026-10-10: A 100-step histogram `[a3, a2, a1, reject]` as steps, each outcome spread
/// evenly through the window (the order inside a window was not logged).
fn spread(windows: &[[usize; 4]]) -> Vec<u8> {
    let mut out = Vec::new();
    for w in windows {
        let mut items: Vec<(f64, u8)> = Vec::new();
        for (slot, &count) in w.iter().enumerate() {
            let accepted = 3 - slot as u8;
            items.extend((0..count).map(|i| ((i as f64 + 0.5) / count as f64, accepted)));
        }
        items.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        out.extend(items.into_iter().map(|x| x.1));
    }
    out
}

fn table(cells: &[[f64; 6]]) -> CostTable {
    let key = TableKey {
        schema: SCHEMA,
        box_class: "gb10".into(),
        recipe: "replay".into(),
        plan_digests: Default::default(),
    };
    let cells: Vec<Cell> = cells
        .iter()
        .map(|c| Cell {
            n: c[0] as usize,
            k: c[1] as usize,
            verify_ms: c[2],
            verify_j: c[3],
            draft_ms: c[4],
            draft_j: c[5],
        })
        .collect();
    CostTable::parse(&CostTable::render(&key, &cells).expect("render")).expect("parse")
}

/// 2026-10-10: Conditional acceptance of a whole trace logged at `k`.
fn conditional(trace: &[u8], k: usize) -> Vec<f64> {
    let m: Vec<f64> = (1..=k)
        .map(|j| trace.iter().filter(|&&a| a as usize >= j).count() as f64 / trace.len() as f64)
        .collect();
    conditional_from_marginal(&m)
}

/// 2026-10-10: The calibration `met bench spec-cost` would fit on this trace: its position
/// priors (no confidences are replayed).
fn calibration(trace: &[u8], k: usize) -> AcceptanceCalibration {
    let priors: Vec<String> = conditional(trace, k)
        .iter()
        .map(|p| format!("{p:?}"))
        .collect();
    AcceptanceCalibration::parse(&format!(
        "[drafter]\nweights_sha256 = \"replay\"\nvocab = 1\nquantization = \"bf16\"\n\
         context = true\n[acceptance]\nedges = [0.0]\np_accept = [0.5]\n\
         prior_by_position = [{}]\n",
        priors.join(", ")
    ))
    .expect("calibration")
}

/// 2026-10-10: The controller's MTP configuration under test: `source` (the measured table,
/// or measured-only online costs), calibration on, per-stream acceptance with the serve-wide
/// prior and, when given, cold-start rates.
fn mtp_controller(
    source: CostSource,
    objective: Objective,
    n: usize,
    cold: ColdPrior,
) -> Controlled {
    let cfg = ControllerConfig {
        source: DraftSource {
            kind: DraftKind::Mtp,
            max_k: 3,
            prefix_stable: true,
            draft_cost: DraftCost::PerDraft { ms: 0.0, j: 0.0 },
        },
        accept: AcceptParams {
            decay: 0.95,
            prior_weight: 4.0,
            cold_weight: 8.0,
            seed: SeedWeight::Unit,
        },
        objective,
        margins: Margins {
            deeper: 0.0,
            shallower: 0.0,
            suspend: 0.03,
        },
        reprobe: ReprobePolicy {
            explore_every: Some(32),
            explore_max: 512,
            resume_after_tokens: Some(256),
            probe_steps: 16,
            soften: 0.25,
        },
    };
    let cost = CostModel {
        source,
        calib: Calibration::new(0.1),
    };
    Controlled::new(SpecController::new(cfg, cost, cold), n, 3)
}

/// 2026-10-10: The reference engine's measured DFlash2 acceptance on GLM-5.3 Flash (marginal
/// by position; the GLM campaign's acceptance control), the DFlash cold-start prior.
const REFERENCE: [f64; 5] = [0.611, 0.312, 0.117, 0.028, 0.008];

/// 2026-10-10: The controller as `--dflash-adaptive-k` configures it, with the reference
/// acceptance as its cold-start prior and a 16-step re-probe window.
/// 2026-10-10: The GLM branch's `--dflash-adaptive-k` (`dflash_adaptive_k.rs` + its
/// `adaptive_spec.rs` suspension) as an exact configuration of the controller: no cold-start
/// prior, no re-probe window, a fixed exploration cadence of 16, suspension re-probed after
/// 256 plain tokens with counts softened by 0.25. (Checked off-tree against that file on the
/// six GLM traces: the same draft count on every one of 24 000 replayed decisions.)
fn adaptive_k_exact_port(step_ms: &[f64], cap: usize) -> Controlled {
    let mut c = dflash_controller(step_ms, cap);
    c.ctl.cold = ColdPrior::None;
    c.ctl.cfg.accept.cold_weight = 0.0;
    c.ctl.cfg.reprobe.probe_steps = 0;
    c.ctl.cfg.reprobe.explore_max = 16;
    c
}

fn dflash_controller(step_ms: &[f64], cap: usize) -> Controlled {
    let spec: Vec<String> = step_ms
        .iter()
        .enumerate()
        .map(|(k, ms)| format!("{k}:{ms}"))
        .collect();
    let cfg = ControllerConfig {
        source: DraftSource {
            kind: DraftKind::DFlash,
            max_k: 7,
            prefix_stable: true,
            draft_cost: DraftCost::PerBlock { ms: 0.0, j: 0.0 },
        },
        accept: AcceptParams {
            decay: 0.95,
            prior_weight: 4.0,
            cold_weight: 8.0,
            seed: SeedWeight::Unit,
        },
        objective: Objective::Latency,
        margins: Margins {
            deeper: 0.0,
            shallower: 0.0,
            suspend: 0.03,
        },
        reprobe: ReprobePolicy {
            explore_every: Some(16),
            explore_max: 256,
            resume_after_tokens: Some(256),
            probe_steps: 16,
            soften: 0.25,
        },
    };
    let cost = CostModel {
        source: CostSource::StepTable(StepTable::parse(&spec.join(",")).expect("table")),
        calib: Calibration::new(0.1),
    };
    Controlled::new(
        SpecController::new(
            cfg,
            cost,
            ColdPrior::Rates(conditional_from_marginal(&REFERENCE)),
        ),
        1,
        cap,
    )
}

fn row(trace: &str, n: usize, name: &str, o: &Outcome) {
    let j = o.tok_per_j().map_or("-".to_string(), |x| format!("{x:.3}"));
    let top = o.by_depth.iter().rposition(|&c| c > 0).unwrap_or(0);
    let mix: Vec<String> = (0..=top)
        .map(|k| format!("{:.2}", o.by_depth[k] as f64 / o.steps as f64))
        .collect();
    println!(
        "REPLAY {trace:<22} n={n} {name:<18} tok/ms {:.5} tok/J {j:>6} K-mix [{}]",
        o.tok_per_ms(),
        mix.join(" ")
    );
}

/// 2026-10-10: The trace's sampling noise: the standard error of its mean tokens per step,
/// relative to the mean. Two policies whose delivered rates differ by less than this on the
/// same trace are not measurably different (a near-tie between two depths).
fn sampling_noise(trace: &[u8]) -> f64 {
    let x: Vec<f64> = trace.iter().map(|&a| 1.0 + a as f64).collect();
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (var / n).sqrt() / mean
}

/// 2026-10-10: Qwen MTP: per run and width, the controller (Throughput) delivers at least the
/// tokens/ms of the ladder, the gate and the measured planner; the controller (Energy, the
/// planner's floor) at least the planner's tokens/J; both up to the trace's sampling noise
/// ([`sampling_noise`]), below which a near-tie between two depths cannot be told apart.
#[test]
fn qwen_mtp_traces_the_controller_is_at_least_as_good_as_every_old_mechanism() {
    let f = qwen();
    for (r, run) in f.run.iter().enumerate() {
        let trace = spread(&run.windows);
        let t = table(&run.cells);
        // 2026-10-10: Out of sample, as a serve would be: the planner's calibration and the
        // controller's cold-start rates come from the OTHER run's trace.
        let other = spread(&f.run[1 - r].windows);
        let cal = calibration(&other, f.k_logged);
        let noise = sampling_noise(&trace);
        println!("REPLAY {} sampling noise {noise:.4}", run.name);
        for (wi, n) in [1usize, 2, 4, 8].into_iter().enumerate() {
            let gate = run.baseline_gate_tok_ms[wi];
            let budget = n as f64 * trace.iter().map(|&a| 1.0 + a as f64).sum::<f64>();
            let cost = |k: usize| super::cost::table_cost(&t, n, k);
            let ladder = metrale_model_layers::speculative::ladder_drafts_from_steps(
                &[(4, 3), (8, 3), (16, 1), (32, 1)],
                n,
                f.k_logged,
            );
            let run_p =
                |p: &mut dyn Policy| replay(&trace, f.k_logged, n, budget, cost, p).unwrap();
            let fixed = run_p(&mut Fixed(ladder));
            let planned_k =
                super::measured::propose_depth(&t, &cal, n, 0.0, 32).clamp(1, f.k_logged);
            let planned = run_p(&mut Fixed(planned_k));
            let cold = ColdPrior::Rates(conditional(&other, f.k_logged));
            let measured = || CostSource::Measured(t.clone());
            let thr = run_p(&mut mtp_controller(
                measured(),
                Objective::Throughput,
                n,
                cold.clone(),
            ));
            let online = CostSource::Online(OnlineTable::new(0.3, 256, 3, 2.0));
            let mut onl_ctl = mtp_controller(online, Objective::Throughput, n, ColdPrior::Chained);
            let onl = run_p(&mut onl_ctl);
            let energy = Objective::Energy {
                slack: 0.0,
                floor: FloorRef::Depth(1),
            };
            let en = run_p(&mut mtp_controller(measured(), energy, n, cold));
            for (name, o) in [
                ("static ladder", &fixed),
                ("measured planner", &planned),
                ("ctl throughput", &thr),
                ("ctl energy", &en),
                ("ctl online", &onl),
            ] {
                row(&run.name, n, name, o);
            }
            let olds = [
                ("ladder", fixed.tok_per_ms()),
                ("gate (recorded)", gate),
                ("planner", planned.tok_per_ms()),
            ];
            for (name, old) in olds {
                assert!(
                    thr.tok_per_ms() >= old * (1.0 - noise),
                    "{} n={n}: controller {:.5} tok/ms < {name} {old:.5}",
                    run.name,
                    thr.tok_per_ms(),
                );
            }
            for (name, old) in &olds[..2] {
                assert!(
                    onl.tok_per_ms() >= old * (1.0 - noise),
                    "{} n={n}: online controller {:.5} tok/ms < {name} {old:.5}",
                    run.name,
                    onl.tok_per_ms(),
                );
            }
            assert!(
                en.tok_per_j().unwrap() >= planned.tok_per_j().unwrap() * (1.0 - noise),
                "{} n={n}: controller {:?} tok/J < planner {:?}",
                run.name,
                en.tok_per_j(),
                planned.tok_per_j()
            );
        }
    }
}

/// 2026-10-10: GLM DFlash, one stream: the controller with the step table delivers at least the
/// tokens/ms of the logged fixed depth, the gamma resolver and the gate on every trace, and of
/// the adaptive-K exact port up to the trace's sampling noise; with
/// measured-only costs and no reference rates (what replaces the gate where no table exists)
/// it does too, up to the trace's sampling noise.
#[test]
fn glm_dflash_traces_the_controller_is_at_least_as_good_as_every_old_mechanism() {
    let f = glm();
    let cost = |k: usize| f.step_ms.get(k).map(|&ms| StepCost { ms, j: None });
    for tr in &f.trace {
        let trace: Vec<u8> = tr.steps.concat().bytes().map(|b| b - b'0').collect();
        let budget = trace.iter().map(|&a| 1.0 + a as f64).sum::<f64>();
        let k = tr.k_logged;
        let run_p = |p: &mut dyn Policy| replay(&trace, k, 1, budget, cost, p).unwrap();
        let fixed = run_p(&mut Fixed(k));
        let ctl = run_p(&mut dflash_controller(&f.step_ms[..=k], k));
        let adaptive_k = run_p(&mut adaptive_k_exact_port(&f.step_ms[..=k], k));
        let mut online = dflash_controller(&f.step_ms[..=k], k);
        online.ctl.cost.source = CostSource::Online(OnlineTable::new(0.3, 256, 3, 2.0));
        online.ctl.cold = ColdPrior::Chained;
        let onl = run_p(&mut online);
        let noise = sampling_noise(&trace);
        for (name, o) in [
            ("fixed (logged)", &fixed),
            ("adaptive-K port", &adaptive_k),
            ("ctl latency", &ctl),
            ("ctl online", &onl),
        ] {
            row(&tr.name, 1, name, o);
        }
        println!("REPLAY {} sampling noise {noise:.4}", tr.name);
        // 2026-10-10: Where plain decode is best the exact port suspends almost at once and
        // the cold-start prior keeps re-probing, so against the port the controller is held to
        // the trace's sampling noise; where speculation pays it is ahead (table in the PR).
        assert!(
            ctl.tok_per_ms() >= adaptive_k.tok_per_ms() * (1.0 - noise),
            "{}: controller {:.5} tok/ms < adaptive-K port {:.5}",
            tr.name,
            ctl.tok_per_ms(),
            adaptive_k.tok_per_ms()
        );
        let olds = [
            ("fixed", fixed.tok_per_ms()),
            ("resolver (recorded)", tr.baseline_resolver_tok_ms),
            ("gate (recorded)", tr.baseline_gate_tok_ms),
        ];
        for (name, old) in olds {
            assert!(
                onl.tok_per_ms() >= old * (1.0 - noise),
                "{}: online controller {:.5} tok/ms < {name} {old:.5}",
                tr.name,
                onl.tok_per_ms(),
            );
            assert!(
                ctl.tok_per_ms() >= old,
                "{}: controller {:.5} tok/ms < {name} {old:.5}",
                tr.name,
                ctl.tok_per_ms(),
            );
        }
    }
}

/// 2026-10-10: The counterfactual rule: a fixed depth below the logged one emits exactly
/// `1 + min(a, k)` per step, and the budget stops every policy at the same work.
#[test]
fn the_replay_truncates_at_the_chosen_depth_and_charges_every_step() {
    let trace = [3u8, 0, 2, 1];
    let cost = |k: usize| {
        Some(StepCost {
            ms: 10.0 + k as f64,
            j: Some(1.0),
        })
    };
    let o = replay(&trace, 3, 1, 1e-9, cost, &mut Fixed(2)).unwrap();
    assert_eq!((o.tokens, o.steps), (3.0, 1));
    let o = replay(&trace, 3, 1, 8.0, cost, &mut Fixed(2)).unwrap();
    assert_eq!((o.tokens, o.steps, o.ms), (3.0 + 1.0 + 3.0 + 2.0, 4, 48.0));
    let o = replay(&trace, 3, 2, 8.0, cost, &mut Fixed(0)).unwrap();
    assert_eq!((o.tokens, o.plain_steps), (8.0, 4));
    assert_eq!(o.tok_per_j(), Some(2.0));
    assert!(replay(&[], 3, 1, 8.0, cost, &mut Fixed(1)).is_none());
    assert_eq!(MAX_POSITIONS, 16);
}
