// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Plan fidelity of GLM-5.3-Flash TP=3 on GB10: every rank's plan at decode n=1,
//! verify n=2 and draft n=1 scored against the f14a and t4c (`--forward circuit`) captures
//! (kernels/circuits/plans/fidelity/),
//! the report current, the score discriminating, and every rank fully plan-describable. Regenerate after an intended change with
//! `cargo test -p met circuit_fidelity -- --ignored regenerate_fidelity`.
//!
//! Owner: server CLI tests.
//! Invariants: the tests read the repository at the workspace root; nothing is written except by
//! the ignored `regenerate_fidelity`.

use std::path::PathBuf;

use metrale_circuit::Mode;
use metrale_circuit::fidelity::render::{RankPlan, render, score};
use metrale_circuit::fidelity::{Trace, parse_trace};
use metrale_circuit::hardware::{self, CircuitSource, ModelSpec, PrecisionChoice};
use metrale_circuit::venn::Run;

use super::{FsTree, registry, source, write_or_check};

const DIR: &str = "kernels/circuits/plans/fidelity";
const REPORT: &str = "kernels/circuits/plans/fidelity/glm-5.3-flash-nvfp4-tp3--gb10.md";
const TRACES: [&str; 4] = [
    "glm-5.3-flash-nvfp4-tp3--gb10--f14a-p1k.trace.toml",
    "glm-5.3-flash-nvfp4-tp3--gb10--f14a-p128k.trace.toml",
    "glm-5.3-flash-nvfp4-tp3--gb10--t4c-p1k.trace.toml",
    "glm-5.3-flash-nvfp4-tp3--gb10--t4c-p128k.trace.toml",
];
const COMMAND: &str = "cargo test -p met circuit_fidelity -- --ignored regenerate_fidelity";
/// 2026-10-23: The runs a speculative TP=3 step makes (verify n=2 with its one draft) and the
/// non-speculative decode step.
const RUNS: [Run; 3] = [
    Run { mode: Mode::Decode, rows: 1 },
    Run { mode: Mode::Verify, rows: 2 },
    Run { mode: Mode::Draft, rows: 1 },
];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

fn traces() -> Vec<(String, Trace)> {
    TRACES
        .iter()
        .map(|f| {
            let text = std::fs::read_to_string(root().join(DIR).join(f)).expect("trace oracle");
            (f.to_string(), parse_trace(&text).unwrap_or_else(|e| panic!("{f}: {e}")))
        })
        .collect()
}

struct Planned {
    rank: u64,
    run: Run,
    plan: hardware::plan::Planned,
}

fn plans() -> Vec<Planned> {
    let tree = FsTree::new(root());
    let reg = registry(&tree).unwrap();
    let checkpoint = "nvidia/GLM-5.3-Flash-NVFP4";
    let dir = root()
        .join(super::MATRIX_CONFIGS)
        .join(checkpoint.replacen('/', "--", 1));
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).ok();
    let (config, hf_quant) = (read("config.json"), read("hf_quant_config.json"));
    assert!(config.is_some(), "the checked-in GLM config the matrix plans from");
    let model = source(&tree)
        .model(&ModelSpec {
            checkpoint,
            config_json: config.as_deref(),
            hf_quant: hf_quant.as_deref(),
            precision: PrecisionChoice::Recipe,
        })
        .unwrap();
    assert_eq!(model.ranks.len(), 3, "the tp3 instance plans three ranks");
    let resolved = hardware::plan::resolve(&reg, "gb10", &tree, &model).unwrap();
    let (policy, _) = hardware::model::policy_on_class(
        &model.policy,
        model.settings_class.as_deref(),
        &resolved.chain[0],
    )
    .unwrap();
    let mut out = Vec::new();
    for run in RUNS {
        for (c, rank) in model.ranks.iter().zip(0u64..) {
            let plan = hardware::plan::fuse_on(&resolved, c, &policy, run)
                .unwrap_or_else(|e| panic!("rank {rank} {run:?}: {e}"));
            out.push(Planned { rank, run, plan });
        }
    }
    out
}

fn report(ps: &[Planned], ts: &[(String, Trace)]) -> String {
    let rank_plans: Vec<RankPlan<'_>> = ps
        .iter()
        .map(|p| RankPlan {
            rank: p.rank,
            run: p.run,
            plan: &p.plan.plan,
        })
        .collect();
    render("GLM-5.3-Flash NVFP4 TP=3 on GB10", COMMAND, ts, &rank_plans)
}

#[test]
fn fidelity_report_is_current() {
    let (ps, ts) = (plans(), traces());
    write_or_check(&root(), REPORT, &report(&ps, &ts), true)
        .unwrap_or_else(|e| panic!("{e:#}\nregenerate with `{COMMAND}`"));
}

#[test]
#[ignore = "writes kernels/circuits/plans/fidelity/; run explicitly to regenerate"]
fn regenerate_fidelity() {
    let (ps, ts) = (plans(), traces());
    write_or_check(&root(), REPORT, &report(&ps, &ts), false).unwrap();
}

/// 2026-10-23: Every run the serve was captured at is planned without a placeholder on every
/// rank, and decode n=1 too: the plans are fully plan-describable.
#[test]
fn every_rank_is_plan_describable_at_every_run() {
    for p in plans() {
        let (_, novel) = metrale_circuit::fidelity::plan_sequence(&p.plan.plan);
        assert!(
            novel.is_empty(),
            "rank {} {:?}: placeholders {novel:?}",
            p.rank,
            p.run
        );
    }
}

/// 2026-10-23: The captured rank's plans score against their segments (a run the oracle states
/// is found), and a plan of another run scores below 1 against it: the score discriminates.
#[test]
fn a_captured_run_is_scored_and_a_wrong_plan_scores_below_it() {
    let (ps, ts) = (plans(), traces());
    let (_, t) = &ts[0];
    let at = |run: Run| ps.iter().find(|p| p.rank == t.rank && p.run == run).unwrap();
    let verify = Run { mode: Mode::Verify, rows: 2 };
    let draft = Run { mode: Mode::Draft, rows: 1 };
    let right = score(&at(verify).plan.plan, t, verify);
    assert!(!right.is_empty(), "the oracle states verify n=2");
    let wrong = score(&at(draft).plan.plan, t, verify);
    assert!(wrong[0].2.share() < right[0].2.share() && wrong[0].2.share() < 0.5);
}

/// 2026-10-23: No rule is refused on any rank at any run. A refused rule is planned around
/// silently (`fuse_arm` drops it and fuses again), and one refusal can cascade: the AR + post-mix
/// rule refused on its stored format left the stream BF16, which then refused every mHC pre-mix
/// reader (verify fidelity 70.9% with no error anywhere).
#[test]
fn no_rule_is_refused_on_any_rank() {
    for p in plans() {
        assert!(p.plan.refused.is_empty(), "rank {} {:?}: refused {:?}", p.rank, p.run, p.plan.refused);
    }
}

/// 2026-10-23: Every captured run's plan launches exactly the captured kernels (multiset), on
/// every capture: a rule change that moves a launch shows here, not only in the report.
#[test]
fn every_captured_run_matches_its_capture() {
    let (ps, ts) = (plans(), traces());
    for (file, t) in &ts {
        for p in ps.iter().filter(|p| p.rank == t.rank) {
            for (variant, _, f) in score(&p.plan.plan, t, p.run) {
                assert!(
                    f.missing.is_empty() && f.extra.is_empty(),
                    "{file} {variant} {:?}: traced, not planned {:?}; planned, not traced {:?}",
                    p.run,
                    f.missing,
                    f.extra
                );
            }
        }
    }
}
