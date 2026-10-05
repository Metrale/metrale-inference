// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Tests for the speculative-cost key: it is rebuilt here from the checked-in golden
//! plans (`kernels/circuits/plans/`, their `digest:` lines), so a key that drifts from the plans
//! the goldens specify fails.
//!
//! Owner: model-layers (circuit_exec).
//! Invariants: none beyond the types.

use std::path::PathBuf;

use super::*;

const DECLARED_27B: &str = "qwen3.8/qwen3.8-27b-nvfp4-unsloth-declared";

fn golden_digests(file: &str) -> Vec<String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/circuits/plans")
        .join(file);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let found: Vec<String> = text
        .lines()
        .filter_map(|l| l.strip_prefix("digest: "))
        .map(str::to_string)
        .collect();
    assert!(!found.is_empty(), "{file} has no digest line");
    found
}

#[test]
fn the_key_is_the_golden_plans_digested_per_mode() {
    let inst = instance(DECLARED_27B).unwrap();
    let key = spec_cost_plan_digests(DECLARED_27B).unwrap();
    let modes: Vec<&str> = key.keys().map(String::as_str).collect();
    assert_eq!(modes, ["draft", "verify", "verify_batch"]);
    for mode in [Mode::Verify, Mode::Draft] {
        let files: Vec<String> = inst.plans[&mode]
            .iter()
            .map(|&r| inst.plan_file(mode, r))
            .collect();
        let want: Vec<String> = files.iter().flat_map(|f| golden_digests(f)).collect();
        assert_eq!(
            key[mode.name()],
            plans_digest(want.iter().map(String::as_str)),
            "{}",
            mode.name()
        );
    }
    let want: Vec<String> = inst
        .verify_batch
        .iter()
        .flat_map(|t| golden_digests(&inst.table_plan_file(t)))
        .collect();
    assert_eq!(
        key["verify_batch"],
        plans_digest(want.iter().map(String::as_str))
    );
}

#[test]
fn modes_key_differently_and_the_key_is_deterministic() {
    let a = spec_cost_plan_digests(DECLARED_27B).unwrap();
    assert_eq!(a, spec_cost_plan_digests(DECLARED_27B).unwrap());
    assert_ne!(a["verify"], a["draft"]);
    assert_ne!(a["verify"], a["verify_batch"]);
}

#[test]
fn a_recipe_without_an_instance_has_no_key() {
    // 2026-10-05: The default NVFP4 35B recipe (`--weight-quantization nvfp4`) has no instance yet.
    let err = spec_cost_plan_digests("qwen3.6/qwen3.6-35b-a3b-nvfp4").unwrap_err();
    assert!(
        format!("{err:#}").contains("needs a circuit instance"),
        "{err:#}"
    );
}

#[test]
fn a_mode_without_declared_plans_is_refused() {
    // 2026-10-04: The FP8 MoE instance declares no batched-verify table today. 2026-10-05: It
    // does now; the NVFP4 35B's row-major variant plans verify widths but no batched-verify table.
    let err =
        spec_cost_plan_digests("qwen3.6/qwen3.6-35b-a3b-nvfp4-declared-row-major").unwrap_err();
    assert!(
        format!("{err:#}").contains("declares no verify_batch plan"),
        "{err:#}"
    );
}

#[test]
fn route_arms_follow_their_plan_as_in_the_golden_files() {
    // 2026-10-04: No verify or draft plan has a runtime-route arm today; multi_seq plans do (two
    // `digest:` lines per golden), so the arm order is proven on them.
    let inst = instance(DECLARED_27B).unwrap();
    let loaded = metrale_circuit::load(&inst, sources(&inst).unwrap()).unwrap();
    let got = mode_plan_digests(&inst, &loaded, Mode::MultiSeq).unwrap();
    let want: Vec<String> = inst.plans[&Mode::MultiSeq]
        .iter()
        .flat_map(|&r| golden_digests(&inst.plan_file(Mode::MultiSeq, r)))
        .collect();
    assert!(
        want.len() > inst.plans[&Mode::MultiSeq].len(),
        "the goldens carry arms"
    );
    assert_eq!(got, want);
}
