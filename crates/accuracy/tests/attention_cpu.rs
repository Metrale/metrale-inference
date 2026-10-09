// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Both arms of the paged decode attention contract, proven on the CPU against the
//! conforming emulation: a conforming "kernel" passes on every input class with a margin; the
//! swapped KV page and the narrower accumulator are caught; a kernel that accumulates in bf16
//! fails the bound; a contract made too loose fails because a mutation passes. Shapes are the
//! swept models' (Qwen3.8-27B: 24 q / 4 kv heads at head_dim 256; Qwen3.6-35B-A3B: 16 / 2),
//! plus a head_dim 128 point; contexts per `attention::CONTEXTS` (up to 4093 tokens).

mod common;

use common::{Behaviour, Emu, Tree, families};
use metrale_accuracy::check::{Job, Outcome, Verdict, run};
use metrale_accuracy::contract::{Contract, parse_contracts};
use metrale_accuracy::elem::BF16;
use metrale_accuracy::inputs::InputClass;
use metrale_accuracy::jobs::{Scope, plan, validate};
use metrale_accuracy::points::Shape;
use metrale_accuracy::points::sweep;
use metrale_circuit::venn::Repo;

/// 2026-10-09: The paged decode attention contract of kernels/gb10/common/ACCURACY.toml.
fn paged() -> Contract {
    let text = Tree.read("kernels/gb10/common/ACCURACY.toml").unwrap();
    parse_contracts(&text)
        .unwrap()
        .contracts
        .into_iter()
        .find(|c| c.family == "paged_decode_attn" && c.reference == "paged_attention")
        .expect("the paged_decode_attn contract")
}

fn shape(rows: u64, q_heads: u64, kv_heads: u64, head_dim: u64) -> Shape {
    let w = q_heads * head_dim;
    Shape {
        op: "paged_attention".into(),
        weight: None,
        activation: Some("bf16".into()),
        output: Some("bf16".into()),
        in_dim: w,
        out_dim: w,
        rows,
        runtime: [
            ("q_heads".to_string(), q_heads.to_string()),
            ("kv_heads".to_string(), kv_heads.to_string()),
        ]
        .into_iter()
        .collect(),
    }
}

fn check(c: Contract, behaviour: Behaviour, s: &Shape, input: InputClass) -> Outcome {
    let family = families()
        .families
        .into_iter()
        .find(|f| f.id == "paged_decode_attn")
        .unwrap();
    // 2026-10-09: No wrong-symbol arm: the runner never sees this name.
    let wrong: (String, common::Wrong) = ("none::none".to_string(), |_| Ok(()));
    let mut e = Emu {
        contract: c.clone(),
        family: family.clone(),
        behaviour,
        wrong,
        shape: s.clone(),
    };
    let job = Job {
        contract: &c,
        family: &family,
        kernel: &c.kernels[0],
        point: &Default::default(),
        shape: s,
        input,
        seed: 20261009,
    };
    run(&job, &mut e)
}

fn report(o: &Outcome) -> String {
    let m: Vec<String> = o
        .mutations
        .iter()
        .map(|m| format!("{}={:.3e}", m.name, m.ratio))
        .collect();
    format!(
        "{} {}: good {:.3e} floor {:.3e} [{}]",
        o.key,
        o.input.name(),
        o.good.as_ref().map_or(f64::NAN, |a| a.ratio),
        o.floor.as_ref().map_or(f64::NAN, |a| a.ratio),
        m.join(", ")
    )
}

fn passes_with_margin(s: &Shape, inputs: &[InputClass]) {
    for &input in inputs {
        let o = check(paged(), Behaviour::Conforming, s, input);
        println!("{}", report(&o));
        assert_eq!(o.verdict, Verdict::Pass, "{}: {o:#?}", input.name());
        let good = o.good.as_ref().unwrap();
        assert!(
            good.ratio < 0.5,
            "{}: good ratio {}",
            input.name(),
            good.ratio
        );
        assert!(o.floor.as_ref().unwrap().ratio < 0.5);
        assert!(good.compared > 100);
        if input == InputClass::Gaussian {
            assert_eq!(o.mutations.len(), 2);
            for m in &o.mutations {
                assert!(m.ratio > 2.0, "{} caught only at ratio {}", m.name, m.ratio);
            }
        }
    }
}

#[test]
fn qwen38_point_passes_every_class_and_catches_every_mutation() {
    // 2026-10-09: Two sequences: 4093 and 17 tokens, both ending on a partial page.
    let all: Vec<InputClass> = InputClass::all().collect();
    passes_with_margin(&shape(2, 24, 4, 256), &all);
}

#[test]
fn qwen36_point_with_four_context_lengths_passes() {
    // 2026-10-09: 4093, 17, 2049 and 1000 tokens over a GQA ratio of 8.
    passes_with_margin(&shape(4, 16, 2, 256), &[InputClass::Gaussian]);
}

#[test]
fn a_head_dim_128_point_passes() {
    passes_with_margin(
        &shape(2, 32, 8, 128),
        &[InputClass::Gaussian, InputClass::Outliers],
    );
}

#[test]
fn a_kernel_that_accumulates_in_bf16_fails_the_bound() {
    let o = check(
        paged(),
        Behaviour::Accumulator(BF16),
        &shape(2, 24, 4, 256),
        InputClass::Gaussian,
    );
    println!("{}", report(&o));
    assert_eq!(o.verdict, Verdict::FailBound, "{o:#?}");
}

#[test]
fn a_contract_too_loose_to_catch_a_mutation_fails_the_run() {
    // 2026-10-09: A declared exp2 error of 2^-10 is wider than the bf16 accumulator it must
    // tell apart; the run must report the mutation that passed, not pass.
    let mut loose = paged();
    loose.approx.insert("exp2".into(), 2f64.powi(-10));
    let o = check(
        loose,
        Behaviour::Conforming,
        &shape(2, 24, 4, 256),
        InputClass::Gaussian,
    );
    println!("{}", report(&o));
    assert!(
        matches!(o.verdict, Verdict::FailMutationPassed(_)),
        "{:?}",
        o.verdict
    );
}

#[test]
fn an_undeclared_exponential_error_is_an_error_not_exact() {
    let mut undeclared = paged();
    undeclared.approx.clear();
    let o = check(
        undeclared,
        Behaviour::Conforming,
        &shape(1, 24, 4, 256),
        InputClass::Gaussian,
    );
    assert!(
        matches!(&o.verdict, Verdict::Error(e) if e.contains("exp2")),
        "{:?}",
        o.verdict
    );
}

#[test]
fn the_contract_fits_the_family_and_covers_every_swept_point() {
    let text = Tree.read("kernels/gb10/common/ACCURACY.toml").unwrap();
    let contracts = parse_contracts(&text).unwrap();
    let fams = families();
    let problems: Vec<String> = validate(&contracts, &fams)
        .into_iter()
        .filter(|p| p.contains("paged_decode_attn"))
        .collect();
    assert!(problems.is_empty(), "{problems:?}");
    // 2026-10-09: Every swept bf16 point runs under the contract, with the head counts the
    // family's runtime parameters carry from the model.
    let s = sweep(&Tree, "gb10").unwrap();
    let (jobs, cov) = plan(
        &s,
        &contracts,
        &fams,
        Scope::Full,
        Some("paged_decode_attn"),
        None,
    );
    assert!(cov.swept_points > 0);
    // 2026-10-09: A plan-only instance names no kernel (the sweep's own rule); every planned
    // point is covered.
    assert!(
        cov.uncovered.values().all(|why| why.contains("plan-only")),
        "{:?}",
        cov.uncovered
    );
    assert!(!jobs.is_empty());
    assert!(jobs.iter().all(|j| {
        let q: u64 = j.shape.runtime["q_heads"].parse().unwrap();
        let kv: u64 = j.shape.runtime["kv_heads"].parse().unwrap();
        j.shape.in_dim % q == 0 && q.is_multiple_of(kv)
    }));
}
