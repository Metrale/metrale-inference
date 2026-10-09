// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Both arms of the GDN conv step contracts of kernels/gb10/common/ACCURACY.toml
//! (conv1d update + SiLU + the q/k heads' L2 norm), proven on the CPU against the conforming
//! emulation at the sweep's two widths (Qwen3.6-35B-A3B: 8192 channels; Qwen3.8-27B: 10240;
//! 16 key heads of 128, 4 taps): every input class passes, judged on the output and on the
//! updated window; a one-step-stale window and a BF16 accumulator are caught; a kernel that
//! accumulates in BF16 fails the bound; a contract made too loose fails because a mutation
//! passes.

mod common;

use common::{Behaviour, Emu, Tree, families};
use metrale_accuracy::check::{Job, Outcome, Verdict, run};
use metrale_accuracy::contract::{Class, Contract, parse_contracts};
use metrale_accuracy::elem::BF16;
use metrale_accuracy::inputs::InputClass;
use metrale_accuracy::plan;
use metrale_accuracy::points::Shape;
use metrale_circuit::pipeline::StepKind;
use metrale_circuit::venn::Repo;

/// 2026-10-09: The committed derived `conv1d_l2norm` contracts (strided first).
fn derived() -> Vec<Contract> {
    let text = Tree.read("kernels/gb10/common/ACCURACY.toml").unwrap();
    let mut v: Vec<Contract> = parse_contracts(&text)
        .unwrap()
        .contracts
        .into_iter()
        .filter(|c| c.reference == "conv1d_l2norm" && c.class == Class::Derived)
        .collect();
    v.sort_by_key(|c| !c.kernels[0].ends_with("_strided"));
    assert_eq!(v.len(), 2, "{v:#?}");
    v
}

/// 2026-10-09: A conv point of `dim` channels whose first 4096 are 32 normalized heads.
fn shape(rows: u64, dim: u64) -> Shape {
    let rt = [
        ("k_heads", "16"),
        ("k_dim", "128"),
        ("d_conv", "4"),
        ("l2_eps", "1e-6"),
    ];
    Shape {
        op: "conv1d_update".into(),
        weight: None,
        activation: Some("bf16".into()),
        output: Some("f32".into()),
        in_dim: dim,
        out_dim: dim,
        rows,
        runtime: rt
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
    }
}

fn check(c: &Contract, behaviour: Behaviour, s: &Shape, input: InputClass) -> Outcome {
    let family = families()
        .families
        .into_iter()
        .find(|f| f.id == c.family)
        .unwrap();
    let mut e = Emu {
        contract: c.clone(),
        family: family.clone(),
        behaviour,
        wrong: ("no::such_symbol".into(), |_| Ok(())),
        shape: s.clone(),
    };
    let job = Job {
        contract: c,
        family: &family,
        kernel: &c.kernels[0],
        point: &Default::default(),
        shape: s,
        input,
        seed: 20261009,
    };
    run(&job, &mut e)
}

#[test]
fn the_committed_conv_contracts_fit_the_families_and_the_fused_norm_matches_the_conv_formats() {
    let text = Tree.read("kernels/gb10/common/ACCURACY.toml").unwrap();
    let all = parse_contracts(&text).unwrap();
    let fams = families();
    let problems = metrale_accuracy::jobs::validate(&all, &fams);
    assert!(
        !problems.iter().any(|p| p.contains("conv1d")),
        "{problems:#?}"
    );
    // 2026-10-09: The reference runs the fused L2 norm at the conv op's compute and output
    // precisions; the family's own `l2_norm` declaration for these kernels must agree.
    let f = fams
        .families
        .iter()
        .find(|f| f.id == "causal_conv1d_l2norm")
        .unwrap();
    for c in derived() {
        let k = &c.kernels[0];
        let conv = plan::declared(f, k, "conv1d_update", &Default::default()).unwrap();
        let l2 = plan::declared(f, k, "l2_norm", &Default::default()).unwrap();
        let compute = |p: &metrale_circuit::pipeline::NodePipeline| {
            p.steps
                .iter()
                .find(|s| s.kind == StepKind::Compute)
                .map(|s| s.value.clone())
        };
        assert_eq!(compute(&conv), compute(&l2), "{k}");
        assert_eq!(conv.outputs, l2.outputs, "{k}");
        assert_eq!(conv.outputs, l2.inputs, "{k}");
    }
}

#[test]
fn both_widths_pass_every_class_and_catch_every_mutation() {
    for c in derived() {
        for dim in [8192, 10240] {
            let s = shape(2, dim);
            for &input in &c.inputs {
                let o = check(&c, Behaviour::Conforming, &s, input);
                let at = format!("{} {dim}, {}", c.kernels[0], input.name());
                assert_eq!(o.verdict, Verdict::Pass, "{at}: {o:#?}");
                let (good, floor) = (o.good.unwrap(), o.floor.unwrap());
                println!(
                    "{at}: good {:.3e} floor {:.3e} over {}",
                    good.ratio, floor.ratio, good.compared
                );
                assert!(
                    good.ratio < 0.5,
                    "{at}: good ratio {} leaves no margin",
                    good.ratio
                );
                assert!(good.compared > 200, "{at}: {} compared", good.compared);
                if input == InputClass::Gaussian {
                    assert_eq!(o.mutations.len(), 2, "{at}");
                    for m in &o.mutations {
                        println!("{at}: {} {:.3e}", m.name, m.ratio);
                        assert!(m.ratio > 2.0, "{at}: {} caught only at {}", m.name, m.ratio);
                    }
                }
            }
        }
    }
}

#[test]
fn a_kernel_that_accumulates_in_bf16_fails_the_bound() {
    let o = check(
        &derived()[0],
        Behaviour::Accumulator(BF16),
        &shape(1, 8192),
        InputClass::Gaussian,
    );
    assert_eq!(o.verdict, Verdict::FailBound, "{o:#?}");
}

#[test]
fn a_contract_too_loose_to_catch_a_mutation_fails_the_run() {
    // 2026-10-09: A million-term chain per tap sum and per head makes those bounds about 6% of
    // their absolute sums: the BF16 accumulator then stays inside them.
    let mut loose = derived().remove(0);
    for levels in loose.reduction.values_mut() {
        for l in levels {
            l.width = "1000000".into();
            l.order = "sequential".into();
        }
    }
    let o = check(
        &loose,
        Behaviour::Conforming,
        &shape(1, 8192),
        InputClass::Gaussian,
    );
    assert!(
        matches!(o.verdict, Verdict::FailMutationPassed(_)),
        "{:?}",
        o.verdict
    );
}

#[test]
fn a_point_without_its_launch_values_is_refused() {
    let c = &derived()[0];
    for missing in ["k_heads", "k_dim", "d_conv", "l2_eps"] {
        let mut s = shape(1, 8192);
        s.runtime.remove(missing);
        let o = check(c, Behaviour::Conforming, &s, InputClass::Gaussian);
        assert!(
            matches!(o.verdict, Verdict::Error(_)),
            "{missing}: {:?}",
            o.verdict
        );
    }
}
