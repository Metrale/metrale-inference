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
use metrale_accuracy::case::{Case, Enc, Tensor};
use metrale_accuracy::check::{Job, Outcome, Verdict, run};
use metrale_accuracy::compare;
use metrale_accuracy::contract::{Class, Contract, parse_contracts};
use metrale_accuracy::elem::{BF16, F32};
use metrale_accuracy::inputs::InputClass;
use metrale_accuracy::plan;
use metrale_accuracy::points::Shape;
use metrale_accuracy::refs::Reference;
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

/// 2026-10-09: The swept conv points' widths (input row, output): Qwen3.6-35B-A3B and
/// Qwen3.8-27B read their q|k|v channels from the q|k|v|z projection row.
const WIDTHS: [(u64, u64); 2] = [(12288, 8192), (16384, 10240)];

/// 2026-10-09: A conv point reading `dim` channels from rows of `in_dim`, the first 4096 of
/// them 32 normalized heads.
fn shape(rows: u64, in_dim: u64, dim: u64) -> Shape {
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
        in_dim,
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
        point: Default::default(),
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
        for (in_dim, dim) in WIDTHS.into_iter().chain([(8192, 8192)]) {
            let s = shape(2, in_dim, dim);
            for &input in &c.inputs {
                let o = check(&c, Behaviour::Conforming, &s, input);
                let at = format!("{} {in_dim}->{dim}, {}", c.kernels[0], input.name());
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
        &shape(1, 12288, 8192),
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
        &shape(1, 12288, 8192),
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
        let mut s = shape(1, 12288, 8192);
        s.runtime.remove(missing);
        let o = check(c, Behaviour::Conforming, &s, InputClass::Gaussian);
        assert!(
            matches!(o.verdict, Verdict::Error(_)),
            "{missing}: {:?}",
            o.verdict
        );
    }
}

#[test]
fn reading_the_input_at_the_output_stride_leaves_the_bound() {
    // 2026-10-09: The kernel reads row b's channels at `b * input_stride`; one that used the
    // output's stride for the input would read row 1 from inside row 0's z channels. The bound
    // must see it: row 1's outputs of that misread lie far outside it.
    let c = &derived()[0];
    let fams = families();
    let family = fams.families.iter().find(|f| f.id == c.family).unwrap();
    let (in_dim, dim) = WIDTHS[0];
    let s = shape(2, in_dim, dim);
    let r = Reference::parse(&c.reference).unwrap();
    let declared = plan::declared(family, &c.kernels[0], &c.op, &Default::default()).unwrap();
    let p = plan::plan(
        c,
        declared.clone(),
        &r.lens(&s, &declared),
        &Default::default(),
    )
    .unwrap();
    let mut case = Case {
        family: family.id.clone(),
        kernel: c.kernels[0].clone(),
        launcher: c.kernels[0].clone(),
        op: c.op.clone(),
        tensors: Default::default(),
        scalars: Default::default(),
        out: (Vec::new(), Enc::F32),
        split: Vec::new(),
    };
    r.fill(&mut case, &p, &s, InputClass::Gaussian, 20261009, "stride")
        .unwrap();
    let x = case.tensor("x").unwrap().clone();
    let flat = x.values();
    let misread: Vec<f64> = (0..2 * in_dim as usize)
        .map(|i| {
            let (row, ch) = (i / in_dim as usize, i % in_dim as usize);
            flat.get(row * dim as usize + ch).copied().unwrap_or(0.0)
        })
        .collect();
    let mut wrong = case.clone();
    wrong.tensors.insert(
        "x".into(),
        Tensor::encode(x.enc, x.dims.clone(), &misread).unwrap(),
    );
    let cols = case.out.0[1];
    let idx: Vec<usize> = (cols..cols + dim as usize).step_by(7).collect();
    let want = r.reference(&case, &p, &idx).unwrap();
    let got = r.emulate(&wrong, &p, None, 0, &idx).unwrap();
    let b = compare::bounded(&got, &want, F32).unwrap();
    assert!(b.max_ratio > 2.0, "{b:?}");
    let same = r.emulate(&case, &p, None, 0, &idx).unwrap();
    assert!(compare::bounded(&same, &want, F32).unwrap().max_ratio < 0.5);
}
