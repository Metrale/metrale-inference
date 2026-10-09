// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Both arms of the projection contracts, proven on the CPU against the conforming
//! emulation: a conforming "kernel" passes on every input class with a margin; every mutation
//! is caught; a kernel that breaks its declaration fails the bound; a contract made too loose
//! fails because a mutation passes. The real gb10 manifest supplies the pipelines.

mod common;

use common::{Behaviour, Emu, contract, families};
use metrale_accuracy::check::{Job, Verdict, run};
use metrale_accuracy::elem::BF16;
use metrale_accuracy::inputs::InputClass;
use metrale_accuracy::mutation::Mutation;
use metrale_accuracy::points::Shape;

const W4A16_SW: &str = r#"
[[contract]]
family = "w4a16_gemv"
kernels = ["w4a16_gemv::w4a16_gemv_sw"]
op = "linear"
reference = "linear"
class = "derived"
reduction.group = [{ level = "thread", width = "16", order = "sequential" }]
reduction.k = [
  { level = "thread", width = "k/2048", order = "sequential" },
  { level = "pair", width = "2", order = "tree" },
  { level = "warp", width = "32", order = "tree" },
  { level = "pair", width = "2", order = "tree" },
]
ftz = false
approx = {}
scale_fold = "group"
inputs = ["gaussian", "outliers", "near_overflow", "denormal", "all_equal", "zero_rows"]
mutations = ["corrupt_block_scale", "swap_scale_granularity", "accumulate:bf16", "symbol:w4a16_gemv::w4a16_gemv_qg"]
"#;

const DENSE_SPLIT: &str = r#"
[[contract]]
family = "dense_bf16"
kernels = ["gemv::dense_gemv_bf16"]
op = "lm_head"
reference = "linear"
class = "derived"
reduction.k = [
  { level = "thread", width = "k/256", order = "sequential" },
  { level = "warp", width = "32", order = "tree" },
  { level = "cta", width = "8", order = "sequential" },
]
ftz = false
approx = {}
scale_fold = "none"
split = { world = 3, align = 64 }
inputs = ["gaussian", "zero_rows"]
mutations = ["split_off_by_one", "accumulate:bf16"]
"#;

fn shape(op: &str, rows: u64, k: u64, n: u64) -> Shape {
    Shape {
        op: op.into(),
        weight: None,
        activation: None,
        output: None,
        in_dim: k,
        out_dim: n,
        rows,
        runtime: Default::default(),
    }
}

fn emu(
    body: &str,
    family: &str,
    behaviour: Behaviour,
    shape: Shape,
) -> (metrale_accuracy::contract::Contract, Emu) {
    let c = contract(body);
    let f = families()
        .families
        .into_iter()
        .find(|f| f.id == family)
        .unwrap();
    let wrong: (String, common::Wrong) = ("w4a16_gemv::w4a16_gemv_qg".to_string(), |case| {
        let linear = metrale_accuracy::refs::Reference::parse("linear").unwrap();
        let mut r = metrale_accuracy::inputs::SplitMix64::new(0);
        linear
            .mutate(case, &Mutation::SwapScaleGranularity, &mut r)
            .map(|_| ())
    });
    let e = Emu {
        contract: c.clone(),
        family: f,
        behaviour,
        wrong,
        shape,
    };
    (c, e)
}

#[test]
fn nvfp4_w4a16_gemv_passes_every_class_and_catches_every_mutation() {
    let s = shape("linear", 2, 5120, 1024);
    let (c, mut e) = emu(W4A16_SW, "w4a16_gemv", Behaviour::Conforming, s.clone());
    for input in c.inputs.clone() {
        let job = Job {
            contract: &c,
            family: &e.family.clone(),
            kernel: &c.kernels[0],
            point: &Default::default(),
            shape: &s,
            input,
            seed: 20261009,
        };
        let o = run(&job, &mut e);
        assert_eq!(o.verdict, Verdict::Pass, "{}: {o:#?}", input.name());
        let good = o.good.unwrap();
        assert!(
            good.ratio < 0.5,
            "{}: good ratio {} leaves no margin",
            input.name(),
            good.ratio
        );
        assert!(good.compared > 100);
        if input == InputClass::Gaussian {
            assert_eq!(o.mutations.len(), 4);
            for m in &o.mutations {
                assert!(m.ratio > 2.0, "{} caught only at ratio {}", m.name, m.ratio);
            }
        }
    }
}

#[test]
fn a_split_lm_head_catches_an_off_by_one_shard() {
    // 2026-10-09: GLM's capped vocabulary, not a multiple of 64: the shard edges fall at 51648
    // and 103296; scaled down 16x to keep the CPU test fast while keeping the misalignment.
    let s = shape("lm_head", 1, 256, 154_856 / 16);
    let (c, mut e) = emu(DENSE_SPLIT, "dense_bf16", Behaviour::Conforming, s.clone());
    let job = Job {
        contract: &c,
        family: &e.family.clone(),
        kernel: &c.kernels[0],
        point: &Default::default(),
        shape: &s,
        input: InputClass::Gaussian,
        seed: 1,
    };
    let o = run(&job, &mut e);
    assert_eq!(o.verdict, Verdict::Pass, "{o:#?}");
    let split = o
        .mutations
        .iter()
        .find(|m| m.name == "split_off_by_one")
        .unwrap();
    assert!(split.ratio > 2.0);
}

#[test]
fn a_kernel_that_breaks_its_declaration_fails_the_bound() {
    let s = shape("linear", 1, 5120, 512);
    let (c, mut e) = emu(
        W4A16_SW,
        "w4a16_gemv",
        Behaviour::Accumulator(BF16),
        s.clone(),
    );
    let job = Job {
        contract: &c,
        family: &e.family.clone(),
        kernel: &c.kernels[0],
        point: &Default::default(),
        shape: &s,
        input: InputClass::Gaussian,
        seed: 3,
    };
    assert_eq!(run(&job, &mut e).verdict, Verdict::FailBound);
}

#[test]
fn a_contract_too_loose_to_catch_a_mutation_fails_the_run() {
    // 2026-10-09: Declaring a million-term sequential chain makes the bound vacuous in practice;
    // the corrupted scale then stays inside it and the run must fail, not pass.
    let loose = W4A16_SW.replace("width = \"k/2048\"", "width = \"1000000\"");
    let s = shape("linear", 1, 5120, 512);
    let (c, mut e) = emu(&loose, "w4a16_gemv", Behaviour::Conforming, s.clone());
    let job = Job {
        contract: &c,
        family: &e.family.clone(),
        kernel: &c.kernels[0],
        point: &Default::default(),
        shape: &s,
        input: InputClass::Gaussian,
        seed: 3,
    };
    let o = run(&job, &mut e);
    assert!(
        matches!(o.verdict, Verdict::FailMutationPassed(_)),
        "{:?}",
        o.verdict
    );
}
