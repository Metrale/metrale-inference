// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `check::run_on`, the envelope sweep's check on shared operands: a conforming
//! kernel judged on a case built by another job of the same shape and formats passes, as on its
//! own case; a kernel that breaks its declaration fails on the shared case too; and the shared
//! case is the one judged (its entry point is replaced, its operands are not).

mod common;

use common::{Behaviour, Emu, contract, families};
use metrale_accuracy::check::{Job, Verdict, case_of, run_on};
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
        point: Default::default(),
    };
    (c, e)
}

fn job<'a>(
    c: &'a metrale_accuracy::contract::Contract,
    f: &'a metrale_circuit::venn::families::Family,
    s: &'a Shape,
    seed: u64,
) -> Job<'a> {
    Job {
        contract: c,
        family: f,
        kernel: &c.kernels[0],
        point: Box::leak(Box::default()),
        shape: s,
        input: InputClass::Gaussian,
        seed,
    }
}

#[test]
fn a_conforming_kernel_passes_on_operands_another_job_built() {
    let s = shape("linear", 4, 4096, 512);
    let (c, mut e) = emu(W4A16_SW, "w4a16_gemv", Behaviour::Conforming, s.clone());
    let f = e.family.clone();
    // 2026-10-10: The shared case comes from a job with another seed (another key): other
    // operands than the judged job would draw itself.
    let shared = case_of(&job(&c, &f, &s, 7)).expect("case");
    let o = run_on(&job(&c, &f, &s, 20261009), &shared, &mut e);
    assert_eq!(o.verdict, Verdict::Pass, "{o:#?}");
    let own = case_of(&job(&c, &f, &s, 20261009)).expect("case");
    assert_ne!(
        own.tensor("w").unwrap().bytes,
        shared.tensor("w").unwrap().bytes,
        "the control: the two jobs draw different weights"
    );
}

#[test]
fn a_kernel_that_breaks_its_declaration_fails_on_shared_operands() {
    let s = shape("linear", 1, 5120, 512);
    let (c, mut e) = emu(
        W4A16_SW,
        "w4a16_gemv",
        Behaviour::Accumulator(BF16),
        s.clone(),
    );
    let f = e.family.clone();
    let shared = case_of(&job(&c, &f, &s, 7)).expect("case");
    let o = run_on(&job(&c, &f, &s, 20261009), &shared, &mut e);
    assert_eq!(o.verdict, Verdict::FailBound, "{o:#?}");
}
