// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Both arms of the gated delta rule decode contracts of
//! kernels/gb10/common/ACCURACY.toml, proven on the CPU against the conforming emulation, at
//! the two GDN geometries the sweep runs (Qwen3.6-35B-A3B: 16 key / 32 value heads;
//! Qwen3.8-27B: 16 / 48; heads of 128): a conforming "kernel" passes every input class, judged
//! on `o` and on the new state; a one-step-stale state and a BF16 accumulator are caught; a
//! kernel that accumulates in BF16 fails the bound; a contract made too loose fails because a
//! mutation passes.

mod common;

use common::{Behaviour, Emu, Tree, families};
use metrale_accuracy::case::{Case, Enc};
use metrale_accuracy::check::{Job, Outcome, Verdict, run};
use metrale_accuracy::contract::{Class, Contract, Level, parse_contracts};
use metrale_accuracy::elem::{BF16, F32};
use metrale_accuracy::inputs::InputClass;
use metrale_accuracy::plan;
use metrale_accuracy::points::Shape;
use metrale_accuracy::refs::Reference;
use metrale_circuit::venn::Repo;

/// 2026-10-09: The committed derived `gdn_recurrence` contracts (strided first).
fn derived() -> Vec<Contract> {
    let text = Tree.read("kernels/gb10/common/ACCURACY.toml").unwrap();
    let all = parse_contracts(&text).unwrap();
    let mut v: Vec<Contract> = all
        .contracts
        .into_iter()
        .filter(|c| c.reference == "gdn_recurrence" && c.class == Class::Derived)
        .collect();
    v.sort_by_key(|c| c.family != "gdn_recurrence_strided");
    assert_eq!(v.len(), 2, "{v:#?}");
    v
}

/// 2026-10-09: A decode point of `v_heads` value heads over 16 key heads of 128.
fn shape(rows: u64, v_heads: u64) -> Shape {
    let rt = [
        ("k_heads", 16u64),
        ("k_dim", 128),
        ("v_heads", v_heads),
        ("v_dim", 128),
    ];
    Shape {
        op: "gdn_recurrence".into(),
        weight: None,
        activation: Some("f32".into()),
        output: Some("f32".into()),
        in_dim: 2 * 16 * 128 + v_heads * 128,
        out_dim: v_heads * 128,
        rows,
        runtime: rt
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
    }
}

fn check(c: &Contract, behaviour: Behaviour, s: &Shape, input: InputClass) -> Outcome {
    check_kernel(c, c, behaviour, s, input)
}

/// 2026-10-09: Contract `c` checked against a CPU "kernel" that conforms to `kernel` (another
/// declaration of the same reference: a kernel with or without the state clamp).
fn check_kernel(
    c: &Contract,
    kernel: &Contract,
    behaviour: Behaviour,
    s: &Shape,
    input: InputClass,
) -> Outcome {
    let family = families()
        .families
        .into_iter()
        .find(|f| f.id == c.family)
        .unwrap();
    let mut e = Emu {
        contract: kernel.clone(),
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
fn the_committed_gdn_contracts_fit_the_families() {
    let text = Tree.read("kernels/gb10/common/ACCURACY.toml").unwrap();
    let all = parse_contracts(&text).unwrap();
    let problems = metrale_accuracy::jobs::validate(&all, &families());
    assert!(!problems.iter().any(|p| p.contains("gdn")), "{problems:#?}");
    let gdn = all
        .contracts
        .iter()
        .filter(|c| c.reference == "gdn_recurrence")
        .count();
    assert_eq!(gdn, 3);
}

#[test]
fn both_geometries_pass_every_class_and_catch_every_mutation() {
    for c in derived() {
        for v_heads in [32, 48] {
            let s = shape(2, v_heads);
            for &input in &c.inputs {
                let o = check(&c, Behaviour::Conforming, &s, input);
                let at = format!("{} {v_heads} heads, {}", c.family, input.name());
                assert_eq!(o.verdict, Verdict::Pass, "{at}: {o:#?}");
                let (good, floor) = (o.good.unwrap(), o.floor.unwrap());
                println!(
                    "{at}: good {:.3e} floor {:.3e} over {}",
                    good.ratio, floor.ratio, good.compared
                );
                // 2026-10-09: Inside the bound; the margin per output is asserted below.
                assert!(good.ratio < 1.0, "{at}: good ratio {}", good.ratio);
                assert!(
                    good.compared > 4 * v_heads as usize,
                    "{at}: {} compared",
                    good.compared
                );
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

/// 2026-10-09: The reference, plan and filled case of contract `c` at `s` for `input`.
fn filled(c: &Contract, s: &Shape, input: InputClass) -> (Reference, plan::Plan, Case) {
    let family = families()
        .families
        .into_iter()
        .find(|f| f.id == c.family)
        .unwrap();
    let r = Reference::parse(&c.reference).unwrap();
    let declared = plan::declared(&family, &c.kernels[0], &c.op, &Default::default()).unwrap();
    let p = plan::plan(
        c,
        declared.clone(),
        &r.lens(s, &declared),
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
    r.fill(&mut case, &p, s, input, 20261009, "gdn_cpu")
        .unwrap();
    (r, p, case)
}

/// 2026-10-09: The conforming emulation's largest err/bound over every `o` of every row and
/// over every 5th state element: (o, state).
fn margins(c: &Contract, s: &Shape, input: InputClass) -> (f64, f64) {
    let (r, p, case) = filled(c, s, input);
    let (rows, cols) = (case.out.0[0], case.out.0[1]);
    let o_len = s.out_dim as usize;
    let worst = |idx: Vec<usize>| -> f64 {
        let want = r.reference(&case, &p, &idx).unwrap();
        let got = r.emulate(&case, &p, None, 0, &idx).unwrap();
        want.iter()
            .zip(&got)
            .map(|(w, g)| {
                let (lo, hi) = F32.preimage(*g);
                (lo - w.v).max(w.v - hi).max(0.0) / w.e
            })
            .fold(0.0, f64::max)
    };
    let o = worst(
        (0..rows)
            .flat_map(|r| (0..o_len).map(move |c| r * cols + c))
            .collect(),
    );
    let st = worst(
        (0..rows)
            .flat_map(|r| (o_len..cols).step_by(5).map(move |c| r * cols + c))
            .collect(),
    );
    (o, st)
}

#[test]
fn the_output_keeps_half_its_bound_and_the_state_stays_inside_it() {
    // 2026-10-09: `o` ends a chain of reductions whose depth bound is loose: a margin of two
    // or more. A state element is `round(round(g*h) + round(k*u))`: the bound credits the last
    // rounding exactly and states the first at its worst case, which a conforming kernel can
    // nearly reach (a double rounding), so the state's margin is above one, not two.
    let c = &derived()[0];
    for v_heads in [32, 48] {
        for &input in &c.inputs {
            let (o, st) = margins(c, &shape(2, v_heads), input);
            println!(
                "{v_heads} heads, {}: o {o:.3e} state {st:.3e}",
                input.name()
            );
            assert!(o < 0.5, "{v_heads} heads, {}: o at {o}", input.name());
            assert!(st < 1.0, "{v_heads} heads, {}: state at {st}", input.name());
        }
    }
}

/// 2026-10-09: Contract `c` declaring the clamp of the 27B strided source: `state_max_norm =
/// 1000` and its sum of squares (k_dim terms in a thread, a 32-lane tree, four warps in
/// sequence).
fn clamped(c: &Contract) -> Contract {
    let level = |level: &str, width: &str, order: &str| Level {
        level: level.into(),
        width: width.into(),
        order: order.into(),
    };
    let mut c = c.clone();
    c.constants.insert("state_max_norm".into(), 1000.0);
    c.reduction
        .insert("state_k".into(), vec![level("thread", "k/1", "sequential")]);
    c.reduction.insert(
        "state_v".into(),
        vec![
            level("warp", "32", "tree"),
            level("warps", "k/32", "sequential"),
        ],
    );
    c
}

#[test]
fn the_committed_contracts_hold_the_decode_step_unclamped() {
    // 2026-10-09: q/k/v near 2^15 push every head's state norm far past 1000; unclamped, the
    // compared state reaches 1e9 and more.
    for c in derived() {
        assert_eq!(c.constants.get("state_max_norm"), Some(&f64::INFINITY));
        let s = shape(1, 32);
        let (r, p, case) = filled(&c, &s, InputClass::NearOverflow);
        let cols = case.out.0[1];
        let idx: Vec<usize> = (s.out_dim as usize..cols).step_by(97).collect();
        let want = r.reference(&case, &p, &idx).unwrap();
        let largest = want.iter().map(|w| w.v.abs()).fold(0.0, f64::max);
        assert!(largest > 1e6, "state value {largest}");
    }
}

#[test]
fn a_kernel_that_clamps_the_state_fails_the_unclamped_contract_where_norms_pass_1000() {
    // 2026-10-09: The GPU finding on the qwen3.8-27b target, reproduced on the CPU: the strided
    // kernel there clamps, so it fails the classes whose states pass a norm of 1000 and passes
    // the gaussian class, whose states stay far below it.
    let c = &derived()[0];
    let s = shape(2, 48);
    for (input, want) in [
        (InputClass::NearOverflow, Verdict::FailBound),
        (InputClass::Outliers, Verdict::FailBound),
        (InputClass::Gaussian, Verdict::Pass),
    ] {
        let o = check_kernel(c, &clamped(c), Behaviour::Conforming, &s, input);
        assert_eq!(o.verdict, want, "{}: {o:#?}", input.name());
    }
}

#[test]
fn the_clamped_declaration_passes_a_clamping_kernel_and_fails_one_that_does_not() {
    // 2026-10-09: The declaration the first contracts made (and the common source compiles):
    // a clamping kernel passes it with the clamp engaged, and the unclamped kernels the swept
    // targets compile fail it, as they failed on the GPU.
    let c = clamped(&derived()[0]);
    let s = shape(1, 32);
    let o = check(&c, Behaviour::Conforming, &s, InputClass::NearOverflow);
    assert_eq!(o.verdict, Verdict::Pass, "{o:#?}");
    let good = o.good.unwrap();
    assert!(good.max_err > 0.0 && good.ratio < 0.5, "{good:?}");
    let (r, p, case) = filled(&c, &s, InputClass::NearOverflow);
    let cols = case.out.0[1];
    let idx: Vec<usize> = (s.out_dim as usize..cols).step_by(97).collect();
    let want = r.reference(&case, &p, &idx).unwrap();
    let largest = want.iter().map(|w| w.v.abs()).fold(0.0, f64::max);
    assert!(largest > 1.0 && largest <= 1000.0, "state value {largest}");
    let o = check_kernel(
        &c,
        &derived()[0],
        Behaviour::Conforming,
        &s,
        InputClass::NearOverflow,
    );
    assert_eq!(o.verdict, Verdict::FailBound, "{o:#?}");
}

#[test]
fn a_kernel_that_accumulates_in_bf16_fails_the_bound() {
    let o = check(
        &derived()[0],
        Behaviour::Accumulator(BF16),
        &shape(1, 32),
        InputClass::Gaussian,
    );
    assert_eq!(o.verdict, Verdict::FailBound, "{o:#?}");
}

#[test]
fn a_contract_too_loose_to_catch_a_mutation_fails_the_run() {
    // 2026-10-09: A million-term chain per dot product makes the bound of every dot product
    // about 6% of its absolute sum: the BF16 accumulator then stays inside it.
    let mut loose = derived().remove(0);
    for l in loose.reduction.get_mut("k").unwrap() {
        if l.width == "k/4" {
            l.width = "1000000".into();
        }
    }
    let o = check(
        &loose,
        Behaviour::Conforming,
        &shape(1, 32),
        InputClass::Gaussian,
    );
    assert!(
        matches!(o.verdict, Verdict::FailMutationPassed(_)),
        "{:?}",
        o.verdict
    );
}

#[test]
fn a_point_without_its_head_geometry_is_refused() {
    let c = &derived()[0];
    let mut s = shape(1, 32);
    s.runtime.remove("v_heads");
    let o = check(c, Behaviour::Conforming, &s, InputClass::Gaussian);
    assert!(matches!(o.verdict, Verdict::Error(_)), "{:?}", o.verdict);
    let mut wrong = shape(1, 32);
    wrong.runtime.insert("v_heads".into(), "48".into());
    let o = check(c, Behaviour::Conforming, &wrong, InputClass::Gaussian);
    assert!(matches!(o.verdict, Verdict::Error(_)), "{:?}", o.verdict);
}
