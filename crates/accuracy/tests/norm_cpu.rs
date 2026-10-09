// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Both arms of the gb10 RMSNorm contracts (the committed ACCURACY.toml rows),
//! proven on the CPU against the conforming emulation at the swept models' shapes: a conforming
//! "kernel" passes on every input class with a margin; every mutation is caught; a kernel that
//! accumulates in bf16 fails the bound; a contract made too loose fails because a mutation
//! passes; the bit-identical twins pass and catch their wrong symbol.

mod common;

use common::{Behaviour, Emu, Tree, Wrong, families};
use metrale_accuracy::case::Case;
use metrale_accuracy::check::{Job, Outcome, Verdict, run};
use metrale_accuracy::contract::{Class, Contract, parse_contracts};
use metrale_accuracy::elem::BF16;
use metrale_accuracy::inputs::InputClass;
use metrale_accuracy::mutation::Mutation;
use metrale_accuracy::points::Shape;
use metrale_circuit::venn::Repo;
use metrale_circuit::venn::families::Values;

/// 2026-10-09: The contract file the tests read.
const FILE: &str = "kernels/gb10/common/ACCURACY.toml";

/// 2026-10-09: The committed contracts.
fn text() -> String {
    Tree.read(FILE).unwrap()
}

/// 2026-10-09: The rms_norm contract of `kernel` and `op` in `text`, derived or bit-identical.
fn contract_in(text: &str, kernel: &str, op: &str, derived: bool) -> Contract {
    parse_contracts(text)
        .unwrap()
        .contracts
        .into_iter()
        .find(|c| {
            c.family == "rms_norm"
                && c.kernels.iter().any(|k| k == kernel)
                && c.op == op
                && (c.class == Class::Derived) == derived
        })
        .unwrap_or_else(|| panic!("no contract for {kernel} {op}"))
}

/// 2026-10-09: A swept shape: `rows` tokens over an edge of `width`.
fn shape(op: &str, rows: u64, width: u64) -> Shape {
    Shape {
        op: op.into(),
        weight: None,
        activation: Some("bf16".into()),
        output: Some("bf16".into()),
        in_dim: width,
        out_dim: width,
        rows,
        runtime: Default::default(),
    }
}

/// 2026-10-09: The point the Qwen norms run at.
fn one_plus() -> Values {
    Values::from([("weight_form".to_string(), "one_plus".to_string())])
}

/// 2026-10-09: The wrong-symbol stand-in: the plain-weight copy multiplies by `w`, not `1 + w`.
fn flip_weight_form(case: &mut Case) -> Result<(), String> {
    let v = case.scalar("one_plus")?;
    case.scalars.insert("one_plus".into(), 1.0 - v);
    Ok(())
}

/// 2026-10-09: The contract's wrong symbol.
fn symbol_of(c: &Contract) -> String {
    c.mutations
        .iter()
        .find_map(|m| match m {
            Mutation::Symbol(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap()
}

/// 2026-10-09: One check of `c` on the CPU runner.
fn check(c: &Contract, s: &Shape, input: InputClass, behaviour: Behaviour, seed: u64) -> Outcome {
    let family = families()
        .families
        .into_iter()
        .find(|f| f.id == c.family)
        .unwrap();
    let wrong: (String, Wrong) = (symbol_of(c), flip_weight_form);
    // 2026-10-09: The runner emulates the declared computation, which a bit-identical contract
    // does not state: it runs the derived contract of the same op.
    let emulated = match c.class {
        Class::Derived => c.clone(),
        Class::BitIdentical { .. } => contract_in(&text(), "norm::rms_norm", &c.op, true),
    };
    let mut e = Emu {
        contract: emulated,
        family: family.clone(),
        behaviour,
        wrong,
        shape: s.clone(),
    };
    let job = Job {
        contract: c,
        family: &family,
        kernel: &c.kernels[0],
        point: &one_plus(),
        shape: s,
        input,
        seed,
    };
    run(&job, &mut e)
}

/// 2026-10-09: Every derived norm contract at a real point: 27B input norm (5120, 2 rows), 35B
/// final norm (2048, 3 rows), 27B k norm (4 heads of 256, 4 tokens), the fused residual-add
/// norms at 5120 and 2048.
#[test]
fn every_norm_contract_passes_every_class_and_catches_every_mutation() {
    let t = text();
    let cases = [
        ("norm::rms_norm", "rms_norm", shape("rms_norm", 2, 5120)),
        ("norm::rms_norm", "final_norm", shape("final_norm", 3, 2048)),
        ("norm::rms_norm", "qk_norm", shape("qk_norm", 4, 1024)),
        (
            "norm::residual_add_rms_norm",
            "rms_norm",
            shape("rms_norm", 2, 5120),
        ),
        (
            "residual_add_rms_norm_exact::residual_add_rms_norm_exact",
            "rms_norm",
            shape("rms_norm", 2, 2048),
        ),
    ];
    for (kernel, op, s) in cases {
        let c = contract_in(&t, kernel, op, true);
        for input in c.inputs.clone() {
            let o = check(&c, &s, input, Behaviour::Conforming, 20261009);
            let tag = format!("{kernel} {op} {}", input.name());
            assert_eq!(o.verdict, Verdict::Pass, "{tag}: {o:#?}");
            let (good, floor) = (o.good.unwrap(), o.floor.unwrap());
            eprintln!(
                "{tag}: good {:.3e} floor {:.3e} compared {}",
                good.ratio, floor.ratio, good.compared
            );
            assert!(good.ratio < 0.5, "{tag}: good ratio {}", good.ratio);
            assert!(good.compared > 100, "{tag}: {} compared", good.compared);
            if input == InputClass::Gaussian {
                assert_eq!(o.mutations.len(), c.mutations.len(), "{tag}");
                for m in &o.mutations {
                    eprintln!("{tag}: mutation {} ratio {:.3e}", m.name, m.ratio);
                    assert!(
                        m.ratio > 2.0,
                        "{tag}: {} caught only at {}",
                        m.name,
                        m.ratio
                    );
                }
            }
        }
    }
}

/// 2026-10-09: The twins on the CPU runner prove only the harness wiring (both sides are the
/// emulation); the bytes are compared on the GPU.
#[test]
fn the_bit_identical_twins_match_and_catch_their_wrong_symbol() {
    let t = text();
    let cases = [
        (
            "norm::rms_norm_strided",
            "qk_norm",
            shape("qk_norm", 3, 6144),
        ),
        (
            "norm::rms_norm_residual",
            "rms_norm",
            shape("rms_norm", 2, 2048),
        ),
        (
            "residual_add_rms_norm_exact::residual_add_rms_norm_exact",
            "rms_norm",
            shape("rms_norm", 1, 5120),
        ),
    ];
    for (kernel, op, s) in cases {
        let c = contract_in(&t, kernel, op, false);
        for input in c.inputs.clone() {
            let o = check(&c, &s, input, Behaviour::Conforming, 7);
            assert_eq!(
                o.verdict,
                Verdict::Pass,
                "{kernel} {}: {o:#?}",
                input.name()
            );
            assert_eq!(o.good.as_ref().unwrap().ratio, 0.0);
            if input == InputClass::Gaussian {
                assert!(o.mutations.iter().all(|m| m.ratio > 0.0), "{o:#?}");
            }
        }
    }
}

/// 2026-10-09: A norm that accumulates in bf16 at the 27B input norm and a 35B k norm.
#[test]
fn a_norm_that_accumulates_in_bf16_fails_the_bound() {
    let c = contract_in(&text(), "norm::rms_norm", "rms_norm", true);
    for (s, seed) in [
        (shape("rms_norm", 1, 5120), 3),
        (shape("qk_norm", 2, 512), 4),
    ] {
        let o = check(
            &c,
            &s,
            InputClass::Gaussian,
            Behaviour::Accumulator(BF16),
            seed,
        );
        assert_eq!(o.verdict, Verdict::FailBound, "{o:#?}");
    }
}

/// 2026-10-09: A loosened rsqrt declaration lets the bf16 accumulator through.
#[test]
fn a_norm_contract_too_loose_to_catch_a_mutation_fails_the_run() {
    // 2026-10-09: Declaring rsqrt good only to 2^-4 widens the bound past the error of a bf16
    // accumulator, so that mutation stays inside it and the run must fail.
    let t = text();
    let at = t
        .find("kernels = [\"norm::rms_norm\"]\nop = \"rms_norm\"")
        .unwrap();
    let (head, tail) = t.split_at(at);
    let loose = format!(
        "{head}{}",
        tail.replacen("rsqrt = \"2^-22\"", "rsqrt = \"2^-4\"", 1)
    );
    assert_ne!(loose, t, "the anchor no longer matches the contract");
    let c = contract_in(&loose, "norm::rms_norm", "rms_norm", true);
    let o = check(
        &c,
        &shape("rms_norm", 1, 5120),
        InputClass::Gaussian,
        Behaviour::Conforming,
        3,
    );
    assert_eq!(
        o.verdict,
        Verdict::FailMutationPassed("accumulate:bf16".into()),
        "{o:#?}"
    );
}

/// 2026-10-09: A partial head and a point without a weight form are errors, never checks.
#[test]
fn a_per_head_norm_needs_whole_heads_and_a_weight_form() {
    let c = contract_in(&text(), "norm::rms_norm", "qk_norm", true);
    let o = check(
        &c,
        &shape("qk_norm", 1, 1000),
        InputClass::Gaussian,
        Behaviour::Conforming,
        1,
    );
    assert!(matches!(o.verdict, Verdict::Error(_)), "{o:#?}");
    let family = families()
        .families
        .into_iter()
        .find(|f| f.id == "rms_norm")
        .unwrap();
    let s = shape("rms_norm", 1, 2048);
    let job = Job {
        contract: &c,
        family: &family,
        kernel: &c.kernels[0],
        point: &Values::new(),
        shape: &s,
        input: InputClass::Gaussian,
        seed: 1,
    };
    let mut e = Emu {
        contract: c.clone(),
        family: family.clone(),
        behaviour: Behaviour::Conforming,
        wrong: (symbol_of(&c), flip_weight_form),
        shape: s.clone(),
    };
    match run(&job, &mut e).verdict {
        Verdict::Error(m) => assert!(m.contains("weight_form"), "{m}"),
        v => panic!("{v:?}"),
    }
}

/// 2026-10-09: The committed norm and rope contracts name kernels of their families with
/// declared pipelines, and cover every swept norm and rope point except the ones listed: the
/// fused kernels' `residual_add` op (its output is one rounding of a sum; the norm they fuse is
/// contracted under `rms_norm`), the plain-weight points of a plan-only instance, and MRoPE.
#[test]
fn the_norm_and_rope_contracts_are_valid_and_cover_the_sweep() {
    let contracts = parse_contracts(&text()).unwrap();
    let fams = families();
    let problems = metrale_accuracy::jobs::validate(&contracts, &fams);
    assert!(
        problems
            .iter()
            .all(|p| !p.contains("`rms_norm`") && !p.contains("`rope`")),
        "{problems:#?}"
    );
    let sweep = metrale_accuracy::points::sweep(&Tree, "gb10").unwrap();
    for family in ["rms_norm", "rope"] {
        let (jobs, cov) = metrale_accuracy::jobs::plan(
            &sweep,
            &contracts,
            &fams,
            metrale_accuracy::jobs::Scope::Full,
            Some(family),
            None,
        );
        assert!(!jobs.is_empty(), "{family}: nothing planned");
        assert!(cov.unused.is_empty(), "{family}: {:?}", cov.unused);
        for ((_, kernels, op), why) in &cov.uncovered {
            eprintln!("{family}: uncovered {kernels} {op}: {why}");
            let expected = op == "residual_add"
                || kernels.starts_with('(')
                || kernels.contains("rope_forward_mrope_interleaved");
            assert!(expected, "{family}: {kernels} {op} uncovered: {why}");
        }
    }
}
