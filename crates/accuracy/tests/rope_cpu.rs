// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Both arms of the gb10 RoPE contracts (the committed ACCURACY.toml rows), proven
//! on the CPU against the conforming emulation at the swept models' Q widths: a conforming
//! "kernel" passes on every input class with a margin; the wrong rope base and the bf16
//! rotation are caught; a kernel that rotates in bf16 fails the bound; a contract whose
//! declared approximations are too loose fails because a mutation passes; the strided twin
//! passes and catches the wrong base.

mod common;

use common::{Behaviour, Emu, Tree, Wrong, families};
use metrale_accuracy::case::Case;
use metrale_accuracy::check::{Job, Outcome, Verdict, run};
use metrale_accuracy::contract::{Class, Contract, parse_contracts};
use metrale_accuracy::elem::BF16;
use metrale_accuracy::inputs::InputClass;
use metrale_accuracy::points::Shape;
use metrale_circuit::venn::Repo;

/// 2026-10-09: The rope contract of `kernel` in `text`, derived or bit-identical.
fn contract_in(text: &str, kernel: &str, derived: bool) -> Contract {
    parse_contracts(text)
        .unwrap()
        .contracts
        .into_iter()
        .find(|c| c.kernels.iter().any(|k| k == kernel) && (c.class == Class::Derived) == derived)
        .unwrap_or_else(|| panic!("no contract for {kernel}"))
}

/// 2026-10-09: The committed contracts.
fn text() -> String {
    Tree.read("kernels/gb10/common/ACCURACY.toml").unwrap()
}

/// 2026-10-09: A swept rope shape: `rows` tokens, Q width `q_width`.
fn shape(rows: u64, q_width: u64) -> Shape {
    Shape {
        op: "rope".into(),
        weight: None,
        activation: Some("bf16".into()),
        output: Some("bf16".into()),
        in_dim: q_width,
        out_dim: q_width,
        rows,
        runtime: Default::default(),
    }
}

/// 2026-10-09: No rope contract names a wrong symbol; the runner's stand-in is never reached.
fn unreachable_symbol(_: &mut Case) -> Result<(), String> {
    Err("no wrong symbol in a rope contract".into())
}

/// 2026-10-09: One check of `c` on the CPU runner.
fn check(c: &Contract, s: &Shape, input: InputClass, behaviour: Behaviour, seed: u64) -> Outcome {
    let family = families()
        .families
        .into_iter()
        .find(|f| f.id == "rope")
        .unwrap();
    let wrong: (String, Wrong) = ("none::none".into(), unreachable_symbol);
    let mut e = Emu {
        contract: c.clone(),
        family: family.clone(),
        behaviour,
        wrong,
        shape: s.clone(),
    };
    let job = Job {
        contract: c,
        family: &family,
        kernel: &c.kernels[0],
        point: &Default::default(),
        shape: s,
        input,
        seed,
    };
    run(&job, &mut e)
}

/// 2026-10-09: The 35B's 16 Q heads (4096) over 2 tokens and the 27B's 24 (6144) over 5.
#[test]
fn rope_passes_every_class_and_catches_every_mutation() {
    let c = contract_in(&text(), "rope::rope_forward", true);
    for s in [shape(2, 4096), shape(5, 6144)] {
        for input in c.inputs.clone() {
            let o = check(&c, &s, input, Behaviour::Conforming, 20261009);
            let tag = format!("rope {} {}", s.in_dim, input.name());
            assert_eq!(o.verdict, Verdict::Pass, "{tag}: {o:#?}");
            let (good, floor) = (o.good.unwrap(), o.floor.unwrap());
            eprintln!(
                "{tag}: good {:.3e} floor {:.3e} compared {}",
                good.ratio, floor.ratio, good.compared
            );
            assert!(good.ratio < 0.5, "{tag}: good ratio {}", good.ratio);
            assert!(good.compared > 100);
            if input == InputClass::Gaussian {
                assert_eq!(o.mutations.len(), 2);
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

/// 2026-10-09: The strided twin at the 27B width (the CPU runner proves the wiring; the GPU
/// the bytes).
#[test]
fn the_strided_rope_matches_and_catches_the_wrong_base() {
    let c = contract_in(&text(), "rope::rope_forward_strided", false);
    for input in c.inputs.clone() {
        let o = check(&c, &shape(3, 6144), input, Behaviour::Conforming, 5);
        assert_eq!(o.verdict, Verdict::Pass, "{}: {o:#?}", input.name());
        if input == InputClass::Gaussian {
            assert!(o.mutations[0].ratio > 0.0, "{o:#?}");
        }
    }
}

/// 2026-10-09: A rope whose rotation runs in bf16.
#[test]
fn a_rope_that_rotates_in_bf16_fails_the_bound() {
    let c = contract_in(&text(), "rope::rope_forward", true);
    let o = check(
        &c,
        &shape(2, 4096),
        InputClass::Gaussian,
        Behaviour::Accumulator(BF16),
        3,
    );
    assert_eq!(o.verdict, Verdict::FailBound, "{o:#?}");
}

/// 2026-10-09: Loosened cos/sin declarations let the bf16 rotation through.
#[test]
fn a_rope_contract_too_loose_to_catch_a_mutation_fails_the_run() {
    // 2026-10-09: Declaring cosf and sinf good to 2^-4 widens the bound past a bf16 rotation.
    let loose = text().replacen(
        "approx = { pow = \"2^-51\", cos = \"2^-22\", sin = \"2^-22\" }",
        "approx = { pow = \"2^-51\", cos = \"2^-4\", sin = \"2^-4\" }",
        1,
    );
    assert_ne!(loose, text(), "the anchor no longer matches the contract");
    let c = contract_in(&loose, "rope::rope_forward", true);
    let o = check(
        &c,
        &shape(2, 4096),
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

/// 2026-10-09: A Q width that is not whole heads is an error, never a check.
#[test]
fn a_rope_over_partial_heads_is_refused() {
    let c = contract_in(&text(), "rope::rope_forward", true);
    let o = check(
        &c,
        &shape(1, 4000),
        InputClass::Gaussian,
        Behaviour::Conforming,
        1,
    );
    assert!(matches!(o.verdict, Verdict::Error(_)), "{o:#?}");
}
