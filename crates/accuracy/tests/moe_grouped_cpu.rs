// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Both arms of the grouped routed-expert contracts, proven on the CPU against the
//! conforming emulation, at the Qwen3.6-35B-A3B points (hidden 2048, moe_inter 512, 256
//! experts, top-8): a conforming "kernel" passes every input class with a margin; a zeroed
//! expert, a corrupted block scale and a BF16 accumulator are each caught; a kernel that breaks
//! its declaration fails the bound; a contract made too loose fails because a mutation passes.
//! The contracts are the checked-in ones, the pipelines the gb10 manifest's.

mod common;

use common::{Behaviour, Emu, Tree, families};
use metrale_accuracy::check::{Job, Verdict, run};
use metrale_accuracy::contract::{Contract, parse_contracts};
use metrale_accuracy::elem::BF16;
use metrale_accuracy::inputs::{InputClass, SplitMix64};
use metrale_accuracy::jobs::{self, Scope};
use metrale_accuracy::points::{Shape, sweep};
use metrale_accuracy::refs::moe_grouped as mg;
use metrale_circuit::venn::Repo;
use metrale_circuit::venn::families::Values;

const TC_GATE_UP: &str = "moe_nvfp4_grouped_tc::moe_expert_gate_up_act_nvfp4_grouped_tc";
const TC_DOWN: &str = "moe_nvfp4_grouped_tc::moe_expert_down_act_nvfp4_grouped_tc";
const FP8_GATE_UP: &str = "moe_shared_expert_fused_fp8_grouped::moe_expert_gate_up_act_fp8_grouped";
const FP8_DOWN: &str = "moe_shared_expert_fused_fp8_grouped::moe_expert_down_act_fp8_grouped";

fn contracts() -> Vec<Contract> {
    parse_contracts(&Tree.read("kernels/gb10/common/ACCURACY.toml").unwrap())
        .unwrap()
        .contracts
}

fn derived(kernel: &str) -> Contract {
    contracts()
        .into_iter()
        .find(|c| c.kernels.iter().any(|k| k == kernel) && c.against().is_none())
        .unwrap_or_else(|| panic!("no derived contract for {kernel}"))
}

trait Against {
    fn against(&self) -> Option<&str>;
}

impl Against for Contract {
    fn against(&self) -> Option<&str> {
        match &self.class {
            metrale_accuracy::contract::Class::BitIdentical { against } => Some(against),
            metrale_accuracy::contract::Class::Derived => None,
        }
    }
}

fn values(kv: &[(&str, &str)]) -> Values {
    kv.iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// 2026-10-09: The family point the 35B NVFP4 instance plans for `op` (the sweep's values).
fn nvfp4_point(op: &str) -> Values {
    if op == "expert_gate_up" {
        values(&[
            ("activation", "bf16"),
            ("epilogue", "silu_mul"),
            ("weight", "nvfp4/g16"),
            ("weight_layout", "row_major"),
        ])
    } else {
        values(&[
            ("down_input", "bf16"),
            ("weight", "nvfp4/g16"),
            ("weight_layout", "row_major"),
        ])
    }
}

/// 2026-10-09: A 35B expert node at `rows` tokens.
fn shape(op: &str, rows: u64) -> Shape {
    let (k, n) = if op == "expert_gate_up" {
        (2048, 1024)
    } else {
        (512, 2048)
    };
    Shape {
        op: op.into(),
        weight: None,
        activation: None,
        output: None,
        in_dim: k,
        out_dim: n,
        rows,
        runtime: values(&[("experts", "256"), ("top_k", "8")]),
    }
}

struct Setup {
    contract: Contract,
    emu: Emu,
    point: Values,
    shape: Shape,
}

fn setup(kernel: &str, rows: u64, behaviour: Behaviour, contract: Option<Contract>) -> Setup {
    let c = contract.unwrap_or_else(|| derived(kernel));
    let family = families()
        .families
        .into_iter()
        .find(|f| f.id == c.family)
        .unwrap();
    let point = if c.family == "moe_grouped_tc" {
        nvfp4_point(&c.op)
    } else {
        Values::new()
    };
    let shape = shape(&c.op, rows);
    let emu = Emu {
        contract: c.clone(),
        family,
        behaviour,
        wrong: (String::new(), |_| Ok(())),
        shape: shape.clone(),
        point: point.clone(),
    };
    Setup {
        contract: c,
        emu,
        point,
        shape,
    }
}

fn check(s: &mut Setup, input: InputClass, seed: u64) -> metrale_accuracy::check::Outcome {
    let family = s.emu.family.clone();
    let job = Job {
        contract: &s.contract,
        family: &family,
        kernel: &s.contract.kernels[0],
        point: &s.point,
        shape: &s.shape,
        input,
        seed,
    };
    run(&job, &mut s.emu)
}

/// 2026-10-09: The good arm must stay under `margin` of the bound. A projection's output
/// keeps half the bound free (0.5). The gate+up product sits behind the BF16 rounding of gate
/// and up: where the exact projection lies within the bound of a BF16 tie the reference admits
/// both outcomes with exactly their spread, so a kernel (or emulation) at the far outcome scores
/// just under 1 by construction; its margin is the bound itself.
fn passes_and_catches(kernel: &str, rows: u64, margin: f64) {
    let mut s = setup(kernel, rows, Behaviour::Conforming, None);
    for input in s.contract.inputs.clone() {
        let o = check(&mut s, input, 20261009);
        assert_eq!(
            o.verdict,
            Verdict::Pass,
            "{kernel} {}: {o:#?}",
            input.name()
        );
        let good = o.good.as_ref().unwrap();
        println!(
            "{kernel} rows={rows} {}: good {:.3e} floor {:.3e} {:?}",
            input.name(),
            good.ratio,
            o.floor.as_ref().unwrap().ratio,
            o.mutations
                .iter()
                .map(|m| (m.name.clone(), m.ratio))
                .collect::<Vec<_>>()
        );
        assert!(
            good.ratio < margin,
            "{kernel} {}: good ratio {} leaves no margin",
            input.name(),
            good.ratio
        );
        assert!(good.compared > 100, "{}", good.compared);
        if input == InputClass::Gaussian {
            assert_eq!(o.mutations.len(), s.contract.mutations.len());
            assert!(o.mutations.len() >= 2, "{:?}", o.mutations);
            for m in &o.mutations {
                assert!(
                    m.ratio > 2.0,
                    "{kernel}: {} caught only at {}",
                    m.name,
                    m.ratio
                );
            }
        }
    }
}

#[test]
fn nvfp4_tc_gate_up_passes_every_class_and_catches_every_mutation() {
    passes_and_catches(TC_GATE_UP, 16, 1.0);
}

#[test]
fn nvfp4_tc_down_passes_every_class_and_catches_every_mutation() {
    passes_and_catches(TC_DOWN, 16, 0.5);
}

#[test]
fn fp8_scalar_gate_up_passes_every_class_and_catches_every_mutation() {
    passes_and_catches(FP8_GATE_UP, 4, 1.0);
}

#[test]
fn fp8_scalar_down_passes_every_class_and_catches_every_mutation() {
    passes_and_catches(FP8_DOWN, 4, 0.5);
}

#[test]
fn one_token_routes_each_slot_to_its_own_expert_and_still_catches_the_mutations() {
    // 2026-10-09: The decode point: eight slots, eight distinct experts, one row each.
    passes_and_catches(TC_GATE_UP, 1, 1.0);
}

#[test]
fn a_kernel_that_accumulates_in_bf16_fails_the_bound() {
    for kernel in [TC_GATE_UP, TC_DOWN, FP8_DOWN] {
        let mut s = setup(kernel, 2, Behaviour::Accumulator(BF16), None);
        let o = check(&mut s, InputClass::Gaussian, 3);
        assert_eq!(o.verdict, Verdict::FailBound, "{kernel}: {:?}", o.good);
    }
}

#[test]
fn a_contract_too_loose_to_catch_a_mutation_fails_the_run() {
    // 2026-10-09: A million-term sequential chain per MMA makes the bound vacuous in practice;
    // the BF16 accumulator then stays inside it and the run must fail, not pass.
    let mut c = derived(TC_GATE_UP);
    for levels in c.reduction.values_mut() {
        for l in levels.iter_mut().filter(|l| l.level == "mma") {
            l.width = "1000000".into();
        }
    }
    let mut s = setup(TC_GATE_UP, 2, Behaviour::Conforming, Some(c));
    let o = check(&mut s, InputClass::Gaussian, 3);
    assert!(
        matches!(o.verdict, Verdict::FailMutationPassed(_)),
        "{:?}",
        o.verdict
    );
}

#[test]
fn an_undeclared_exponential_is_an_error_not_an_exact_exp() {
    let mut c = derived(TC_GATE_UP);
    c.approx.remove("ex2");
    let mut s = setup(TC_GATE_UP, 1, Behaviour::Conforming, Some(c));
    let o = check(&mut s, InputClass::Gaussian, 3);
    assert!(
        matches!(&o.verdict, Verdict::Error(e) if e.contains("ex2")),
        "{:?}",
        o.verdict
    );
}

#[test]
fn a_point_without_the_routing_sizes_is_refused() {
    let mut s = setup(TC_DOWN, 2, Behaviour::Conforming, None);
    s.shape.runtime.remove("experts");
    let o = check(&mut s, InputClass::Gaussian, 3);
    assert!(
        matches!(&o.verdict, Verdict::Error(e) if e.contains("experts")),
        "{:?}",
        o.verdict
    );
}

#[test]
fn the_contracts_fit_the_manifest_and_the_sweep_plans_them_at_the_35b_points() {
    let text = Tree.read("kernels/gb10/common/ACCURACY.toml").unwrap();
    let all = parse_contracts(&text).unwrap();
    let fams = families();
    assert_eq!(jobs::validate(&all, &fams), Vec::<String>::new());
    let s = sweep(&Tree, "gb10").unwrap();
    let (planned, _) = jobs::plan(&s, &all, &fams, Scope::Full, None, None);
    for k in [TC_GATE_UP, TC_DOWN, FP8_GATE_UP, FP8_DOWN] {
        let mine: Vec<_> = planned.iter().filter(|p| p.kernel == k).collect();
        assert!(!mine.is_empty(), "{k} is planned nowhere");
        for p in mine {
            assert_eq!(
                p.shape.runtime.get("experts").map(String::as_str),
                Some("256")
            );
            assert_eq!(p.shape.runtime.get("top_k").map(String::as_str), Some("8"));
        }
    }
}

#[test]
fn the_routing_is_seeded_distinct_per_token_and_skewed() {
    let draw = |seed| mg::zipf_routing(&mut SplitMix64::new(seed), 256, 8, 64);
    let a = draw(7);
    assert_eq!(a, draw(7));
    assert_ne!(a, draw(8));
    for row in a.chunks(8) {
        let mut r = row.to_vec();
        r.sort_unstable();
        r.dedup();
        assert_eq!(r.len(), 8, "{row:?}");
    }
    let by = mg::slots_by_expert(&a.iter().map(|&e| e as usize).collect::<Vec<_>>());
    // 2026-10-09: Skewed: fewer distinct experts than a uniform draw (about 220 of 256 at 512
    // slots) and one expert carrying more than a tensor-core pass (8 rows).
    assert!(by.len() < 200, "{} distinct experts", by.len());
    assert!(by.values().any(|s| s.len() > 8));
}

#[test]
fn sorted_rows_round_trip_and_a_pair_decodes_to_hi_plus_lo() {
    let perm = [2i32, 0, 3, 1];
    let by_slot: Vec<u8> = (0..8u8).collect();
    let by_pos = mg::rows_by_position(&by_slot, &perm, 2).unwrap();
    assert_eq!(by_pos, vec![2, 3, 6, 7, 0, 1, 4, 5]);
    assert_eq!(mg::rows_by_slot(&by_pos, &perm, 2).unwrap(), by_slot);
    assert!(mg::rows_by_slot(&by_pos, &[0, 1, 2, 9], 2).is_err());
    // 2026-10-09: One slot, n = 1: hi = 1.0, lo = 2^-9 decode to 1 + 2^-9; a sentinel half is
    // not a plausible value.
    let pair = [0x80u8, 0x3f, 0x00, 0x3b];
    let f = mg::pair_rows_by_slot(&pair, &[0], 1).unwrap();
    assert_eq!(
        f32::from_le_bytes([f[0], f[1], f[2], f[3]]),
        1.0 + 2f32.powi(-9)
    );
    let sent = mg::pair_rows_by_slot(&[0x7f; 4], &[0], 1).unwrap();
    assert!(f32::from_le_bytes([sent[0], sent[1], sent[2], sent[3]]).abs() > 1e38);
}
