// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The deployment of N devices derived from each circuit's one `[parallel]` spec
//! (crates/circuit/src/deploy.rs), for N in 1..=8 on every instance whose circuit declares one
//! (GLM-5.3 and Qwen3.6 MoE): N = 1 is no deployment (the whole model); an admissible N
//! instantiates N rank circuits whose collectives match over every communicator and whose
//! per-rank plans fuse at the shapes a rank installs, under the target's compiled kernels, with
//! no rule refused; an inadmissible N is refused naming the part and dim that cannot split. The
//! GLM recipe's stated TP=3 deployment is this derivation at 3.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod common;

use std::collections::BTreeMap;

use metrale_circuit::ir::{Circuit, OpKind};
use metrale_circuit::{Instance, Mode, deploy};
use metrale_error::ErrorCode;

/// 2026-10-23: The rank counts the gate covers.
const MAX_RANKS: u64 = 8;
/// 2026-10-23: The shapes a tensor-parallel rank fuses (`install::GATED` + `DRAFT_GATED` at one
/// draft: decode 1, verify 2, draft 1).
const RANK_SHAPES: [(Mode, u64); 3] = [(Mode::Decode, 1), (Mode::Verify, 2), (Mode::Draft, 1)];

fn spec_of(t: &common::Texts) -> metrale_circuit::parallel::ParallelSpec {
    let blocks: Vec<(&str, &str)> = t.blocks.iter().map(|(n, s)| (n.as_str(), s.as_str())).collect();
    metrale_circuit::parallel_spec(&t.circuit, &blocks).unwrap()
}

fn find(recipe: &str) -> Instance {
    common::instances().into_iter().find(|i| i.recipe == recipe).expect(recipe)
}

/// 2026-10-23: Over every communicator, its members' collectives in order (nothing deadlocks),
/// and no non-member on it.
fn collectives_match(rs: &[Circuit]) -> Result<usize, String> {
    let mut per_comm: BTreeMap<String, BTreeMap<u64, Vec<String>>> = BTreeMap::new();
    for (rank, c) in rs.iter().enumerate() {
        for n in c.nodes.iter().filter(|n| matches!(n.op, OpKind::AllReduce | OpKind::AllGather | OpKind::Broadcast)) {
            let g = n.params.get("group").cloned().ok_or_else(|| format!("{} names no group", n.id))?;
            per_comm.entry(g).or_default().entry(rank as u64).or_default().push(format!("{}:{:?}", n.id, n.op));
        }
    }
    let mut total = 0;
    for (g, by_rank) in &per_comm {
        let members: Vec<u64> = g.split(',').map(|r| r.parse().unwrap()).collect();
        if by_rank.keys().any(|r| !members.contains(r)) {
            return Err(format!("a non-member on comm {g}"));
        }
        let none = Vec::new();
        let seqs: Vec<&Vec<String>> = members.iter().map(|r| by_rank.get(r).unwrap_or(&none)).collect();
        if !seqs.windows(2).all(|w| w[0] == w[1]) {
            return Err(format!("comm {g}: members disagree"));
        }
        total += seqs[0].len();
    }
    Ok(total)
}

/// 2026-10-23: What N gives on `inst`: `Ok(collectives per rank)` with every rank fused at the
/// rank shapes its plans list, or the refusal (code, text): `CircuitRanksRefused` from the
/// derivation, or `CircuitPlanUncovered` naming the first rank shape a rule does not cover.
fn at(inst: &Instance, n: u64) -> Result<(Vec<Circuit>, usize), (ErrorCode, String)> {
    let t = common::Texts::of(inst);
    let spec = spec_of(&t);
    let d = deploy::tensor_parallel(&spec, &inst.shape.dims, n).map_err(|e| (e.code(), e.to_string()))?;
    if n == 1 {
        assert!(d.is_none(), "{}: one device is the whole model", inst.recipe);
        return Ok((Vec::new(), 0));
    }
    let d = d.expect("a deployment over several devices");
    assert_eq!((d.ranks, d.degree.len()), (n, metrale_circuit::parallel::parts(&spec).count()));
    assert!(d.degree.values().all(|&k| k == n), "every part over all N ranks");
    let mut i = inst.clone();
    i.parallel = Some(d);
    let loaded = common::load_texts(&i, &t).unwrap_or_else(|e| panic!("{} N={n}: a placed layout must instantiate: {e}", inst.recipe));
    assert_eq!(loaded.ranks.len() as u64, n);
    let per = collectives_match(&loaded.ranks).unwrap_or_else(|e| panic!("{} N={n}: {e}", inst.recipe));
    let avail = common::available(&i, &loaded.rules);
    for (rank, c) in loaded.ranks.iter().enumerate() {
        for (mode, rows) in RANK_SHAPES.into_iter().filter(|&(m, r)| inst.plans.get(&m).is_some_and(|v| v.contains(&r))) {
            if let Err(e) = metrale_circuit::fuse(c, &loaded.rules, &avail, &inst.policy, mode, rows) {
                return Err((ErrorCode::CircuitPlanUncovered, format!("rank {rank} {mode:?} {rows}: {e}")));
            }
        }
    }
    Ok((loaded.ranks, per))
}

/// 2026-10-23: Every instance with a `[parallel]` spec at every N in 1..=8: admitted (matched
/// collectives, every rank fused) or refused with `CircuitRanksRefused` naming a part. GLM-5.3
/// is admitted at 1..=4 and refused at 5..=8 by the vocab part (the draft head's exchange holds
/// four ranks); Qwen3.6 MoE is admitted at 1 and 2 and refused at 3..=8 by its KV heads (two,
/// split exactly, `TpSupport::Even`).
#[test]
fn every_rank_count_is_admitted_or_refused_naming_its_part() {
    let expect: [(&str, &[u64], &str); 2] = [
        ("glm-5.3/glm-5.3-flash-nvfp4-tp3-512k", &[1, 2, 3, 4], "vocab"),
        ("qwen3.6/qwen3.6-35b-a3b-fp8-bf16head", &[1, 2], "kv_heads"),
    ];
    for (recipe, admitted, part) in expect {
        let inst = find(recipe);
        let mut ok = Vec::new();
        for n in 1..=MAX_RANKS {
            match at(&inst, n) {
                Ok((rs, per)) => {
                    assert!(n == 1 || per > 0, "{recipe} N={n}: a split model derives collectives");
                    assert_eq!(rs.len() as u64, if n == 1 { 0 } else { n });
                    ok.push(n);
                }
                Err((code, text)) => {
                    assert_eq!(code, ErrorCode::CircuitRanksRefused, "{recipe} N={n}: {text}");
                    assert!(text.contains(&format!("{n} ranks: part `{part}`")), "{recipe} N={n}: {text}");
                }
            }
        }
        assert_eq!(ok, admitted, "{recipe}: the admissible rank counts");
    }
}

/// 2026-10-23: The one-device GLM recipe (`glm-5.3-flash-nvfp4`, every lever at its code
/// default, `moe_expert_layout: ep`) is not tensor-parallel servable at any N: its spec places
/// (the derivation admits 2..=4 as for the TP recipe), but its levers leave a rank shape no rule
/// covers or read a setting the instance does not state. A refusal for its policy, not for N.
#[test]
fn the_one_device_glm_recipe_is_refused_by_its_policy_at_every_split() {
    let inst = find("glm-5.3/glm-5.3-flash-nvfp4");
    for n in 2..=MAX_RANKS {
        let (code, text) = at(&inst, n).err().unwrap_or_else(|| panic!("N={n} admitted"));
        if n <= 4 {
            assert_eq!(code, ErrorCode::CircuitPlanUncovered, "N={n}: {text}");
        } else {
            assert_eq!(code, ErrorCode::CircuitRanksRefused, "N={n}: {text}");
        }
    }
}

/// 2026-10-23: The GLM recipe's stated `[instance.parallel]` is the derivation at its N, and its
/// rank circuits are the derived ones: today's TP=3 circuits, field for field.
#[test]
fn the_stated_tp3_deployment_is_the_derived_one() {
    let inst = find("glm-5.3/glm-5.3-flash-nvfp4-tp3-512k");
    let stated = inst.parallel.clone().expect("the recipe states its deployment");
    let t = common::Texts::of(&inst);
    let d = deploy::tensor_parallel(&spec_of(&t), &inst.shape.dims, stated.ranks).unwrap().unwrap();
    assert_eq!((d.ranks, &d.degree), (stated.ranks, &stated.degree));
    let today = common::load_texts(&inst, &t).unwrap().ranks;
    let (derived, _) = at(&inst, stated.ranks).unwrap();
    assert!(today == derived, "the derived TP=3 rank circuits differ from today's");
}

/// 2026-10-23: The negative: a part whose split cannot reach N is refused at N, naming it and
/// its dim, while N - 1 is admitted; a tied dim names its part (the KDA value heads, tied to the
/// key heads) and the refusal carries the part's reason.
#[test]
fn a_part_that_cannot_split_refuses_naming_part_and_dim() {
    let inst = find("glm-5.3/glm-5.3-flash-nvfp4-tp3-512k");
    let t = common::Texts::of(&inst);
    let mut spec = spec_of(&t);
    spec.get_mut("lin_k_heads").unwrap().degree.max = 2;
    let e = deploy::tensor_parallel(&spec, &inst.shape.dims, 3).unwrap_err();
    assert_eq!(e.code(), ErrorCode::CircuitRanksRefused);
    let text = e.to_string();
    assert!(text.contains("3 ranks: part `lin_k_heads` (dim `lin_k_heads`)"), "{text}");
    assert!(text.contains("bounds 1..=2"), "{text}");
    assert!(deploy::tensor_parallel(&spec, &inst.shape.dims, 2).unwrap().is_some());
    let mut dims = inst.shape.dims.clone();
    dims.insert("lin_v_heads".into(), 62);
    let e = deploy::tensor_parallel(&spec_of(&t), &dims, 3).unwrap_err();
    assert!(e.to_string().contains("part `lin_k_heads` (dim `lin_v_heads`)"), "{e}");
}

