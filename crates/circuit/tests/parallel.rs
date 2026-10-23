// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Rank circuits from the one circuit (crates/circuit/src/parallel.rs): N = 1 is the
//! whole model on every instance; GLM-5.3 at its served TP=3 has the engine's rank shapes and a
//! collective at every sharding change and nowhere else; a dropped declaration drops its
//! collective; a part on two of three ranks idles the third and crosses its group boundary over
//! the group's communicator; and for every N <= 8 and degree d <= N the collectives balance across
//! the ranks or the layout is refused with its reason.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod common;

use std::collections::BTreeMap;

use metrale_circuit::ir::{Circuit, OpKind};
use metrale_circuit::{Deployment, Instance};

fn glm() -> Instance {
    common::instances()
        .into_iter()
        .find(|i| i.recipe == "glm-5.3/glm-5.3-flash-nvfp4-tp3-512k")
        .expect("the tp3 instance")
}

fn spec_of(text: &common::Texts) -> metrale_circuit::parallel::ParallelSpec {
    let blocks: Vec<(&str, &str)> = text.blocks.iter().map(|(n, s)| (n.as_str(), s.as_str())).collect();
    metrale_circuit::parallel_spec(&text.circuit, &blocks).unwrap()
}

fn deploy(inst: &Instance, ranks: u64, degree: BTreeMap<String, u64>) -> Instance {
    let mut i = inst.clone();
    i.parallel = Some(Deployment {
        ranks,
        degree,
        why: "test".into(),
    });
    i
}

fn uniform(spec: &metrale_circuit::parallel::ParallelSpec, d: u64) -> BTreeMap<String, u64> {
    metrale_circuit::parallel::parts(spec).map(|k| (k.clone(), d)).collect()
}

fn ranks_of(inst: &Instance, t: &common::Texts) -> Vec<Circuit> {
    common::load_texts(inst, t)
        .unwrap_or_else(|e| panic!("{}: {e}", inst.recipe))
        .ranks
}

fn collectives(c: &Circuit) -> Vec<(String, OpKind, String)> {
    c.nodes
        .iter()
        .filter(|n| matches!(n.op, OpKind::AllReduce | OpKind::AllGather | OpKind::Broadcast))
        .map(|n| (n.id.clone(), n.op, n.params.get("group").cloned().unwrap_or_default()))
        .collect()
}

fn consumer_op(c: &Circuit, collective: &str) -> OpKind {
    let n = &c.nodes[c.node(collective).unwrap()];
    let out = &c.edges[n.outputs[0]];
    c.nodes[out.consumers[0]].op
}

/// 2026-10-23: At N = 1 every part is replicated, and the rank circuit IS the whole-model circuit
/// on every instance, field for field (the N = 1 identity the owner asked for).
#[test]
fn one_rank_is_the_whole_model_on_every_instance() {
    for inst in common::instances() {
        let t = common::Texts::of(&inst);
        let one = deploy(&inst, 1, uniform(&spec_of(&t), 1));
        let loaded = common::load_texts(&one, &t).unwrap_or_else(|e| panic!("{}: {e}", inst.recipe));
        assert_eq!(loaded.ranks.len(), 1);
        assert!(loaded.ranks[0] == loaded.circuit, "{}: the N=1 rank differs", inst.recipe);
    }
}

/// 2026-10-23: The served layout's rank shapes are the engine's: KDA heads 22/22/20 (pairs), DSA
/// heads 22/21/21, routed 704/704/640, shared 768/768/512, dense 4096, vocab 51648/51648/51584;
/// the stream stays whole and only the marked eh_proj row is 1408/1344/1344.
#[test]
fn glm_at_its_served_layout_has_the_engines_rank_shapes() {
    let inst = glm();
    let rs = ranks_of(&inst, &common::Texts::of(&inst));
    let col = |d: &str| rs.iter().map(|c| c.dims[d]).collect::<Vec<_>>();
    assert_eq!(col("lin_k_heads"), [22, 22, 20]);
    assert_eq!(col("q_heads"), [22, 21, 21]);
    assert_eq!(col("moe_inter"), [704, 704, 640]);
    assert_eq!(col("shared_inter"), [768, 768, 512]);
    assert_eq!(col("inter"), [4096, 4096, 4096]);
    assert_eq!(col("vocab"), [51648, 51648, 51584]);
    assert_eq!(col("hidden"), [4096, 4096, 4096]);
    let h0 = |c: &Circuit| c.edges[c.edge("draft.glm_mtp_in.h0").unwrap()].dim_value;
    assert_eq!(rs.iter().map(h0).collect::<Vec<_>>(), [1408, 1344, 1344]);
}

/// 2026-10-23: Every rank derives the same 95 collectives the served step launches (90 for the
/// 45 layers' mixer and FFN sites, each into the post-mix; the head's gather into the argmax; the
/// draft's eh_proj gather, its two partial sums into the residual adds, its vocab gather), all
/// over the world.
#[test]
fn glm_collectives_are_derived_at_every_sharding_change() {
    let inst = glm();
    for c in ranks_of(&inst, &common::Texts::of(&inst)) {
        let all = collectives(&c);
        let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
        for (id, op, _) in &all {
            let site = id.split_once('.').map_or(id.as_str(), |(_, r)| r);
            *kinds.entry(format!("{op:?} {site}")).or_default() += 1;
        }
        assert_eq!(all.len(), 95, "{kinds:#?}");
        assert!(all.iter().all(|(_, _, g)| g == "0,1,2"));
        let (main, draft): (Vec<_>, Vec<_>) = all.iter().partition(|(id, _, _)| !id.starts_with("draft."));
        let ar_main: Vec<_> = main.iter().filter(|(_, op, _)| *op == OpKind::AllReduce).collect();
        assert_eq!(ar_main.len(), 90);
        assert!(ar_main.iter().all(|(id, _, _)| consumer_op(&c, id) == OpKind::HcPost), "every layer partial sums into the post-mix");
        assert_eq!(
            main.iter().filter(|(_, op, _)| *op == OpKind::AllGather).map(|(id, _, _)| id.as_str()).collect::<Vec<_>>(),
            ["head.logits@all_gather"]
        );
        let d: Vec<(&str, OpKind)> = draft.iter().map(|(id, op, _)| (id.as_str(), *op)).collect();
        assert_eq!(
            d,
            [
                ("draft.mtp_dsa.x@all_gather", OpKind::AllGather),
                ("draft.mtp_dsa.o@all_reduce", OpKind::AllReduce),
                ("draft.mtp_moe.f@all_reduce", OpKind::AllReduce),
                ("draft.glm_mtp_out.logits@all_gather", OpKind::AllGather),
            ]
        );
    }
}

/// 2026-10-23: The negative of the derivation: without the routed and shared expert splits the
/// MoE sites hold whole tensors and their 42 all-reduces vanish (the 45 mixer sites, the 3 dense
/// FFN sites remain), as does the draft MoE's.
#[test]
fn dropping_a_declaration_drops_its_collective() {
    let inst = glm();
    let mut t = common::Texts::of(&inst);
    let ffn_text = t.blocks.iter().map(|(_, s)| s.clone()).collect::<String>();
    assert!(!ffn_text.contains("[parallel"), "the spec lives in the circuit file only");
    for dim in ["moe_inter", "shared_inter"] {
        let start = t.circuit.find(&format!("[parallel.{dim}]")).unwrap();
        let end = t.circuit[start..].find("\n\n").unwrap() + start;
        t.circuit.replace_range(start..end, "");
    }
    let degree = uniform(&spec_of(&t), 3);
    assert!(!degree.contains_key("moe_inter"));
    let rs = ranks_of(&deploy(&inst, 3, degree), &t);
    let ar = |c: &Circuit, draft: bool| {
        collectives(c)
            .iter()
            .filter(|(id, op, _)| *op == OpKind::AllReduce && id.starts_with("draft.") == draft)
            .count()
    };
    assert_eq!(ar(&rs[0], false), 48);
    assert_eq!(ar(&rs[0], true), 1, "only the draft attention's partial remains");
}

/// 2026-10-23: The shared expert on ranks {0, 1} of three, the rest on all three: rank 2 runs no
/// shared-expert node; ranks 0 and 1 sum its partials over the {0, 1} communicator and broadcast
/// from rank 0; rank 2 only receives the broadcast; the routed partial, no longer blended with a
/// partial of its own group, is summed over the world before the blend.
#[test]
fn a_part_on_two_of_three_ranks_idles_the_third_and_crosses_its_boundary_on_its_group() {
    let inst = glm();
    let t = common::Texts::of(&inst);
    let mut degree = uniform(&spec_of(&t), 3);
    degree.insert("shared_inter".into(), 2);
    let rs = ranks_of(&deploy(&inst, 3, degree), &t);
    let site = "l5.moe";
    let ops = |c: &Circuit| {
        c.nodes
            .iter()
            .filter(|n| n.id.starts_with(&format!("{site}.")))
            .map(|n| (n.local.clone(), n.op, n.params.get("group").cloned(), n.params.get("root").cloned(), n.inputs.len()))
            .collect::<Vec<_>>()
    };
    let has = |c: &Circuit, local: &str| c.node(&format!("{site}.{local}")).is_some();
    for c in &rs[..2] {
        assert!(has(c, "shared_gate_up") && has(c, "shared_down"));
        let o = ops(c);
        let sd = o.iter().find(|x| x.0 == "sdown@all_reduce").expect("the shared partial sums");
        assert_eq!(sd.2.as_deref(), Some("0,1"), "over the shared expert's group, not the world");
        let bc = o.iter().find(|x| x.0 == "sdown@broadcast").expect("then leaves the group");
        assert_eq!((bc.2.as_deref(), bc.3.as_deref(), bc.4), (Some("0,1,2"), Some("0"), 1));
        let ed = o.iter().find(|x| x.0 == "edown@all_reduce").expect("the routed partial sums alone");
        assert_eq!(ed.2.as_deref(), Some("0,1,2"));
    }
    let r2 = &rs[2];
    assert_eq!(r2.dims["shared_inter"], 0, "rank 2 holds no shared slice");
    for local in ["shared_gate_up", "shared_clamp", "shared_act", "shared_down", "sdown@all_reduce"] {
        assert!(!has(r2, local), "rank 2 idles for `{local}`");
    }
    let o = ops(r2);
    let bc = o.iter().find(|x| x.0 == "sdown@broadcast").expect("rank 2 receives the shared result");
    assert_eq!((bc.2.as_deref(), bc.3.as_deref(), bc.4), (Some("0,1,2"), Some("0"), 0));
    assert!(has(r2, "blend") && has(r2, "post"));
}

/// 2026-10-23: For N in 1..=8 and every degree d <= N (every part at d or its own cap, then the
/// shared expert one lower; a part past its cap is refused): the layout is refused with the part's reason, or every rank instantiates; over each
/// communicator its members derive the same collectives in the same order (nothing deadlocks);
/// each split dim's slices over its group add up to the padded dim.
#[test]
fn collectives_balance_at_every_rank_count_and_degree() {
    let inst = glm();
    let t = common::Texts::of(&inst);
    let spec = spec_of(&t);
    let mut checked = 0;
    for n in 1..=8u64 {
        for d in 1..=n {
            // 2026-10-23: Each part at d, or at its own cap where d passes it (the vocab
            // exchange's 4): a part's degree is its own, below N.
            let capped: BTreeMap<String, u64> = metrale_circuit::parallel::parts(&spec)
                .map(|k| (k.clone(), d.min(spec[k].degree.max)))
                .collect();
            let mut layouts = vec![capped.clone()];
            if d > 1 {
                let mut m = capped.clone();
                m.insert("shared_inter".into(), (d - 1).min(spec["shared_inter"].degree.max));
                layouts.push(m);
            }
            if d > spec["vocab"].degree.max {
                let refused = common::load_texts(&deploy(&inst, n, uniform(&spec, d)), &t);
                assert!(refused.is_err_and(|e| e.to_string().contains("`vocab`")), "n={n} d={d}: the vocab cap");
            }
            for degree in layouts {
                let i = deploy(&inst, n, degree.clone());
                let rs = match common::load_texts(&i, &t) {
                    Ok(l) => l.ranks,
                    Err(e) => {
                        let why = e.to_string();
                        assert!(why.contains("parallel"), "n={n} d={d}: refused without a parallel reason: {why}");
                        continue;
                    }
                };
                checked += 1;
                let mut per_comm: BTreeMap<String, BTreeMap<u64, Vec<String>>> = BTreeMap::new();
                for (rank, c) in rs.iter().enumerate() {
                    for (id, op, g) in collectives(c) {
                        per_comm.entry(g).or_default().entry(rank as u64).or_default().push(format!("{id}:{op:?}"));
                    }
                }
                for (g, by_rank) in &per_comm {
                    let members: Vec<u64> = g.split(',').map(|r| r.parse().unwrap()).collect();
                    let none = Vec::new();
                    let seqs: Vec<&Vec<String>> = members.iter().map(|r| by_rank.get(r).unwrap_or(&none)).collect();
                    assert!(seqs.windows(2).all(|w| w[0] == w[1]), "n={n} d={d} comm {g}: members disagree");
                    assert!(by_rank.keys().all(|r| members.contains(r)), "n={n} d={d}: a non-member on comm {g}");
                }
                for (dim, s) in &spec {
                    if s.scope != metrale_circuit::parallel::Scope::Every || degree[metrale_circuit::parallel::part_of(&spec, dim)] == 1 {
                        continue;
                    }
                    let sum: u64 = rs.iter().map(|c| c.dims[dim]).sum();
                    let whole = inst.shape.dims[dim];
                    assert_eq!(sum, whole.div_ceil(s.unit) * s.unit, "n={n} d={d} {dim}");
                }
            }
        }
    }
    assert_eq!(checked, 64, "the property ran on {checked} layouts");
}

/// 2026-10-23: The same mechanism on another model (FACE): Qwen3.6-35B-A3B NVFP4 at N = 2 shards
/// its attention and linear attention by head and keeps the MoE whole, so every layer's mixer
/// partial is summed once before its residual add, no MoE site has a collective, and the head
/// gathers its vocab slices. At N = 3 its even-only head split refuses.
#[test]
fn qwen_moe_at_two_ranks_shards_the_mixers_and_keeps_the_moe_whole() {
    let inst = common::instances()
        .into_iter()
        .find(|i| i.recipe == "qwen3.6/qwen3.6-35b-a3b-nvfp4-declared")
        .unwrap();
    let t = common::Texts::of(&inst);
    let spec = spec_of(&t);
    let rs = ranks_of(&deploy(&inst, 2, uniform(&spec, 2)), &t);
    let layers = inst.shape.layer_kinds.len();
    for c in &rs {
        assert_eq!(
            (c.dims["q_heads"], c.dims["kv_heads"], c.dims["lin_k_heads"], c.dims["lin_v_heads"]),
            (8, 1, 8, 16)
        );
        assert_eq!((c.dims["moe_inter"], c.dims["experts"]), (512, 256), "the MoE is whole");
        let all = collectives(c);
        let main: Vec<_> = all.iter().filter(|(id, _, _)| !id.starts_with("draft.")).collect();
        let ar: Vec<_> = main.iter().filter(|(_, op, _)| *op == OpKind::AllReduce).collect();
        assert_eq!(ar.len(), layers);
        assert!(ar.iter().all(|(id, _, _)| consumer_op(c, id) == OpKind::ResidualAdd));
        assert!(all.iter().all(|(id, _, _)| !id.contains(".moe_ffn.")), "no MoE collective");
        assert_eq!(main.iter().filter(|(_, op, _)| *op == OpKind::AllGather).count(), 1);
    }
    let three = common::load_texts(&deploy(&inst, 3, uniform(&spec, 3)), &t);
    assert!(three.is_err_and(|e| e.to_string().contains("kv_heads")));
}

/// 2026-10-23: Every `[parallel]` split of every circuit cites engine files that exist, at lines
/// they have: a split rule is checkable against the code that splits.
#[test]
fn every_parallel_split_cites_the_engine() {
    let mut cited = 0;
    for inst in common::instances() {
        let t = common::Texts::of(&inst);
        for (dim, s) in spec_of(&t) {
            for word in s.why.split([' ', ',', ';', '(', ')']) {
                let Some((file, lines)) = word.split_once(".rs:") else {
                    continue;
                };
                let file = format!("{file}.rs");
                let text = std::fs::read_to_string(common::root().join(&file))
                    .unwrap_or_else(|_| panic!("{}: `{dim}` cites {file}, which does not exist", inst.arch));
                let max = lines
                    .split(['-', ','])
                    .filter_map(|n| n.trim_end_matches(|c: char| !c.is_ascii_digit()).parse::<usize>().ok())
                    .max()
                    .unwrap();
                assert!(max <= text.lines().count(), "`{dim}` cites {file}:{max}, past its end");
                cited += 1;
            }
        }
    }
    assert!(cited >= 12, "only {cited} citations parsed");
}
