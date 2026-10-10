// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: M5 state edges: every node that touches a layer's state names it
//! (`state = { conv = "update" }`), and instantiation refuses a block whose states are not
//! touched as their kind requires. Checked over the instances in INSTANCES.toml (the dense and
//! MoE Qwen circuits, Nemotron-3.5-Lightning's Mamba2 layers) and over mutated block texts.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod common;

use metrale_circuit::state::{StateAccess, StateKind};
use metrale_circuit::{Circuit, LayerKind, OpKind, Section};

/// 2026-09-30: `(op, state local, access)` of every state reference of `c`'s layer `layer`.
fn refs(c: &Circuit, layer: usize, section: Section) -> Vec<(OpKind, String, StateAccess)> {
    let mut out: Vec<_> = c
        .nodes
        .iter()
        .filter(|n| n.layer == Some(layer))
        .flat_map(|n| {
            n.state
                .iter()
                .map(move |&(i, a)| (n.op, c.states[i].local.clone(), a))
        })
        .filter(|(_, l, _)| {
            c.states
                .iter()
                .any(|s| &s.local == l && s.layer == Some(layer) && s.section == section)
        })
        .collect();
    out.sort();
    out
}

#[test]
fn every_recurrent_and_attention_layer_names_the_state_it_touches() {
    for inst in common::instances() {
        let c = common::load(&inst).circuit;
        for (layer, kind) in c.layer_kinds.iter().enumerate() {
            let r = refs(&c, layer, Section::Main);
            let want: Vec<(OpKind, &str, StateAccess)> = match kind {
                LayerKind::LinearAttention => vec![
                    (OpKind::Conv1dUpdate, "conv", StateAccess::Update),
                    (OpKind::StateSnapshot, "conv", StateAccess::Snapshot),
                    (OpKind::GdnRecurrence, "h", StateAccess::Update),
                ],
                LayerKind::Mamba => vec![
                    (OpKind::Conv1dUpdate, "conv", StateAccess::Update),
                    (OpKind::StateSnapshot, "conv", StateAccess::Snapshot),
                    (OpKind::StateSnapshot, "h", StateAccess::Snapshot),
                    (OpKind::SsmUpdate, "h", StateAccess::Update),
                ],
                // 2026-10-10: Dense latent attention (`mla_moe`): one latent cache, written and
                // read, where a GQA layer keeps its K and V sides.
                LayerKind::FullAttention
                    if c.nodes
                        .iter()
                        .any(|n| n.layer == Some(layer) && n.op == OpKind::MlaAttention) =>
                {
                    vec![
                        (OpKind::KvWrite, "latent", StateAccess::Write),
                        (OpKind::MlaAttention, "latent", StateAccess::Read),
                    ]
                }
                LayerKind::FullAttention => vec![
                    (OpKind::KvWrite, "k", StateAccess::Write),
                    (OpKind::KvWrite, "v", StateAccess::Write),
                    (OpKind::PagedAttention, "k", StateAccess::Read),
                    (OpKind::PagedAttention, "v", StateAccess::Read),
                ],
                LayerKind::Moe => vec![],
                // 2026-10-08: GLM-5 DSA: the latent cache written and read, the pooled index
                // keys written by the pooling and read by the selection, the pool tail updated.
                LayerKind::SparseAttention => vec![
                    (OpKind::KvWrite, "latent", StateAccess::Write),
                    (OpKind::MlaAttention, "latent", StateAccess::Read),
                    (OpKind::KpoolCompress, "index", StateAccess::Write),
                    (OpKind::KpoolCompress, "tail", StateAccess::Update),
                    (OpKind::IndexSelect, "index", StateAccess::Read),
                ],
            };
            let mut want: Vec<_> = want
                .into_iter()
                .map(|(o, l, a)| (o, l.to_string(), a))
                .collect();
            want.sort();
            assert_eq!(r, want, "{} layer {layer} ({kind:?})", inst.recipe);
        }
        // 2026-09-30: The draft head's attention keeps its own KV cache. 2026-10-08: A sparse
        // attention draft (GLM-5's MTP layer) also keeps its own indexer pool tail.
        let draft: Vec<_> = c
            .states
            .iter()
            .filter(|s| s.section == Section::Draft)
            .collect();
        if !draft.is_empty() {
            assert!(
                draft.iter().all(|s| s.kind == StateKind::PagedKv
                    || (s.kind == StateKind::Recurrent && s.local == "tail")),
                "{}",
                inst.recipe
            );
        }
    }
}

/// 2026-09-30: The block text of `inst` with `from` replaced by `to`, instantiated.
fn mutated(recipe: &str, from: &str, to: &str) -> String {
    let inst = common::instances()
        .into_iter()
        .find(|i| i.recipe == recipe)
        .unwrap();
    let mut t = common::Texts::of(&inst);
    let mut hit = t.circuit.contains(from);
    t.circuit = t.circuit.replacen(from, to, 1);
    for (_, b) in &mut t.blocks {
        if !hit && b.contains(from) {
            *b = b.replacen(from, to, 1);
            hit = true;
        }
    }
    assert!(hit, "`{from}` is not in the texts of {recipe}");
    common::load_texts(&inst, &t)
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default()
}

#[test]
fn a_state_touched_wrongly_or_not_at_all_is_refused() {
    let dense = "qwen3.8/qwen3.8-27b-nvfp4-unsloth";
    let lightning = "nemotron-3.5/nemotron-3.5-lightning-30b-a3b-nvfp4";
    for (recipe, from, to, want) in [
        (
            dense,
            "state = { h = \"update\" }\n",
            "",
            "updated by exactly one node",
        ),
        (
            dense,
            "state = { conv = \"snapshot\" }",
            "state = { conv = \"update\" }",
            "updated by exactly one node",
        ),
        (
            dense,
            "state = { k = \"read\", v = \"read\" }\n",
            "",
            "read by at least one",
        ),
        (
            dense,
            "state = { h = \"update\" }",
            "state = { hh = \"update\" }",
            "declares no such state",
        ),
        (
            dense,
            "state = { h = \"update\" }",
            "state = { h = \"rewrite\" }",
            "is not read, write",
        ),
        (
            lightning,
            "state = { h = \"snapshot\" }",
            "state = { conv = \"snapshot\" }",
            "snapshotted by 2",
        ),
    ] {
        let e = mutated(recipe, from, to);
        assert!(e.contains(want), "{recipe}: `{from}` -> `{to}`: {e}");
    }
}

/// 2026-09-30: Every output a served circuit declares binds a model buffer: the head's logits
/// and tokens, and the draft head's embedding when there is a draft (LIFECYCLE-DESIGN.md 3.4).
#[test]
fn every_declared_output_of_a_served_circuit_binds_a_model_buffer() {
    use metrale_circuit::model_buffer::ModelBuffer;
    for inst in common::instances() {
        let c = common::load(&inst).circuit;
        let outs: Vec<_> = c.edges.iter().filter(|e| e.is_output).collect();
        assert!(outs.iter().all(|e| e.binds.is_some()), "{}", inst.recipe);
        let mut bound: Vec<_> = outs.iter().filter_map(|e| e.binds).collect();
        bound.sort();
        bound.dedup();
        let draft = c.blocks.iter().any(|b| b.section == Section::Draft);
        let want: &[ModelBuffer] = if draft {
            &[
                ModelBuffer::Logits,
                ModelBuffer::Tokens,
                ModelBuffer::DraftEmbed,
            ]
        } else {
            &[ModelBuffer::Logits, ModelBuffer::Tokens]
        };
        assert_eq!(bound, want, "{}", inst.recipe);
    }
    let e = mutated(
        "qwen3.8/qwen3.8-27b-nvfp4-unsloth",
        "{ edge = \"token\", buffer = \"tokens\" }",
        "{ edge = \"token\", buffer = \"token_ids\" }",
    );
    assert!(e.contains("is no model buffer"), "{e}");
}

/// 2026-09-30: A declared lifetime that is not its kind's is refused.
#[test]
fn a_state_declared_with_another_kinds_lifetime_is_refused() {
    let dense = "qwen3.8/qwen3.8-27b-nvfp4-unsloth";
    for (from, to, want) in [
        (
            "kind = \"recurrent\"\nlifetime = \"sequence\"",
            "kind = \"recurrent\"\nlifetime = \"model\"",
            "lives `sequence`, not `model`",
        ),
        (
            "kind = \"paged_kv\"\nlifetime = \"model\"",
            "kind = \"paged_kv\"\nlifetime = \"verify\"",
            "lives `model`, not `verify`",
        ),
        (
            "kind = \"paged_kv\"\nlifetime = \"model\"",
            "kind = \"paged_kv\"\nlifetime = \"forever\"",
            "is not model, sequence, verify or snapshot",
        ),
        (
            "kind = \"prefix_snapshot\"\nlifetime = \"snapshot\"",
            "kind = \"prefix_snapshot\"\nlifetime = \"sequence\"",
            "lives `snapshot`, not `sequence`",
        ),
    ] {
        let e = mutated(dense, from, to);
        assert!(e.contains(want), "`{to}`: {e}");
    }
}

/// 2026-10-02: A cache kind outside the vocabulary is refused by name with the known kinds; a
/// copy names a state of its own block; a cache no layer node may touch; and a non-copy kind
/// may not name a state to copy.
#[test]
fn an_unknown_or_misused_cache_kind_is_refused() {
    let dense = "qwen3.8/qwen3.8-27b-nvfp4-unsloth";
    for (from, to, want) in [
        (
            "id = \"prefix_h\"\nkind = \"prefix_snapshot\"",
            "id = \"prefix_h\"\nkind = \"acceptance_cache\"",
            "kind `acceptance_cache` is not one of recurrent, paged_kv, prefix_snapshot",
        ),
        (
            "kind = \"prefix_snapshot\"\nlifetime = \"snapshot\"\nformat = \"f32\"\nof = \"h\"",
            "kind = \"prefix_snapshot\"\nlifetime = \"snapshot\"\nformat = \"f32\"\nof = \"hh\"",
            "`of = \"hh\"` names no state of the block",
        ),
        (
            "kind = \"carry_stash\"\nlifetime = \"verify\"\nformat = \"u32\"\nshape = \"1\"",
            "kind = \"carry_stash\"\nlifetime = \"verify\"\nformat = \"u32\"\nof = \"h\"",
            "a carry_stash state copies no state",
        ),
        (
            "state = { conv = \"update\" }",
            "state = { conv = \"update\", prefix_conv = \"read\" }",
            "a prefix_snapshot cache is touched by no layer node",
        ),
    ] {
        let e = mutated(dense, from, to);
        assert!(e.contains(want), "`{to}`: {e}");
    }
}
