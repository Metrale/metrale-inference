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
                LayerKind::FullAttention => vec![
                    (OpKind::KvWrite, "k", StateAccess::Write),
                    (OpKind::KvWrite, "v", StateAccess::Write),
                    (OpKind::PagedAttention, "k", StateAccess::Read),
                    (OpKind::PagedAttention, "v", StateAccess::Read),
                ],
                LayerKind::Moe => vec![],
            };
            let mut want: Vec<_> = want
                .into_iter()
                .map(|(o, l, a)| (o, l.to_string(), a))
                .collect();
            want.sort();
            assert_eq!(r, want, "{} layer {layer} ({kind:?})", inst.recipe);
        }
        // 2026-09-30: The draft head's attention keeps its own KV cache.
        let draft: Vec<_> = c
            .states
            .iter()
            .filter(|s| s.section == Section::Draft)
            .collect();
        if !draft.is_empty() {
            assert!(
                draft.iter().all(|s| s.kind == StateKind::PagedKv),
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
