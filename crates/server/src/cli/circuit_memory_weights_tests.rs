// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: Which checkpoint tensors a weight node binds, and the MODEL.toml defaults.
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn a_tensor_is_bound_by_its_binding_an_ancestor_binding_or_a_held_one() {
    let experts = "model.layers.3.mlp.experts.*.gate_proj";
    assert!(bound(experts, "model.layers.3.mlp.experts.7.gate_proj"));
    // 2026-10-02: A fused experts tensor holds every expert's projection.
    assert!(bound(experts, "model.layers.3.mlp.experts"));
    assert!(bound("lm_head", "lm_head"));
    assert!(!bound(
        experts,
        "model.layers.3.mlp.shared_expert.gate_proj"
    ));
    assert!(!bound(experts, "model.layers.4.mlp.experts.7.gate_proj"));
    assert!(!bound(
        "model.layers.3.self_attn.q_proj",
        "model.layers.3.input_layernorm"
    ));
    assert!(!bound(
        "model.layers.3.self_attn.q_proj",
        "model.visual.blocks.0.attn.qkv"
    ));
}

#[test]
fn model_toml_behavior_reads_its_three_defaults_and_absent_keys_are_zero() {
    let b = Behavior::parse(
        "[model]\nname = \"x\"\n[behavior]\ndefault_num_drafts = 3\nmtp_max_seqs = 128\n\
         default_kv_dtype = \"fp8\"\n",
        "t".into(),
    )
    .unwrap();
    assert_eq!(
        (
            b.default_num_drafts,
            b.mtp_max_seqs,
            b.default_kv_dtype.as_str()
        ),
        (3, 128, "fp8")
    );
    let none = Behavior::parse("[model]\nname = \"x\"\n", "t".into()).unwrap();
    assert_eq!((none.default_num_drafts, none.mtp_max_seqs), (0, 0));
    assert!(none.default_kv_dtype.is_empty());
}
