// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Tests for the pool's overlay spec from adapter targets.
//!
//! Owner: model-layers (FEATURES workstream).
//! Invariants: none beyond the module's.

use std::collections::BTreeSet;

use metrale_circuit::LinearRole;
use metrale_config::ModelConfig;

use super::*;

/// 2026-10-03: The factory config (full attention at 3, 7, ..., 47) as a dense model, so the
/// dense-FFN targets classify.
fn dense() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.num_experts = 0;
    c
}

fn keys(names: &[(usize, &str)]) -> Vec<String> {
    names
        .iter()
        .flat_map(|(l, m)| {
            ["A", "B"].map(|ab| format!("base_model.model.model.layers.{l}.{m}.lora_{ab}.weight"))
        })
        .collect()
}

fn targets(names: &[(usize, &str)]) -> AdapterTargets {
    targets_from_keys(keys(names).iter().map(String::as_str), &dense()).unwrap()
}

fn roles(r: &[LinearRole]) -> BTreeSet<LinearRole> {
    r.iter().copied().collect()
}

// 2026-10-03: Attention is the union over the pool; FFN and out_proj follow the active adapter.
// Mutations: taking attention from the active adapter only drops layer 3's V and layer 7;
// taking the FFN from every adapter adds layer 1's Down (active 0) or layer 0 (active 1).
#[test]
fn attention_is_the_pool_union_and_the_ffn_the_active_adapters() {
    let a = targets(&[
        (3, "self_attn.q_proj"),
        (0, "mlp.gate_proj"),
        (0, "mlp.up_proj"),
        (0, "mlp.down_proj"),
        (2, "linear_attn.out_proj"),
    ]);
    let b = targets(&[
        (3, "self_attn.v_proj"),
        (7, "self_attn.k_proj"),
        (1, "mlp.down_proj"),
    ]);
    let pool = [a, b];
    let s = spec_from_targets(16, &pool, 0).unwrap();
    assert_eq!(s.rank, 16);
    assert_eq!(
        s.layers.keys().copied().collect::<Vec<_>>(),
        vec![0, 2, 3, 7]
    );
    assert_eq!(s.layers[&0], roles(&[LinearRole::GateUp, LinearRole::Down]));
    assert_eq!(s.layers[&2], roles(&[LinearRole::GdnOut]));
    assert_eq!(s.layers[&3], roles(&[LinearRole::Q, LinearRole::V]));
    assert_eq!(s.layers[&7], roles(&[LinearRole::K]));
    let s = spec_from_targets(16, &pool, 1).unwrap();
    assert_eq!(s.layers.keys().copied().collect::<Vec<_>>(), vec![1, 3, 7]);
    assert_eq!(s.layers[&1], roles(&[LinearRole::Down]));
}

// 2026-10-03: The gate|up fold takes both pairs (as the binding does); one alone is refused,
// but only on the active adapter, whose FFN pairs are folded.
#[test]
fn one_of_gate_and_up_is_refused_on_the_active_adapter() {
    let half = targets(&[(0, "mlp.up_proj")]);
    let whole = targets(&[(4, "mlp.gate_proj"), (4, "mlp.up_proj")]);
    let pool = [whole, half];
    let e = spec_from_targets(8, &pool, 1).unwrap_err();
    assert!(format!("{e:#}").contains("layer 0"), "{e:#}");
    let s = spec_from_targets(8, &pool, 0).unwrap();
    assert_eq!(s.layers[&4], roles(&[LinearRole::GateUp]));
    assert!(!s.layers.contains_key(&0));
}

// 2026-10-03: MoE targets, a key `classify_key` refuses, and an active index outside the pool
// are refused, never dropped.
#[test]
fn moe_targets_unknown_keys_and_a_missing_active_are_refused() {
    let moe = ModelConfig::qwen3_next_80b_nvfp4();
    let router = keys(&[(3, "mlp.gate")]);
    let e = targets_from_keys(router.iter().map(String::as_str), &moe).unwrap_err();
    assert!(format!("{e:#}").contains("MoE LoRA"), "{e:#}");
    let bad = ["base_model.model.model.layers.0.linear_attn.in_proj_qkvz.lora_A.weight"];
    assert!(targets_from_keys(bad, &dense()).is_err());
    assert!(spec_from_targets(8, &[targets(&[(3, "self_attn.q_proj")])], 1).is_err());
}
