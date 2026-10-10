// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The GQA + routed-MoE circuit (`gqa_moe`) on MiniMax-M2.7: its config map
//! reproduces the instance's shape from the checkpoint's own `config.json`, every layer is
//! attention then MoE with no draft head, the Q/K norms span the layer, the routing binds the
//! checkpoint's tensors, every site runs its declared formats, the checkpoint's declared plan
//! agrees with the precision table node by node, and the config map refuses what the circuit
//! does not model.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod checkpoint_fixtures;
mod common;

use checkpoint_fixtures::*;
use metrale_circuit::ir::Circuit;
use metrale_circuit::{Format, Instance, Section};

const MINIMAX: &str = "minimax-m2.7/minimax-m2.7-nvfp4-ep2";
const FIXTURE: &str = "lukealonso--MiniMax-M2.7-NVFP4";

fn instance() -> Instance {
    common::instances()
        .into_iter()
        .find(|i| i.recipe == MINIMAX)
        .expect("the MiniMax instance")
}

fn circuit() -> Circuit {
    common::load(&instance()).circuit
}

fn weight(c: &Circuit, id: &str) -> Option<Format> {
    node(c, id).weight
}

fn edited_config(edit: impl Fn(&mut serde_json::Value)) -> String {
    let mut v: serde_json::Value = serde_json::from_str(&fixture(FIXTURE).0).unwrap();
    edit(&mut v);
    v.to_string()
}

#[test]
fn the_config_map_reproduces_the_instance_shape() {
    let m = metrale_circuit::map_checkpoint(&fixture(FIXTURE).0).unwrap();
    assert_eq!(m.arch, "gqa_moe");
    assert_eq!(m.model_type, "minimax_m2");
    assert_eq!(m.shape, instance().shape);
    let p = |k: &str| m.params.get(k).map(String::as_str);
    assert_eq!(p("qk_norm_type"), Some("\"per_layer\""));
    assert_eq!(p("rope.rope_type"), Some("\"default\""));
    assert_eq!(p("rope.rope_theta"), Some("5000000"));
    assert_eq!(p("num_mtp_modules"), Some("3"));
    // 2026-10-10: The two partial-RoPE spellings agree: rotary_dim = head_dim * factor.
    let factor: f64 = p("rope.partial_rotary_factor").unwrap().parse().unwrap();
    let rotary: f64 = p("rotary_dim").unwrap().parse().unwrap();
    assert_eq!(p("partial_rotary_factor"), p("rope.partial_rotary_factor"));
    assert_eq!(rotary, m.shape.dims["head_dim"] as f64 * factor);
    assert_eq!(rotary, 64.0);
}

#[test]
fn every_layer_is_attention_then_moe_and_there_is_no_draft_head() {
    let c = circuit();
    assert_eq!(c.layer_kinds.len(), 62);
    for l in 0..62 {
        let blocks: Vec<&str> = c
            .blocks
            .iter()
            .filter(|b| b.layer == Some(l))
            .map(|b| b.template.as_str())
            .collect();
        assert_eq!(blocks, ["attn", "moe"], "layer {l}");
    }
    assert!(c.blocks.iter().all(|b| b.section != Section::Draft));
    // 2026-10-10: Layer 0's MoE output is layer 1's attention input, and the head reads layer
    // 61's.
    let l0 = c.edge("l0.moe.y").unwrap();
    let attn1 = c
        .blocks
        .iter()
        .find(|b| b.layer == Some(1) && b.template == "attn")
        .unwrap();
    assert_eq!(attn1.stream_in, Some(l0));
    let head = c.blocks.iter().find(|b| b.template == "head").unwrap();
    assert_eq!(head.stream_in, c.edge("l61.moe.y"));
}

#[test]
fn the_qk_norms_span_the_layer_and_the_routing_binds_the_checkpoints_tensors() {
    let c = circuit();
    for (id, width, module) in [
        (
            "l7.attn.q_norm",
            48 * 128,
            "model.layers.7.self_attn.q_norm",
        ),
        ("l7.attn.k_norm", 8 * 128, "model.layers.7.self_attn.k_norm"),
    ] {
        let n = node(&c, id);
        assert_eq!(n.params.get("scope").map(String::as_str), Some("layer"));
        assert_eq!(
            n.params.get("weight_form").map(String::as_str),
            Some("plain")
        );
        assert_eq!(c.edges[n.outputs[0]].dim_value, width, "{id}");
        assert_eq!(n.binding, [module]);
    }
    let top_k = node(&c, "l7.moe.top_k");
    assert_eq!(
        top_k.binding,
        ["model.layers.7.block_sparse_moe.e_score_correction_bias"]
    );
    for (k, v) in [
        ("scoring", "sigmoid_bias"),
        ("renormalize", "true"),
        ("top_k", "8"),
    ] {
        assert_eq!(top_k.params.get(k).map(String::as_str), Some(v), "{k}");
    }
    assert_eq!(
        node(&c, "l7.moe.experts_gate_up").binding,
        [
            "model.layers.7.block_sparse_moe.experts.*.w1",
            "model.layers.7.block_sparse_moe.experts.*.w3"
        ]
    );
    assert_eq!(
        node(&c, "l7.moe.experts_down").binding,
        ["model.layers.7.block_sparse_moe.experts.*.w2"]
    );
    let blend = node(&c, "l7.moe.blend");
    assert_eq!(blend.inputs.len(), 2, "no shared expert");
    assert_eq!(
        c.weight_shape(node(&c, "l7.moe.experts_gate_up")),
        Some((1536 * 2, 3072))
    );
    assert_eq!(
        c.weight_shape(node(&c, "l7.attn.q")),
        Some((48 * 128, 3072))
    );
}

#[test]
fn each_site_runs_its_declared_formats() {
    let c = circuit();
    let bf = Format::Bf16;
    for (id, w, a) in [
        ("l0.moe.experts_gate_up", NVFP4, NVFP4),
        ("l61.moe.experts_down", NVFP4, NVFP4),
        ("l0.moe.gate", bf, bf),
        ("l0.attn.q", bf, bf),
        ("l0.attn.k", bf, bf),
        ("l30.attn.v", bf, bf),
        ("l61.attn.o", bf, bf),
        ("head.lm_head", bf, bf),
    ] {
        assert_eq!(weight(&c, id), Some(w), "{id} weight");
        assert_eq!(input_format(&c, id), a, "{id} input");
    }
    let logits = node(&c, "l0.moe.gate").outputs[0];
    // 2026-10-10: The reference's router GEMM runs at the gate weight's BF16; only the sigmoid
    // widens to FP32 (inside top_k).
    assert_eq!(c.edges[logits].format, Format::Bf16, "BF16 router logits");
    let topk_w = node(&c, "l0.moe.top_k").outputs[0];
    assert_eq!(c.edges[topk_w].format, Format::F32, "FP32 routing weights");
    // 2026-10-10: The W4A4 inputs come from quantizer nodes the precision inserted, one per
    // projection's static input scale.
    for id in ["l0.moe.experts_gate_up", "l0.moe.experts_down"] {
        let q = c.edges[node(&c, id).inputs[0]].producer.unwrap();
        assert_eq!(c.nodes[q].op.name(), "act_quant:nvfp4/g16", "{id}");
    }
}

#[test]
fn the_checkpoints_declared_plan_gives_every_node_the_tables_formats() {
    let r = ok(FIXTURE);
    let table = circuit();
    assert_eq!(r.arch, "gqa_moe");
    assert_eq!(r.kv_cache, None, "no declared KV-cache scheme: 16-bit");
    let ids = |c: &Circuit| c.nodes.iter().map(|n| n.id.clone()).collect::<Vec<_>>();
    assert_eq!(ids(&r.circuit), ids(&table));
    for n in &table.nodes {
        assert_eq!(weight(&r.circuit, &n.id), n.weight, "{} weight", n.id);
        if !n.inputs.is_empty() {
            assert_eq!(
                input_format(&r.circuit, &n.id),
                input_format(&table, &n.id),
                "{} input",
                n.id
            );
        }
    }
}

#[test]
fn the_config_map_refuses_what_the_circuit_does_not_model() {
    let refused = |edit: &dyn Fn(&mut serde_json::Value), want: &str| {
        let e = metrale_circuit::map_checkpoint(&edited_config(edit))
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(e.contains(want), "{want}: {e}");
    };
    // 2026-10-10: A lightning-attention layer (MiniMax-Text-01's `attn_type_list` 0).
    refused(&|v| v["attn_type_list"][5] = 0.into(), "attn_type_list");
    refused(
        &|v| v["qk_norm_type"] = "per_head".into(),
        "per-head norm is not modelled",
    );
    refused(&|v| v["use_qk_norm"] = false.into(), "use_qk_norm");
    refused(&|v| v["scoring_func"] = "softmax".into(), "scoring_func");
    refused(
        &|v| v["use_routing_bias"] = false.into(),
        "use_routing_bias",
    );
    refused(
        &|v| v["shared_intermediate_size"] = 1536.into(),
        "no shared expert",
    );
    refused(
        &|v| v["rope_parameters"]["rope_type"] = "yarn".into(),
        "rope_parameters.rope_type",
    );
    refused(&|v| v["sliding_window"] = 4096.into(), "sliding-window");
    refused(&|v| v["tie_word_embeddings"] = true.into(), "tied");
    refused(&|v| v["hidden_act"] = "gelu".into(), "SiLU");
    refused(
        &|v| v["logit_softcapping"] = 30.0.into(),
        "logit_softcapping",
    );
    // 2026-10-10: A layer list that disagrees with the layer count.
    refused(
        &|v| {
            v["attn_type_list"].as_array_mut().unwrap().pop();
        },
        "num_hidden_layers",
    );
}
