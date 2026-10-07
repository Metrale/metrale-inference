// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Architecture fidelity and refusal controls for the pinned GPT-OSS checkpoint.
use metrale_circuit::config_map::{ConfigMap, map_config};
use metrale_circuit::fuser::Policy;
use metrale_circuit::{AvailableKernels, Format, LayerKind, Mode, fuse};

const CONFIG: &str = include_str!("fixtures/checkpoints/openai--gpt-oss-20b/config.json");
const MAP: &str = include_str!("../../../kernels/circuits/gpt_oss.config.toml");

fn build() -> metrale_circuit::Circuit {
    let config = serde_json::from_str(CONFIG).unwrap();
    metrale_circuit::gpt_oss::architecture(&config).unwrap()
}

#[test]
fn reference_layer_mix_bindings_and_storage_survive_instantiation() {
    let c = build();
    assert_eq!(c.layer_kinds.len(), 24);
    assert_eq!(c.states.len(), 48);
    for layer in 0..24 {
        assert_eq!(
            c.layer_kinds[layer],
            if layer % 2 == 0 {
                LayerKind::SlidingAttention
            } else {
                LayerKind::FullAttention
            }
        );
        let attention = c
            .nodes
            .iter()
            .find(|n| n.layer == Some(layer) && n.local == "attend")
            .unwrap();
        assert_eq!(
            attention.params["window"],
            if layer % 2 == 0 { "128" } else { "none" }
        );
        assert_eq!(
            attention.binding,
            [format!("model.layers.{layer}.self_attn.sinks")]
        );
        let gate = c
            .nodes
            .iter()
            .find(|n| n.layer == Some(layer) && n.local == "gate_up")
            .unwrap();
        assert_eq!(gate.weight, Some(Format::Mxfp4));
        assert_eq!(
            gate.binding,
            [format!("model.layers.{layer}.mlp.experts.gate_up_proj")]
        );
        assert_eq!(gate.params["layout"], "interleaved_even_gate_odd_up");
        let act = c
            .nodes
            .iter()
            .find(|n| n.layer == Some(layer) && n.local == "act")
            .unwrap();
        assert_eq!(act.params["alpha"], "1.702");
        assert_eq!(act.params["gate_clamp_max"], "7");
        assert_eq!(act.params["up_offset"], "1");
        let down = c
            .nodes
            .iter()
            .find(|n| n.layer == Some(layer) && n.local == "down")
            .unwrap();
        assert_eq!(down.params["bias_order"], "before_routing_weight");
    }
}

#[test]
fn changed_missing_and_unclassified_math_never_silently_use_pinned_policy() {
    let map = ConfigMap::parse(MAP).unwrap();
    let original: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    for key in [
        "swiglu_limit",
        "attention_bias",
        "rope_scaling",
        "quantization_config",
        "tie_word_embeddings",
    ] {
        let mut config = original.clone();
        config.as_object_mut().unwrap().remove(key);
        assert!(map_config(&map, &config).is_err(), "missing {key}");
    }
    for (pointer, value) in [
        ("/swiglu_limit", serde_json::json!(6.0)),
        ("/rope_scaling/truncate", serde_json::json!(true)),
        ("/rope_scaling/factor", serde_json::json!(16.0)),
        (
            "/quantization_config/quant_method",
            serde_json::json!("nvfp4"),
        ),
        ("/experts_per_token", serde_json::json!(8)),
    ] {
        let mut config = original.clone();
        *config.pointer_mut(pointer).unwrap() = value;
        assert!(map_config(&map, &config).is_err(), "changed {pointer}");
    }
    let mut config = original;
    config["qk_norm"] = serde_json::json!(true);
    assert!(map_config(&map, &config).is_err());
}

#[test]
fn unlowered_semantics_cannot_match_even_an_explicit_op_rule() {
    let mut c = build();
    // 2026-10-07: Isolate one residual so unrelated uncovered ops cannot satisfy this oracle.
    let original = c.nodes.iter().find(|n| n.local == "act").unwrap().clone();
    c.nodes = vec![original];
    c.nodes[0].inputs.clear();
    c.nodes[0].outputs.clear();
    c.blocks.truncate(1);
    c.blocks[0].first = 0;
    c.blocks[0].end = 1;
    c.edges.clear();
    let rule = r#"
schema = 1
[[rule]]
id = "plain-silu"
modes = ["decode"]
pattern = [{ op = "silu_mul" }]
kernels = [{ module = "silu", func = "plain_silu" }]
repeat = "once"
rows = [1, 128]
priority = 1
cite = "test-only-control"
emitter = "plain_silu"
numerics = "reference"
"#;
    let rules = metrale_circuit::parse_rules(rule).unwrap();
    let available = AvailableKernels::all_named_by(&rules);
    assert!(fuse(&c, &rules, &available, &Policy::default(), Mode::Decode, 1).is_err());
    // 2026-10-07: Valid control: identical graph and rule without a residual must fuse.
    c.nodes[0].params.remove("unlowered");
    assert!(fuse(&c, &rules, &available, &Policy::default(), Mode::Decode, 1).is_ok());
}
