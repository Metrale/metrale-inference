// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The Gemma-4 circuit (`gemma4`; Gemma-4-31B dense and Gemma-4-26B-A4B MoE): its
//! config map reproduces both instances' shapes from the checkpoints' own `config.json`, the
//! layers take the sliding or global attention block and the dense or MoE FFN site, each kind
//! keeps its own head geometry and KV state, every site runs its declared formats, the declared
//! plans agree with the precision tables node by node, the estimate costs each attention at its
//! own geometry and window, and the config map refuses what the circuit does not model.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod checkpoint_fixtures;
mod common;

use std::collections::BTreeMap;

use checkpoint_fixtures::*;
use metrale_circuit::ir::Circuit;
use metrale_circuit::{Format, Instance, Mode, OpKind, Section};

const DENSE: &str = "gemma4/gemma-4-31b-nvfp4";
const DENSE_FIXTURE: &str = "nvidia--Gemma-4-31B-IT-NVFP4";
const MOE: &str = "gemma4/gemma-4-26b-a4b-nvfp4";
const MOE_FIXTURE: &str = "bg-digitalservices--Gemma-4-26B-A4B-it-NVFP4A16";

fn instance(recipe: &str) -> Instance {
    common::instances()
        .into_iter()
        .find(|i| i.recipe == recipe)
        .unwrap_or_else(|| panic!("no instance {recipe}"))
}

fn circuit(recipe: &str) -> Circuit {
    common::load(&instance(recipe)).circuit
}

fn edited(fixture_name: &str, edit: impl Fn(&mut serde_json::Value)) -> String {
    let mut v: serde_json::Value = serde_json::from_str(&fixture(fixture_name).0).unwrap();
    edit(&mut v);
    v.to_string()
}

fn per_layer(c: &Circuit) -> BTreeMap<usize, Vec<&str>> {
    let mut out: BTreeMap<usize, Vec<&str>> = BTreeMap::new();
    for b in &c.blocks {
        if let Some(l) = b.layer {
            out.entry(l).or_default().push(b.template.as_str());
        }
    }
    out
}

fn width(c: &Circuit, edge: &str) -> u64 {
    c.edges[c.edge(edge).unwrap_or_else(|| panic!("no edge {edge}"))].dim_value
}

#[test]
fn the_config_map_reproduces_both_instance_shapes_and_params() {
    for (recipe, name) in [(DENSE, DENSE_FIXTURE), (MOE, MOE_FIXTURE)] {
        let m = metrale_circuit::map_checkpoint(&fixture(name).0).unwrap();
        assert_eq!(m.arch, "gemma4", "{name}");
        assert_eq!(m.shape, instance(recipe).shape, "{name}");
        for (k, v) in [
            ("final_logit_softcapping", "30.0"),
            ("rms_norm_eps", "1e-6"),
            ("rope.sliding_attention.rope_type", "\"default\""),
            ("rope.sliding_attention.rope_theta", "10000.0"),
            ("rope.full_attention.rope_type", "\"proportional\""),
            ("rope.full_attention.rope_theta", "1000000.0"),
            ("rope.full_attention.partial_rotary_factor", "0.25"),
        ] {
            assert_eq!(m.params.get(k).map(String::as_str), Some(v), "{name} {k}");
        }
    }
}

#[test]
fn the_layers_interleave_five_sliding_then_one_global_and_the_moe_switch_picks_the_ffn() {
    for (recipe, layers, ffn) in [(DENSE, 60, "ffn"), (MOE, 30, "ffn_moe")] {
        let c = circuit(recipe);
        let blocks = per_layer(&c);
        assert_eq!(blocks.len(), layers, "{recipe}");
        for (l, b) in &blocks {
            let attn = if (l + 1) % 6 == 0 { "fa" } else { "swa" };
            assert_eq!(b, &[attn, ffn], "{recipe} layer {l}");
        }
        assert!(c.blocks.iter().all(|b| b.section == Section::Main));
        let prologue: Vec<&str> = c.nodes[..2].iter().map(|n| n.id.as_str()).collect();
        assert_eq!(prologue, ["embed.embed", "embed.scale"], "{recipe}");
    }
}

#[test]
fn each_attention_kind_keeps_its_own_geometry_window_and_kv() {
    let c = circuit(DENSE);
    // 2026-10-10: Sliding: 32 x 256 queries over 16 x 256 KV; global: 32 x 512 over 4 x 512.
    assert_eq!(width(&c, "l0.swa.q"), 32 * 256);
    assert_eq!(width(&c, "l0.swa.k"), 16 * 256);
    assert_eq!(width(&c, "l5.fa.q"), 32 * 512);
    assert_eq!(width(&c, "l5.fa.k"), 4 * 512);
    let elements = |id: &str| c.states.iter().find(|s| s.id == id).unwrap().elements;
    assert_eq!(elements("l0.swa.k"), 16 * 256);
    assert_eq!(elements("l0.swa.v"), 16 * 256);
    assert_eq!(elements("l5.fa.k"), 4 * 512);
    assert_eq!(elements("l5.fa.v"), 4 * 512);
    let params = |id: &str| node(&c, id).params.clone();
    assert_eq!(params("l0.swa.attend")["window"], "1024");
    assert_eq!(params("l0.swa.attend")["softmax_scale"], "1");
    assert!(!params("l5.fa.attend").contains_key("window"));
    assert_eq!(params("l5.fa.attend")["softmax_scale"], "1");
    assert_eq!(params("l5.fa.rope")["rope_type"], "proportional");
    assert_eq!(params("l0.swa.rope")["rope_type"], "default");
    // 2026-10-10: K = V on the global layers: no v_proj, and V is normed from the K
    // projection's own output (before k_norm and RoPE); the sliding layers project V.
    assert!(c.node("l5.fa.v").is_none());
    assert_eq!(
        node(&c, "l5.fa.v_norm").inputs,
        node(&c, "l5.fa.k_norm").inputs
    );
    assert_eq!(
        node(&c, "l0.swa.v_norm").inputs[0],
        node(&c, "l0.swa.v").outputs[0]
    );
    assert!(node(&c, "l0.swa.v_norm").binding.is_empty());
    let kv_write = node(&c, "l5.fa.kv_write");
    assert_eq!(kv_write.inputs[1], node(&c, "l5.fa.v_norm").outputs[0]);
}

#[test]
fn the_moe_site_routes_the_raw_residual_and_sums_two_normed_branches() {
    let c = circuit(MOE);
    let x = node(&c, "l0.ffn_moe.pre_norm").inputs[0];
    assert_eq!(node(&c, "l0.ffn_moe.router_norm").inputs, [x]);
    assert_eq!(node(&c, "l0.ffn_moe.pre_norm_2").inputs, [x]);
    assert_eq!(
        node(&c, "l0.ffn_moe.router_norm").binding,
        ["model.language_model.layers.0.router.scale"]
    );
    let top_k = node(&c, "l0.ffn_moe.top_k");
    assert_eq!(top_k.params["scoring"], "softmax");
    assert_eq!(top_k.params["top_k"], "8");
    assert_eq!(
        top_k.binding,
        ["model.language_model.layers.0.router.per_expert_scale"]
    );
    assert_eq!(width(&c, "l0.ffn_moe.egu"), 704 * 2);
    assert_eq!(width(&c, "l0.ffn_moe.gu"), 2112 * 2);
    assert_eq!(
        node(&c, "l0.ffn_moe.experts_act").op,
        OpKind::GeluTanhMul,
        "GeGLU experts"
    );
    let combine = node(&c, "l0.ffn_moe.combine");
    assert_eq!(
        combine.inputs,
        [
            node(&c, "l0.ffn_moe.post_norm_1").outputs[0],
            node(&c, "l0.ffn_moe.post_norm_2").outputs[0]
        ]
    );
    // 2026-10-10: The layer ends by scaling the whole stream, residual included.
    let scale = node(&c, "l0.ffn_moe.layer_scale");
    assert_eq!(scale.inputs, node(&c, "l0.ffn_moe.add").outputs);
    assert_eq!(
        scale.binding,
        ["model.language_model.layers.0.layer_scalar"]
    );
}

#[test]
fn the_head_softcaps_the_tied_logits_only_when_the_config_sets_a_cap() {
    let c = circuit(DENSE);
    let softcap = node(&c, "head.softcap");
    assert_eq!(softcap.op, OpKind::LogitSoftcap);
    assert_eq!(softcap.inputs, node(&c, "head.lm_head").outputs);
    assert_eq!(node(&c, "head.argmax").inputs, softcap.outputs);
    assert!(c.edges[softcap.outputs[0]].is_output);
    assert_eq!(
        node(&c, "head.lm_head").params["tied"],
        "model.language_model.embed_tokens"
    );
    let uncapped = edited(DENSE_FIXTURE, |v| {
        v["text_config"]["final_logit_softcapping"] = serde_json::Value::Null;
    });
    let r = metrale_circuit::resolve_checkpoint(
        &uncapped,
        metrale_circuit::QuantMetadata::default(),
        &metrale_circuit::ServePrecision::Declared,
    )
    .unwrap();
    assert_eq!(r.shape.dims["softcap"], 0);
    assert!(r.circuit.node("head.softcap").is_none());
    let lm = node(&r.circuit, "head.lm_head").outputs[0];
    assert!(r.circuit.edges[lm].is_output);
    assert_eq!(node(&r.circuit, "head.argmax").inputs, [lm]);
}

#[test]
fn each_site_runs_its_declared_formats() {
    let bf16 = Format::Bf16;
    for (recipe, cases) in [
        (
            DENSE,
            vec![
                ("l0.swa.q", bf16, bf16),
                ("l0.swa.v", bf16, bf16),
                ("l5.fa.k", bf16, bf16),
                ("l5.fa.o", bf16, bf16),
                ("l0.ffn.gate_up", NVFP4, NVFP4),
                ("l59.ffn.down", NVFP4, NVFP4),
                ("head.lm_head", bf16, bf16),
            ],
        ),
        (
            MOE,
            vec![
                ("l0.swa.q", NVFP4, NVFP4),
                ("l5.fa.o", NVFP4, NVFP4),
                ("l0.ffn_moe.gate_up", NVFP4, NVFP4),
                ("l0.ffn_moe.router", bf16, bf16),
                ("l0.ffn_moe.experts_gate_up", NVFP4, NVFP4),
                ("l29.ffn_moe.experts_down", NVFP4, NVFP4),
                ("head.lm_head", bf16, bf16),
            ],
        ),
    ] {
        let c = circuit(recipe);
        for (id, w, a) in cases {
            assert_eq!(node(&c, id).weight, Some(w), "{recipe} {id} weight");
            assert_eq!(input_format(&c, id), a, "{recipe} {id} input");
        }
        // 2026-10-10: The W4A4 inputs come from quantizer nodes the precision inserted.
        let q = c.edges[node(&c, "l0.swa.k").inputs[0]].producer.unwrap();
        let want = if recipe == MOE {
            "act_quant:nvfp4/g16"
        } else {
            "rms_norm"
        };
        assert_eq!(c.nodes[q].op.name(), want, "{recipe}");
    }
}

#[test]
fn the_checkpoints_declared_plans_give_every_node_the_tables_formats() {
    for (recipe, name, kv) in [
        (DENSE, DENSE_FIXTURE, Some(FP8_TENSOR)),
        (MOE, MOE_FIXTURE, None),
    ] {
        let r = ok(name);
        let table = circuit(recipe);
        assert_eq!(r.arch, "gemma4");
        assert_eq!(r.kv_cache, kv, "{name}");
        let ids = |c: &Circuit| c.nodes.iter().map(|n| n.id.clone()).collect::<Vec<_>>();
        assert_eq!(ids(&r.circuit), ids(&table), "{name}");
        for n in &table.nodes {
            assert_eq!(node(&r.circuit, &n.id).weight, n.weight, "{} weight", n.id);
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
}

#[test]
fn the_estimate_costs_each_attention_at_its_geometry_and_window() {
    let inst = instance(DENSE);
    let c = circuit(DENSE);
    let fams = metrale_circuit::venn::parse_families(&common::read(
        "kernels/gb10/common/KERNEL_FAMILIES.toml",
    ))
    .unwrap();
    let r = fams.roofline;
    let settings = &inst.policy.settings;
    let ctx = r.context_tokens as f64;
    let cost = |id: &str| {
        metrale_circuit::venn::roofline::node_cost(&c, node(&c, id), Mode::Decode, 1, settings, &r)
            .unwrap()
    };
    let edges = |id: &str| -> f64 {
        let n = node(&c, id);
        n.inputs
            .iter()
            .chain(&n.outputs)
            .map(|&e| c.edges[e].dim_value as f64 * 2.0)
            .sum()
    };
    // 2026-10-10: FP8 KV (one byte), one token of K and of V per attended token.
    let swa = cost("l0.swa.attend");
    let attended = ctx.min(1024.0);
    assert_eq!(
        swa.bytes - edges("l0.swa.attend"),
        attended * 2.0 * 16.0 * 256.0
    );
    assert_eq!(swa.flops, 4.0 * attended * 32.0 * 256.0);
    let fa = cost("l5.fa.attend");
    assert_eq!(fa.bytes - edges("l5.fa.attend"), ctx * 2.0 * 4.0 * 512.0);
    assert_eq!(fa.flops, 4.0 * ctx * 32.0 * 512.0);
    assert!(
        ctx > 1024.0,
        "the window must bind at the roofline's context"
    );
    // 2026-10-10: GeGLU states its FLOPs per output element; the footprint sums every layer's
    // own KV geometry.
    assert_eq!(cost("l0.ffn.act").flops, 10.0 * 21504.0);
    let f = metrale_circuit::hardware::estimate::footprint(&c, settings).unwrap();
    assert_eq!(
        f.kv_per_token,
        50.0 * 2.0 * 16.0 * 256.0 + 10.0 * 2.0 * 4.0 * 512.0
    );
}

#[test]
fn the_config_map_refuses_what_the_circuit_does_not_model() {
    let refused = |name: &str, edit: &dyn Fn(&mut serde_json::Value), want: &str| {
        let e = metrale_circuit::map_checkpoint(&edited(name, edit))
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(e.contains(want), "{want}: {e}");
    };
    let t = |k: &'static str, v: serde_json::Value| {
        move |j: &mut serde_json::Value| j["text_config"][k] = v.clone()
    };
    refused(
        DENSE_FIXTURE,
        &t("hidden_size_per_layer_input", 256.into()),
        "per-layer embeddings",
    );
    refused(
        DENSE_FIXTURE,
        &t("num_kv_shared_layers", 20.into()),
        "KV-shared layers",
    );
    refused(
        MOE_FIXTURE,
        &t("attention_k_eq_v", false.into()),
        "share K and V",
    );
    refused(
        DENSE_FIXTURE,
        &t("use_bidirectional_attention", "all".into()),
        "the circuit is causal",
    );
    refused(
        DENSE_FIXTURE,
        &t("hidden_activation", "silu".into()),
        "tanh-GELU",
    );
    refused(
        MOE_FIXTURE,
        &|j| j["text_config"]["rope_parameters"]["full_attention"]["rope_type"] = "yarn".into(),
        "proportional RoPE",
    );
    refused(
        DENSE_FIXTURE,
        &|j| {
            j["text_config"]["rope_parameters"]["sliding_attention"]["partial_rotary_factor"] =
                0.5.into()
        },
        "rotate the whole head",
    );
    refused(
        DENSE_FIXTURE,
        &t("expert_intermediate_size", 704.into()),
        "expert_intermediate_size",
    );
    refused(
        DENSE_FIXTURE,
        &t("attn_logit_softcapping", 50.0.into()),
        "`text_config.attn_logit_softcapping` is not mapped",
    );
    refused(
        MOE_FIXTURE,
        &|j| j["tie_word_embeddings"] = false.into(),
        "tied to the embedding",
    );
    refused(
        DENSE_FIXTURE,
        &|j| j["text_config"]["layer_types"][0] = "chunked_attention".into(),
        "layer_types",
    );
}
