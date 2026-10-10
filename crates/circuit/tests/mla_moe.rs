// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The MLA + MoE circuit (`mla_moe`, Mistral-Small-4-119B): its config map reproduces
//! the instance's shape from the base repo's HF `config.json`, the NVFP4 checkpoint's own
//! `params.json` states the same architecture (so the two cannot drift), every layer is latent
//! attention then the MoE, every site runs its declared formats, the declared plan of
//! `params.json`'s quantization agrees with the precision table node by node except at the head
//! (whose ignore entry is written under its HF name), latent attention reads `wkv_b` and its
//! estimate counts the rotary key, and the config map refuses what the circuit does not model.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod checkpoint_fixtures;
mod common;

use std::collections::BTreeSet;

use checkpoint_fixtures::*;
use metrale_circuit::ir::Circuit;
use metrale_circuit::{
    Format, Instance, LayerKind, Mode, OpKind, QuantMetadata, Section, ServePrecision,
};
use serde_json::Value;

const RECIPE: &str = "mistral-small-4/mistral-small-4-119b-nvfp4";
const BASE: &str = "mistralai--Mistral-Small-4-119B-2603";
const PARAMS: &str = "crates/circuit/tests/fixtures/checkpoints/mistralai--Mistral-Small-4-119B-2603-NVFP4/params.json";

fn instance() -> Instance {
    common::instances()
        .into_iter()
        .find(|i| i.recipe == RECIPE)
        .expect("the Mistral-Small-4 instance")
}

fn circuit() -> Circuit {
    common::load(&instance()).circuit
}

fn params_json() -> Value {
    serde_json::from_str(&common::read(PARAMS)).unwrap()
}

fn base() -> Value {
    serde_json::from_str(&fixture(BASE).0).unwrap()
}

/// 2026-10-10: A mapped param (JSON text) as a number.
fn num(m: &metrale_circuit::config_map::MappedConfig, k: &str) -> f64 {
    let text = m.params.get(k).unwrap_or_else(|| panic!("no param {k}"));
    serde_json::from_str::<Value>(text)
        .unwrap()
        .as_f64()
        .unwrap()
}

fn weight(c: &Circuit, id: &str) -> Option<Format> {
    node(c, id).weight
}

#[test]
fn the_config_map_reproduces_the_instance_shape() {
    let m = metrale_circuit::map_checkpoint(&fixture(BASE).0).unwrap();
    assert_eq!(
        (m.arch.as_str(), m.model_type.as_str()),
        ("mla_moe", "mistral3")
    );
    assert_eq!(m.shape, instance().shape);
    for (k, v) in [
        ("routed_scaling_factor", "1.0"),
        ("norm_topk_prob", "true"),
        ("rope_interleave", "true"),
        ("rope.rope_type", "\"yarn\""),
        ("rope.factor", "128.0"),
        ("rope.mscale_all_dim", "1.0"),
        ("rope.llama_4_scaling_beta", "0.1"),
        ("rope.original_max_position_embeddings", "8192"),
    ] {
        assert_eq!(m.params.get(k).map(String::as_str), Some(v), "{k}");
    }
}

/// 2026-10-10: The NVFP4 checkpoint ships no config.json; the circuit maps the base repo's. Every
/// mapped dim and math param must equal the Mistral-native value, and every params.json key is
/// accounted for here, so a key params.json adds fails this test until it is classified.
#[test]
fn the_nvfp4_params_json_states_the_same_architecture() {
    let p = params_json();
    let m = metrale_circuit::map_checkpoint(&fixture(BASE).0).unwrap();
    let u = |v: &Value| {
        v.as_u64()
            .unwrap_or_else(|| panic!("{v} is not an integer"))
    };
    let moe = &p["moe"];
    for (dim, native) in [
        ("hidden", u(&p["dim"])),
        ("vocab", u(&p["vocab_size"])),
        ("q_heads", u(&p["n_heads"])),
        ("q_lora", u(&p["q_lora_rank"])),
        ("kv_lora", u(&p["kv_lora_rank"])),
        ("mla_qk", u(&p["qk_nope_head_dim"])),
        ("mla_rope", u(&p["qk_rope_head_dim"])),
        ("mla_v", u(&p["v_head_dim"])),
        ("experts", u(&moe["num_experts"])),
        ("top_k", u(&moe["num_experts_per_tok"])),
        ("moe_inter", u(&moe["expert_hidden_dim"])),
        (
            "shared_inter",
            u(&moe["expert_hidden_dim"]) * u(&moe["num_shared_experts"]),
        ),
    ] {
        assert_eq!(m.shape.dims[dim], native, "{dim}");
    }
    assert_eq!(m.shape.layer_kinds.len() as u64, u(&p["n_layers"]));
    assert!(
        m.shape
            .layer_kinds
            .iter()
            .all(|k| *k == LayerKind::FullAttention)
    );
    // 2026-10-10: What the config map requires, params.json states too.
    assert_eq!(
        u(&p["head_dim"]),
        m.shape.dims["mla_qk"] + m.shape.dims["mla_rope"]
    );
    assert_eq!(p["n_kv_heads"], p["n_heads"]);
    assert_eq!(p["tied_embeddings"], false);
    for (k, want) in [
        ("first_k_dense_replace", 0),
        ("num_expert_groups", 1),
        ("num_expert_groups_per_tok", 1),
        ("num_shared_experts", 1),
        ("route_every_n", 1),
        ("expert_parallel", 1),
        ("expert_model_parallel", 1),
    ] {
        assert_eq!(u(&moe[k]), want, "moe.{k}");
    }
    let f = |v: &Value| v.as_f64().unwrap();
    for (param, native) in [
        ("rms_norm_eps", f(&p["norm_eps"])),
        ("routed_scaling_factor", f(&moe["routed_scale"])),
        ("rope.rope_theta", f(&p["rope_theta"])),
        ("rope.factor", f(&p["yarn"]["factor"])),
        ("rope.beta_fast", f(&p["yarn"]["beta"])),
        ("rope.beta_slow", f(&p["yarn"]["alpha"])),
        (
            "rope.original_max_position_embeddings",
            f(&p["yarn"]["original_max_position_embeddings"]),
        ),
        (
            "rope.llama_4_scaling_beta",
            f(&p["llama_4_scaling"]["beta"]),
        ),
        (
            "rope.original_max_position_embeddings",
            f(&p["llama_4_scaling"]["original_max_position_embeddings"]),
        ),
    ] {
        assert_eq!(num(&m, param), native, "{param}");
    }
    // 2026-10-10: `yarn.apply_scale = false` (no YaRN magnitude on cos/sin) is the HF config's
    // mscale == mscale_all_dim (their ratio, the cos/sin factor, is 1).
    assert_eq!(p["yarn"]["apply_scale"], false);
    assert_eq!(num(&m, "rope.mscale"), num(&m, "rope.mscale_all_dim"));
    // 2026-10-10: Ignored by the circuit with a reason: the dense MLP width (no dense layers),
    // the context limit, the vision tower, and the quantization (the precision table's source).
    let base = base();
    assert_eq!(p["hidden_dim"], base["text_config"]["intermediate_size"]);
    assert_eq!(
        p["max_position_embeddings"],
        base["text_config"]["max_position_embeddings"]
    );
    let keys: BTreeSet<&str> = p.as_object().unwrap().keys().map(String::as_str).collect();
    let accounted: BTreeSet<&str> = [
        "dim",
        "head_dim",
        "hidden_dim",
        "kv_lora_rank",
        "llama_4_scaling",
        "max_position_embeddings",
        "moe",
        "n_heads",
        "n_kv_heads",
        "n_layers",
        "norm_eps",
        "q_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "quantization_config",
        "rope_theta",
        "tied_embeddings",
        "v_head_dim",
        "vision_encoder",
        "vocab_size",
        "yarn",
    ]
    .into();
    assert_eq!(keys, accounted);
    // 2026-10-10: No config map reads params.json itself (a config-side residual): it is refused
    // by its missing model_type, never mapped by a guess.
    let e = metrale_circuit::map_checkpoint(&common::read(PARAMS))
        .unwrap_err()
        .to_string();
    assert!(e.contains("no circuit serves model_type ``"), "{e}");
}

#[test]
fn every_layer_is_latent_attention_then_the_moe() {
    let c = circuit();
    let mut layers = 0;
    for l in 0..36 {
        let blocks: Vec<&str> = c
            .blocks
            .iter()
            .filter(|b| b.layer == Some(l))
            .map(|b| b.template.as_str())
            .collect();
        assert_eq!(blocks, ["mla", "moe"], "layer {l}");
        layers += 1;
    }
    assert_eq!(layers, c.layer_kinds.len());
    assert!(!c.blocks.iter().any(|b| b.section == Section::Draft));
    // 2026-10-10: One cache per layer: the 256-wide latent plus the 64-wide rotary key per token.
    let latent: Vec<_> = c.states.iter().filter(|s| s.local == "latent").collect();
    assert_eq!(latent.len(), 36);
    assert!(latent.iter().all(|s| s.elements == 256 + 64));
    // 2026-10-10: [nope 64 | rope 64] per head for 32 heads; RoPE keeps the query's width and
    // rotates the 64-wide shared key, which the cache write reads beside the normed latent.
    let width = |id: &str| c.edges[c.edge(id).unwrap()].dim_value;
    assert_eq!(width("l0.mla.q"), 32 * 128);
    assert_eq!(width("l0.mla.kva"), 256 + 64);
    assert_eq!(width("l0.mla.qr"), 32 * 128);
    assert_eq!(width("l0.mla.kr"), 64);
    assert_eq!(width("l0.mla.a"), 32 * 128);
    let write = node(&c, "l0.mla.kv_write");
    assert_eq!(
        write.inputs,
        [c.edge("l0.mla.c").unwrap(), c.edge("l0.mla.kr").unwrap()]
    );
    let attend = node(&c, "l0.mla.attend");
    assert_eq!(attend.inputs, [c.edge("l0.mla.qr").unwrap()]);
    for (k, v) in [
        ("selection", "all"),
        ("rope", "decoupled"),
        ("softmax_scale", "yarn_mscale_all_dim"),
        ("query_scale", "llama_4"),
    ] {
        assert_eq!(attend.params.get(k).map(String::as_str), Some(v), "{k}");
    }
    assert_eq!(
        node(&c, "l0.mla.rope")
            .params
            .get("rotary_dim")
            .map(String::as_str),
        Some("64")
    );
    let topk = node(&c, "l0.moe.top_k");
    assert_eq!(topk.params.get("top_k").map(String::as_str), Some("4"));
    assert_eq!(
        topk.params.get("scoring").map(String::as_str),
        Some("softmax")
    );
    assert_eq!(
        node(&c, "l35.moe.experts_down").binding,
        ["layers.35.experts.*.w2"]
    );
}

#[test]
fn each_site_runs_its_declared_formats() {
    let c = circuit();
    let bf16 = Format::Bf16;
    for (id, w, a) in [
        ("l0.moe.experts_gate_up", NVFP4, NVFP4),
        ("l0.moe.experts_down", NVFP4, NVFP4),
        ("l0.moe.shared_gate_up", NVFP4, NVFP4),
        ("l35.moe.shared_down", NVFP4, NVFP4),
        ("l0.moe.gate", bf16, bf16),
        ("l0.mla.q_a", bf16, bf16),
        ("l0.mla.q_b", bf16, bf16),
        ("l0.mla.kv_a", bf16, bf16),
        ("l0.mla.attend", bf16, bf16),
        ("l0.mla.o", bf16, bf16),
        ("head.lm_head", bf16, bf16),
    ] {
        assert_eq!(weight(&c, id), Some(w), "{id} weight");
        assert_eq!(input_format(&c, id), a, "{id} input");
    }
    // 2026-10-10: The routed and shared experts read one NVFP4 quantizer of the normed row; the
    // router reads it unquantized.
    let gu = node(&c, "l0.moe.experts_gate_up").inputs[0];
    assert_eq!(node(&c, "l0.moe.shared_gate_up").inputs[0], gu);
    let q = c.edges[gu].producer.unwrap();
    assert_eq!(c.nodes[q].op.name(), "act_quant:nvfp4/g16");
    assert_eq!(c.nodes[q].inputs, node(&c, "l0.moe.gate").inputs);
}

#[test]
fn the_declared_plan_of_params_json_agrees_with_the_table_except_at_the_head() {
    let mut config = base();
    config["quantization_config"] = params_json()["quantization_config"].clone();
    let r = metrale_circuit::resolve_checkpoint(
        &config.to_string(),
        QuantMetadata::default(),
        &ServePrecision::Declared,
    )
    .unwrap();
    assert_eq!(r.arch, "mla_moe");
    assert_eq!(r.kv_cache, None, "kv_cache_scheme is null: a 16-bit cache");
    let table = circuit();
    // 2026-10-10: The ignore list names the head `lm_head` (its HF name); the native `output`
    // matches only the `Linear` target, so the plan declares it W4A4 and quantizes its input,
    // although `output.weight` is stored BF16 with no scales. The table states the stored format.
    let head = "head.lm_head";
    assert_eq!(weight(&r.circuit, head), Some(NVFP4));
    assert_eq!(input_format(&r.circuit, head), NVFP4);
    let ids = |c: &Circuit| c.nodes.iter().map(|n| n.id.clone()).collect::<Vec<_>>();
    let mut declared = ids(&r.circuit);
    let extra = declared.iter().position(|i| i == "head.xn_quant").unwrap();
    declared.remove(extra);
    assert_eq!(declared, ids(&table));
    for n in table.nodes.iter().filter(|n| n.id != head) {
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
fn latent_attention_reads_wkv_b_and_the_estimate_counts_the_rotary_key() {
    let c = circuit();
    let attend = node(&c, "l0.mla.attend");
    // 2026-10-10: `wkv_b` is [6144, 256]: 32 heads x (64 NoPE key + 128 value), latent 256.
    assert_eq!(c.weight_shape(attend), Some((6144, 256)));
    let fams = metrale_circuit::venn::parse_families(&common::read(
        "kernels/gb10/common/KERNEL_FAMILIES.toml",
    ))
    .unwrap();
    let r = fams.roofline;
    let settings = &instance().policy.settings;
    let one = metrale_circuit::venn::roofline::node_cost(&c, attend, Mode::Decode, 1, settings, &r)
        .unwrap();
    let ctx = r.context_tokens as f64;
    let edges = (4096 * 2 + 4096 * 2) as f64;
    // 2026-10-10: Dense: every cached row (BF16, latent + rotary key) once per sequence.
    assert_eq!(one.bytes, 6144.0 * 256.0 * 2.0 + ctx * 320.0 * 2.0 + edges);
    let absorb = 2.0 * 32.0 * 256.0 * (64.0 + 128.0);
    assert_eq!(one.flops, absorb + 2.0 * ctx * 32.0 * (320.0 + 256.0));
    let wide =
        metrale_circuit::venn::roofline::node_cost(&c, attend, Mode::MultiSeq, 16, settings, &r)
            .unwrap();
    assert_eq!(
        wide.bytes - 16.0 * edges,
        6144.0 * 256.0 * 2.0 + 16.0 * ctx * 320.0 * 2.0
    );
    assert!(c.nodes.iter().all(|n| n.op != OpKind::PagedAttention));
}

#[test]
fn the_config_map_refuses_what_the_circuit_does_not_model() {
    let refused = |edit: &dyn Fn(&mut Value), want: &str| {
        let mut v = base();
        edit(&mut v);
        let e = metrale_circuit::map_checkpoint(&v.to_string())
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(e.contains(want), "{want}: {e}");
    };
    let tc = |k: &'static str, x: Value| move |v: &mut Value| v["text_config"][k] = x.clone();
    refused(
        &tc("first_k_dense_replace", 3.into()),
        "first_k_dense_replace",
    );
    refused(&tc("n_group", 8.into()), "grouped (n_group)");
    refused(&tc("n_shared_experts", 2.into()), "one shared expert");
    refused(&tc("rope_interleave", false.into()), "interleaved pairs");
    refused(&tc("norm_topk_prob", false.into()), "renormalises");
    refused(&tc("sliding_window", 4096.into()), "sliding-window");
    refused(&tc("attention_bias", true.into()), "no bias");
    refused(&tc("rms_norm_eps", 1e-5.into()), "fixed eps 1e-6");
    // 2026-10-10: A `mistral3` wrapper around a dense `mistral` text stack (Mistral-Small-3.1).
    refused(&tc("model_type", "mistral".into()), "Mistral-4 MLA + MoE");
    // 2026-10-10: A plain query projection (`q_lora_rank` null) has no `q_a` / `q_b` here.
    refused(&tc("q_lora_rank", Value::Null), "`q_lora_rank`");
    refused(
        &|v: &mut Value| v["text_config"]["rope_parameters"]["rope_type"] = "default".into(),
        "YaRN",
    );
    refused(
        &|v: &mut Value| v["text_config"]["rope_parameters"]["partial_rotary_factor"] = 0.5.into(),
        "`text_config.rope_parameters.partial_rotary_factor` is not mapped",
    );
    refused(
        &tc("index_topk", 2048.into()),
        "`text_config.index_topk` is not mapped",
    );
    refused(
        &|v: &mut Value| v["tie_word_embeddings"] = true.into(),
        "tied",
    );
}
