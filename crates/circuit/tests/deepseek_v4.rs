// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The DeepSeek-V4 circuit (`deepseek_v4`, DeepSeek-V4-Flash): its config map
//! reproduces the instance's shape from the checkpoint's own `config.json` (the compress ratios
//! picking the layer kinds, the MTP layer's entry dropped), the layers take the hash-routed
//! prefix and the per-ratio attention under the hyper-connection highway, every site runs its
//! declared formats, the checkpoint's declared plan (a ModelOpt export over an FP8 base) agrees
//! with the precision table node by node, the grouped projection and the attention cost what
//! they read, and the config map refuses what the circuit does not model.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod checkpoint_fixtures;
mod common;

use std::collections::BTreeMap;

use checkpoint_fixtures::*;
use metrale_circuit::ir::Circuit;
use metrale_circuit::{Format, Instance, LayerKind, Mode, OpKind, Scale, Section};

const DSV4: &str = "deepseek-v4/deepseek-v4-flash-nvfp4-ep2";
const FIXTURE: &str = "nvidia--DeepSeek-V4-Flash-NVFP4";
const FP8_BLOCK: Format = Format::Fp8E4m3 {
    scale: Scale::Block(128, 128),
};
const FP8_G128: Format = Format::Fp8E4m3 {
    scale: Scale::Group(128),
};

fn instance() -> Instance {
    common::instances()
        .into_iter()
        .find(|i| i.recipe == DSV4)
        .expect("the DeepSeek-V4 instance")
}

fn circuit() -> Circuit {
    common::load(&instance()).circuit
}

fn edited_config(edit: impl Fn(&mut serde_json::Value)) -> String {
    let mut v: serde_json::Value = serde_json::from_str(&fixture(FIXTURE).0).unwrap();
    edit(&mut v);
    v.to_string()
}

fn refusal(edit: impl Fn(&mut serde_json::Value)) -> String {
    metrale_circuit::map_checkpoint(&edited_config(edit))
        .expect_err("refused")
        .to_string()
}

#[test]
fn the_config_map_reproduces_the_instance_shape() {
    let m = metrale_circuit::map_checkpoint(&fixture(FIXTURE).0).unwrap();
    assert_eq!(m.arch, "deepseek_v4");
    assert_eq!(m.shape, instance().shape);
    let count = |k: LayerKind| m.shape.layer_kinds.iter().filter(|&&x| x == k).count();
    assert_eq!(
        m.shape.layer_kinds.len(),
        43,
        "the MTP layer's ratio is dropped"
    );
    assert_eq!(count(LayerKind::SlidingAttention), 2);
    assert_eq!(count(LayerKind::CompressedSparseAttention), 21);
    assert_eq!(count(LayerKind::HeavilyCompressedAttention), 20);
    assert_eq!(
        m.shape.layer_kinds[42],
        LayerKind::CompressedSparseAttention
    );
    for (k, v) in [
        ("swiglu_limit", "10.0"),
        ("routed_scaling_factor", "1.5"),
        ("hc_sinkhorn_iters", "20"),
        ("hc_eps", "1e-6"),
        ("norm_topk_prob", "true"),
        ("rope_theta", "10000"),
        ("compress_rope_theta", "160000"),
        ("rope_scaling.factor", "16"),
        ("rope_scaling.original_max_position_embeddings", "65536"),
    ] {
        assert_eq!(m.params.get(k).map(String::as_str), Some(v), "{k}");
    }
}

#[test]
fn the_layers_take_the_hash_prefix_and_their_ratios_attention_under_the_highway() {
    let c = circuit();
    let mut per_layer: BTreeMap<usize, Vec<&str>> = BTreeMap::new();
    for b in &c.blocks {
        if let Some(l) = b.layer {
            per_layer.entry(l).or_default().push(b.template.as_str());
        }
    }
    assert_eq!(per_layer.len(), 43);
    assert_eq!(per_layer[&0], ["swa", "hash_moe"]);
    assert_eq!(per_layer[&1], ["swa", "hash_moe"]);
    assert_eq!(per_layer[&2], ["csa", "hash_moe"]);
    assert_eq!(per_layer[&3], ["hca", "moe"]);
    assert_eq!(per_layer[&4], ["csa", "moe"]);
    assert_eq!(per_layer[&41], ["hca", "moe"]);
    assert_eq!(per_layer[&42], ["csa", "moe"]);
    assert!(
        c.blocks.iter().all(|b| b.section == Section::Main),
        "no draft"
    );
    // 2026-10-10: The stream between layers is the 4-stream highway, and the head collapses it
    // with its own learned mix, not the GLM-5 mean.
    let highway = c.edge("l0.hash_moe.y").unwrap();
    assert_eq!(c.edges[highway].dim_value, 4 * 4096);
    let l1 = c.blocks.iter().find(|b| b.layer == Some(1)).unwrap();
    assert_eq!(l1.stream_in, Some(highway));
    let contract = node(&c, "head.contract");
    assert_eq!(contract.op, OpKind::HcContract);
    assert_eq!(contract.params["weights"], "sigmoid_mix");
    assert_eq!(
        c.edges[contract.inputs[1]].dim_value, 4,
        "the head's own hc mixes"
    );
    // 2026-10-10: Hash routing reads the token-id table; learned routing reads the bias.
    let hash = node(&c, "l0.hash_moe.top_k");
    assert_eq!(hash.binding, ["layers.0.ffn.gate.tid2eid"]);
    assert_eq!(hash.params["selection"], "token_hash");
    let learned = node(&c, "l3.moe.top_k");
    assert_eq!(learned.binding, ["layers.3.ffn.gate.bias"]);
    assert_eq!(learned.params["scoring"], "sqrtsoftplus_bias");
    assert_eq!(learned.params["top_k"], "6");
    // 2026-10-10: The CSA attention reads the indexer's selection; HCA and sliding do not.
    assert_eq!(node(&c, "l2.csa.attend").params["compressed"], "selected");
    assert_eq!(node(&c, "l3.hca.attend").params["compressed"], "all");
    assert_eq!(node(&c, "l0.swa.attend").params["compressed"], "none");
    let sel = c.edge("l2.csa.sel").unwrap();
    assert_eq!(node(&c, "l2.csa.attend").inputs[1], sel);
    assert_eq!(c.edges[sel].dim_value, 512);
    // 2026-10-10: A compressor's output has one row per run.
    let pool = c.edge("l3.hca.cpool").unwrap();
    assert_eq!(c.edges[pool].rows.text(), "n/hca_ratio");
}

#[test]
fn each_site_runs_its_declared_formats() {
    let c = circuit();
    let bf16 = Format::Bf16;
    for (id, w, a) in [
        ("l3.moe.experts_gate_up", NVFP4, NVFP4),
        ("l3.moe.experts_down", NVFP4, NVFP4),
        ("l0.hash_moe.experts_gate_up", NVFP4, NVFP4),
        ("l3.moe.shared_gate_up", FP8_BLOCK, FP8_G128),
        ("l3.moe.shared_down", FP8_BLOCK, FP8_G128),
        ("l2.csa.q_a", FP8_BLOCK, FP8_G128),
        ("l2.csa.q_b", FP8_BLOCK, FP8_G128),
        ("l2.csa.kv", FP8_BLOCK, FP8_G128),
        ("l2.csa.o_a", FP8_BLOCK, FP8_G128),
        ("l2.csa.o_b", FP8_BLOCK, FP8_G128),
        ("l2.csa.i_q", FP8_BLOCK, FP8_G128),
        ("l2.csa.c_kv", bf16, bf16),
        ("l2.csa.c_gate", bf16, bf16),
        ("l2.csa.i_kv", bf16, bf16),
        ("l2.csa.i_w", bf16, bf16),
        ("l3.hca.c_kv", bf16, bf16),
        ("l3.moe.gate", bf16, bf16),
        ("l0.swa.mix", bf16, bf16),
        ("head.mix", bf16, bf16),
        ("head.lm_head", bf16, bf16),
    ] {
        assert_eq!(node(&c, id).weight, Some(w), "{id} weight");
        assert_eq!(input_format(&c, id), a, "{id} input");
    }
    let logits = node(&c, "l3.moe.gate").outputs[0];
    assert_eq!(c.edges[logits].format, Format::F32, "FP32 router logits");
    // 2026-10-10: One dynamic FP8 quantizer of the normed row feeds both FP8 projections that
    // read it; the routed experts get their own NVFP4 one.
    let q_in = node(&c, "l2.csa.q_a").inputs[0];
    assert_eq!(node(&c, "l2.csa.kv").inputs[0], q_in);
    let q = c.edges[q_in].producer.unwrap();
    assert_eq!(c.nodes[q].op.name(), "act_quant:fp8/g128");
    let e = c.edges[node(&c, "l3.moe.experts_gate_up").inputs[0]]
        .producer
        .unwrap();
    assert_eq!(c.nodes[e].op.name(), "act_quant:nvfp4/g16");
}

#[test]
fn the_checkpoints_declared_plan_gives_every_node_the_tables_formats() {
    let r = ok(FIXTURE);
    let table = circuit();
    assert_eq!(r.arch, "deepseek_v4");
    assert_eq!(r.kv_cache, None, "no KV-cache format declared");
    let ids = |c: &Circuit| c.nodes.iter().map(|n| n.id.clone()).collect::<Vec<_>>();
    assert_eq!(ids(&r.circuit), ids(&table));
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
    // 2026-10-10: Read alone, the config's `quant_method = fp8` would declare the routed
    // experts FP8 too; the export's plan is what makes them NVFP4.
    assert_eq!(node(&r.circuit, "l5.moe.experts_down").weight, Some(NVFP4));
}

#[test]
fn the_grouped_projection_and_the_attention_cost_what_they_read() {
    let c = circuit();
    // 2026-10-10: `wo_a` is [8192, 4096]: eight 1024 x 4096 blocks, not 8192 x 32768.
    assert_eq!(
        c.weight_shape(node(&c, "l2.csa.o_a")),
        Some((8 * 1024, 64 * 512 / 8))
    );
    let fams = common::families(&instance());
    let roof = metrale_circuit::venn::families::Roofline {
        context_tokens: 8192,
        ..fams.roofline
    };
    let settings = &instance().policy.settings;
    let cost = |id: &str| {
        metrale_circuit::venn::roofline::node_cost(
            &c,
            node(&c, id),
            Mode::Decode,
            1,
            settings,
            &roof,
        )
        .unwrap()
    };
    // 2026-10-10: Per token row read: the 128-row window, plus 8192 / 128 = 64 HCA rows, plus
    // the indexer's top 512 of 8192 / 4 = 2048 CSA rows; FP8 rows of 512, 64 heads.
    let rows = |id: &str| cost(id).flops / (4.0 * 64.0 * 512.0);
    assert_eq!(rows("l0.swa.attend"), 128.0);
    assert_eq!(rows("l3.hca.attend"), 128.0 + 64.0);
    assert_eq!(rows("l2.csa.attend"), 128.0 + 512.0);
    // 2026-10-10: The footprint's KV per token: every window row (an upper bound) at FP8, the
    // compressed rows' share, and the indexer's BF16 key share.
    let f = metrale_circuit::hardware::estimate::footprint(&c, settings).unwrap();
    let swa = 512.0;
    let hca = 512.0 + 512.0 / 128.0;
    let csa = 512.0 + 512.0 / 4.0 + 128.0 / 4.0 * 2.0;
    assert_eq!(f.kv_per_token, 2.0 * swa + 20.0 * hca + 21.0 * csa);
}

#[test]
fn the_config_map_refuses_what_the_circuit_does_not_model() {
    let e = refusal(|v| v["scoring_func"] = "softmax".into());
    assert!(e.contains("sqrt(softplus"), "{e}");
    let e = refusal(|v| v["compress_ratios"][5] = 8.into());
    assert!(e.contains("compress_ratios"), "{e}");
    // 2026-10-10: The list must hold the 43 text layers and the one MTP layer.
    let e = refusal(|v| {
        v["compress_ratios"].as_array_mut().unwrap().pop();
    });
    assert!(e.contains("num_hidden_layers"), "{e}");
    let e = refusal(|v| v["num_key_value_heads"] = 2.into());
    assert!(e.contains("one shared KV row"), "{e}");
    let e = refusal(|v| v["rope_scaling"]["type"] = "linear".into());
    assert!(e.contains("YaRN"), "{e}");
    let e = refusal(|v| v["n_shared_experts"] = 2.into());
    assert!(e.contains("one shared expert"), "{e}");
    let e = refusal(|v| v["topk_method"] = "greedy".into());
    assert!(e.contains("correction bias"), "{e}");
    // 2026-10-10: Explicit compression rates would override the ratios the circuit restates.
    let e =
        refusal(|v| v["compress_rates"] = serde_json::json!({"compressed_sparse_attention": 8}));
    assert!(e.contains("compress_rates"), "{e}");
}
