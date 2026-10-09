// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The GLM-5 circuit (`glm5_next`, GLM-5.3-Flash): its config map reproduces the
//! instance's shape from the checkpoint's own `config.json`, the layers take the dense prefix and
//! the hyper-connection highway, every site runs its declared formats, the checkpoint's declared
//! plan agrees with the precision table node by node, latent attention reads `kv_b_proj`, and the
//! config map refuses what the circuit does not model.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod checkpoint_fixtures;
mod common;

use std::collections::BTreeMap;

use checkpoint_fixtures::*;
use metrale_circuit::ir::Circuit;
use metrale_circuit::{Format, Instance, Mode, Section};

const GLM: &str = "glm-5.3/glm-5.3-flash-nvfp4";
const FIXTURE: &str = "nvidia--GLM-5.3-Flash-NVFP4";

fn instance() -> Instance {
    common::instances()
        .into_iter()
        .find(|i| i.recipe == GLM)
        .expect("the GLM instance")
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
    assert_eq!(m.arch, "glm5_next");
    assert_eq!(m.shape, instance().shape);
    for (k, v) in [
        ("swiglu_limit", "10.0"),
        ("routed_scaling_factor", "2.5"),
        ("linear_lower_bound", "-5.0"),
        ("hc_sinkhorn_iters", "20"),
        ("norm_topk_prob", "true"),
    ] {
        assert_eq!(m.params.get(k).map(String::as_str), Some(v), "{k}");
    }
}

#[test]
fn the_layers_take_the_dense_prefix_then_the_moe_under_the_highway() {
    let c = circuit();
    let mut per_layer: BTreeMap<usize, Vec<&str>> = BTreeMap::new();
    for b in &c.blocks {
        if let Some(l) = b.layer {
            per_layer.entry(l).or_default().push(b.template.as_str());
        }
    }
    assert_eq!(per_layer.len(), 45);
    for l in 0..3 {
        assert_eq!(per_layer[&l], ["kda", "mlp"], "layer {l}");
    }
    assert_eq!(per_layer[&3], ["dsa", "moe"]);
    assert_eq!(per_layer[&4], ["kda", "moe"]);
    assert_eq!(per_layer[&43], ["dsa", "moe"]);
    assert_eq!(per_layer[&44], ["kda", "moe"]);
    let draft: Vec<&str> = c
        .blocks
        .iter()
        .filter(|b| b.section == Section::Draft)
        .map(|b| b.template.as_str())
        .collect();
    assert_eq!(draft, ["mtp_in", "mtp_dsa", "mtp_moe", "mtp_out"]);
    // 2026-10-08: The stream between layers is the 4-stream highway; the head contracts it and
    // the MTP head reads the final-normed row.
    let highway = c.edge("l0.mlp.y").unwrap();
    assert_eq!(c.edges[highway].dim_value, 4 * 4096);
    assert_eq!(
        c.blocks
            .iter()
            .find(|b| b.layer == Some(1))
            .unwrap()
            .stream_in,
        Some(highway)
    );
    assert_eq!(c.edges[c.edge("head.h").unwrap()].dim_value, 4096);
    let mtp_in = c.blocks.iter().find(|b| b.template == "mtp_in").unwrap();
    assert_eq!(mtp_in.stream_in, c.edge("head.xn"));
}

#[test]
fn each_site_runs_its_declared_formats() {
    let c = circuit();
    let fp4 = NVFP4;
    for (id, w, a) in [
        ("l3.moe.experts_gate_up", fp4, fp4),
        ("l3.moe.experts_down", fp4, fp4),
        ("l0.mlp.gate_up", fp4, fp4),
        ("l2.mlp.down", fp4, fp4),
        ("l3.moe.shared_gate_up", Format::Bf16, Format::Bf16),
        ("l3.moe.gate", Format::Bf16, Format::Bf16),
        ("l0.kda.q", Format::Bf16, Format::Bf16),
        ("l0.kda.f_b", Format::Bf16, Format::Bf16),
        ("l3.dsa.attend", Format::Bf16, Format::Bf16),
        ("l3.dsa.idx_q", Format::Bf16, Format::Bf16),
        ("l0.kda.mix", Format::Bf16, Format::Bf16),
        ("head.lm_head", Format::Bf16, Format::Bf16),
        ("draft.mtp_moe.experts_gate_up", Format::Bf16, Format::Bf16),
    ] {
        assert_eq!(weight(&c, id), Some(w), "{id} weight");
        assert_eq!(input_format(&c, id), a, "{id} input");
    }
    let logits = node(&c, "l3.moe.gate").outputs[0];
    assert_eq!(c.edges[logits].format, Format::F32, "FP32 router logits");
    // 2026-10-08: The W4A4 inputs come from quantizer nodes the precision inserted.
    let q = c.edges[node(&c, "l3.moe.experts_gate_up").inputs[0]]
        .producer
        .unwrap();
    assert_eq!(c.nodes[q].op.name(), "act_quant:nvfp4/g16");
}

#[test]
fn the_checkpoints_declared_plan_gives_every_node_the_tables_formats() {
    let r = ok(FIXTURE);
    let table = circuit();
    assert_eq!(r.arch, "glm5_next");
    assert_eq!(r.kv_cache, Some(FP8_TENSOR));
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
fn latent_attention_reads_kv_b_proj_and_the_estimate_counts_it() {
    let c = circuit();
    let attend = node(&c, "l3.dsa.attend");
    // 2026-10-08: The checkpoint's `kv_b_proj` is [32768, 512] (64 heads x (256 + 256), latent 512).
    assert_eq!(c.weight_shape(attend), Some((32768, 512)));
    assert_eq!(c.weight_shape(node(&c, "l0.kda.q")), Some((8192, 4096)));
    let fams = metrale_circuit::venn::parse_families(&common::read(
        "kernels/gb10/common/KERNEL_FAMILIES.toml",
    ))
    .unwrap();
    let r = fams.roofline;
    let settings = &instance().policy.settings;
    let cost = |rows: u64| {
        metrale_circuit::venn::roofline::node_cost(&c, attend, Mode::MultiSeq, rows, settings, &r)
            .unwrap()
    };
    let attended = (r.context_tokens as f64).min(2048.0 + 4.0);
    let one = metrale_circuit::venn::roofline::node_cost(&c, attend, Mode::Decode, 1, settings, &r)
        .unwrap();
    let edges = (64 * 256 * 2 + 512 * 4 + 64 * 256 * 2) as f64;
    assert_eq!(one.bytes, 32768.0 * 512.0 * 2.0 + attended * 512.0 + edges);
    // 2026-10-08: The weight is read once per step; the latent rows once per sequence.
    let wide = cost(16).bytes - 16.0 * edges;
    assert_eq!(wide, 32768.0 * 512.0 * 2.0 + 16.0 * attended * 512.0);
    let select = node(&c, "l3.dsa.select");
    let s = metrale_circuit::venn::roofline::node_cost(&c, select, Mode::Decode, 1, settings, &r)
        .unwrap();
    assert!(
        s.bytes >= r.context_tokens as f64 * 32.0 * 2.0,
        "{}",
        s.bytes
    );
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
    refused(
        &|v| v["text_config"]["qk_rope_head_dim"] = 64.into(),
        "qk_rope_head_dim",
    );
    refused(
        &|v| v["text_config"]["scoring_func"] = "softmax".into(),
        "scoring_func",
    );
    refused(
        &|v| v["text_config"]["mlp_layer_types"][3] = "dense".into(),
        "mlp_layer_types",
    );
    refused(
        &|v| v["text_config"]["mla_rope_scaling"] = 1.into(),
        "mla_rope_scaling",
    );
    // 2026-10-08: A DSA layer inside the dense prefix maps, but the circuit's prefix has no DSA
    // blocks, so it does not instantiate.
    let dsa_first = edited_config(|v: &mut serde_json::Value| {
        v["text_config"]["layer_types"][1] = "deepseek_sparse_attention".into();
    });
    let e = metrale_circuit::resolve_checkpoint(
        &dsa_first,
        metrale_circuit::QuantMetadata::default(),
        &metrale_circuit::ServePrecision::Declared,
    )
    .map(|_| ())
    .unwrap_err()
    .to_string();
    assert!(e.contains("`first_dense` prefix maps to no blocks"), "{e}");
}
