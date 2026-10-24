// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The DeepSeek-V4.1 circuit (`deepseek_v41`, DeepSeek-V4.1-Flash-1017) against
//! its checkpoint: the instance shape is the checkpoint's `config.json`, every layer takes the
//! template its config lists (sliding window, compressed source, reuse, re-index, Engram), every
//! binding names a stored tensor of the bound shape and stored format, every stored text-model
//! tensor is bound by a node, and the KV footprint is the packed compressed rows of the source
//! layers; rank 0 of the three-box split, from the stored tensors, boots at the class's util
//! ceiling and holds the sequences its plan promises.
//!
//! Fixtures: the config is the G1 parser's (crates/config/src/tests/fixtures/deepseek_v41_flash);
//! the tensor list is every shard header's text-model entries (`name dtype shape`), routed
//! experts folded to `experts.*` (expert 0's entry; all experts share one shape and dtype).
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod common;

use std::collections::{BTreeMap, BTreeSet};

use metrale_circuit::hardware::device::parse_devices;
use metrale_circuit::hardware::estimate::footprint;
use metrale_circuit::ir::Circuit;
use metrale_circuit::memory::box_plan::{BoxInputs, BoxPlan, BoxPlanError, BoxTerm, Util};
use metrale_circuit::memory::budget::DriverTerms;
use metrale_circuit::{Format, Instance, Scale};
use serde_json::Value;

const RECIPE: &str = "deepseek-v4.1/deepseek-v4.1-flash-mxfp4";
const CONFIG: &str = "crates/config/src/tests/fixtures/deepseek_v41_flash/config.json";
const TENSORS: &str =
    "crates/circuit/tests/fixtures/checkpoints/deepseek-ai--DeepSeek-V4.1-Flash-1017/tensors.txt";

fn instance() -> Instance {
    common::instances()
        .into_iter()
        .find(|i| i.recipe == RECIPE)
        .expect("the DeepSeek-V4.1 instance")
}

fn circuit() -> Circuit {
    common::load(&instance()).circuit
}

fn text_config() -> Value {
    let v: Value = serde_json::from_str(&common::read(CONFIG)).expect("config parses");
    v["text_config"].clone()
}

fn u(v: &Value, k: &str) -> u64 {
    v[k].as_u64()
        .unwrap_or_else(|| panic!("config `{k}` is not an integer"))
}

fn ids(v: &Value, k: &str) -> BTreeSet<u64> {
    v[k].as_array()
        .unwrap_or_else(|| panic!("config `{k}` is not a list"))
        .iter()
        .map(|x| x.as_u64().expect("an integer id"))
        .collect()
}

/// 2026-10-18: The stored tensors, name to (dtype, shape).
fn tensors() -> BTreeMap<String, (String, Vec<u64>)> {
    common::read(TENSORS)
        .lines()
        .map(|l| {
            let f: Vec<&str> = l.split(' ').collect();
            let shape = f[2].split('x').map(|d| d.parse().expect("a dim")).collect();
            (f[0].to_string(), (f[1].to_string(), shape))
        })
        .collect()
}

#[test]
fn the_instance_shape_is_the_checkpoint_config() {
    let t = text_config();
    let ratios: Vec<u64> = t["compress_ratios"]
        .as_array()
        .expect("compress_ratios")
        .iter()
        .map(|r| r.as_u64().expect("a ratio"))
        .collect();
    let layers = u(&t, "num_hidden_layers");
    let first_of = |r: u64| ratios.iter().position(|&x| x == r).expect("a ratio") as u64;
    let want = BTreeMap::from([
        ("hidden", u(&t, "hidden_size")),
        ("vocab", u(&t, "vocab_size")),
        ("hc", u(&t, "hc_mult")),
        ("q_heads", u(&t, "num_attention_heads")),
        ("q_lora", u(&t, "q_lora_rank")),
        ("head_dim", u(&t, "head_dim")),
        ("rope_dim", u(&t, "qk_rope_head_dim")),
        ("o_groups", u(&t, "o_groups")),
        ("o_lora", u(&t, "o_lora_rank")),
        ("window", u(&t, "sliding_window")),
        ("csa_ratio", ratios[first_of(2) as usize]),
        ("hca_ratio", ratios[first_of(1) as usize]),
        ("late_first", first_of(1)),
        ("index_heads", u(&t, "index_n_heads")),
        ("index_head_dim", u(&t, "index_head_dim")),
        ("index_topk", u(&t, "index_topk")),
        ("experts", u(&t, "n_routed_experts")),
        ("top_k", u(&t, "num_experts_per_tok")),
        ("moe_inter", u(&t, "moe_intermediate_size")),
        (
            "shared_inter",
            u(&t, "n_shared_experts") * u(&t, "moe_intermediate_size"),
        ),
        (
            "engram_cols",
            (u(&t, "engram_max_ngram_size") - 1) * u(&t, "engram_n_heads"),
        ),
        ("engram_head_dim", u(&t, "engram_head_dim")),
    ]);
    let shape = instance().shape;
    let got: BTreeMap<&str, u64> = shape.dims.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    assert_eq!(got, want);
    assert_eq!(shape.layer_kinds.len() as u64, layers);
    // 2026-10-18: The MTP/DSpark layers carry the trailing ratios; the circuit models none.
    assert_eq!(
        ratios.len() as u64,
        layers + u(&t, "num_nextn_predict_layers")
    );
}

#[test]
fn each_layer_takes_the_template_its_config_lists() {
    let t = text_config();
    let (kv_src, idx_src, engram) = (
        ids(&t, "kv_source_layer_ids"),
        ids(&t, "index_source_layer_ids"),
        ids(&t, "engram_layer_ids"),
    );
    let c = circuit();
    let mut got: BTreeMap<u64, Vec<&str>> = BTreeMap::new();
    for b in &c.blocks {
        if let Some(l) = b.layer {
            got.entry(l as u64).or_default().push(b.template.as_str());
        }
    }
    let layers = u(&t, "num_hidden_layers");
    assert_eq!(got.len() as u64, layers);
    for (l, templates) in &got {
        let ratio = t["compress_ratios"][*l as usize].as_u64().expect("a ratio");
        let attn = match (ratio, kv_src.contains(l), idx_src.contains(l)) {
            (0, _, _) => "swa",
            (2, true, _) => "csa_src",
            (2, false, _) => "csa_reuse",
            (1, true, _) => "hca_src",
            (1, false, true) => "hca_reindex",
            (1, false, false) => "hca_reuse",
            other => panic!("layer {l}: no template for {other:?}"),
        };
        let mut want = Vec::new();
        if engram.contains(l) {
            want.push("engram");
        }
        want.extend([attn, "moe"]);
        assert_eq!(templates, &want, "layer {l}");
    }
}

/// 2026-10-18: The stored dtype of a weight of format `f`, and whether it ships a scale tensor.
/// Every quantized weight here carries E8M0 scales (`scale_fmt = ue8m0`).
fn stored(f: Format) -> (&'static str, bool) {
    match f {
        Format::Mxfp4 { group: 32 } => ("I8", true),
        Format::Fp8E4m3 {
            scale: Scale::Block(32, 32),
        } => ("F8_E4M3", true),
        Format::Bf16 => ("BF16", false),
        other => panic!("no stored form declared for {other:?}"),
    }
}

#[test]
fn every_binding_names_a_tensor_of_the_bound_shape_and_format() {
    let (c, t) = (circuit(), tensors());
    let mut checked = 0;
    for n in c.nodes.iter().filter(|n| n.layer.is_some()) {
        for b in &n.binding {
            assert!(
                t.contains_key(b) || t.contains_key(&format!("{b}.weight")),
                "{}: `{b}` names no stored tensor",
                n.id
            );
        }
        let (Some(f), Some((out, k))) = (n.weight, c.weight_shape(n)) else {
            continue;
        };
        let (dtype, scaled) = stored(f);
        let mut rows = 0;
        for b in &n.binding {
            let (d, shape) = t
                .get(&format!("{b}.weight"))
                .or_else(|| t.get(b))
                .unwrap_or_else(|| panic!("{}: `{b}` stores no weight", n.id));
            // 2026-10-18: The hyper-connection mix stores FP32; the circuit counts it 16-bit.
            let hc_fn = b.ends_with("_fn") && d == "F32";
            assert!(
                d == dtype || hc_fn,
                "{}: `{b}` stores {d}, declared {f:?}",
                n.id
            );
            let cols = if d == "I8" { 2 * shape[1] } else { shape[1] };
            assert_eq!(cols, k, "{}: `{b}` K", n.id);
            assert_eq!(
                t.contains_key(&format!("{b}.scale")),
                scaled,
                "{}: `{b}` scale",
                n.id
            );
            rows += shape[0];
        }
        assert_eq!(rows, out, "{}: rows", n.id);
        checked += 1;
    }
    assert!(checked > 0, "no weight was compared");
}

#[test]
fn every_stored_text_tensor_is_bound() {
    let c = circuit();
    let bound: BTreeSet<String> = c
        .nodes
        .iter()
        .flat_map(|n| n.binding.iter())
        .flat_map(|b| [b.clone(), format!("{b}.weight")])
        .collect();
    let unbound: Vec<String> = tensors()
        .into_keys()
        // 2026-10-18: Scales ride their weight; `bias_vl` routes image spans, not modelled.
        .filter(|k| !k.ends_with(".scale") && !k.ends_with(".bias_vl"))
        .filter(|k| !bound.contains(k))
        .collect();
    assert!(unbound.is_empty(), "unbound: {unbound:?}");
}

#[test]
fn the_kv_footprint_is_the_packed_compressed_rows_of_the_source_layers() {
    let t = text_config();
    let (d, id) = (u(&t, "head_dim"), u(&t, "index_head_dim"));
    // 2026-10-18: Per source layer and ratio `r`: the latent row as E2M1 plus one byte per 16,
    // the index key as E2M1 plus one byte per 32, one row each per `r` tokens.
    let want: u64 = ids(&t, "kv_source_layer_ids")
        .iter()
        .map(|&l| {
            let r = t["compress_ratios"][l as usize].as_u64().expect("a ratio");
            (d / 2 + d / 16 + id / 2 + id / 32) / r
        })
        .sum();
    let f = footprint(&circuit(), &instance().policy.settings).expect("footprint");
    assert_eq!(f.kv_per_token, want as f64);
}

#[test]
fn the_grouped_output_projection_reads_one_group_of_heads() {
    let c = circuit();
    let n = &c.nodes[c.node("l0.swa.o_a").expect("o_a")];
    let s = instance().shape.dims;
    assert_eq!(
        c.weight_shape(n),
        Some((
            s["o_groups"] * s["o_lora"],
            s["q_heads"] * s["head_dim"] / s["o_groups"]
        ))
    );
}

/// 2026-10-18: The bytes of a stored tensor.
fn stored_bytes(dtype: &str, shape: &[u64]) -> u64 {
    let size = match dtype {
        "BF16" => 2,
        "F32" => 4,
        "F8_E4M3" | "F8_E8M0" | "I8" => 1,
        other => panic!("no element size for {other}"),
    };
    shape.iter().product::<u64>() * size
}

/// 2026-10-18: Rank `rank`'s resident main weights at TP `tp`, everything on the device: the
/// routed and shared experts split their intermediate columns evenly, the grouped attention
/// projections (`wq_b`, `wo_a`, `wo_b`) split in whole output groups, the embedding and head
/// split their vocabulary rows, and every other tensor (the latent `wkv`, which one KV head
/// cannot split, and the hyper-connection, router, compressor, indexer and norm tensors) is
/// replicated. The Engram tables stay off the device. A scale rides its weight.
fn rank_weights(tp: usize, rank: usize) -> u64 {
    let t = text_config();
    let groups = u(&t, "o_groups") as usize;
    let owned = metrale_config::tp_split(groups, tp, rank, 1)
        .expect("the output groups split")
        .len as u64;
    let experts = u(&t, "n_routed_experts");
    assert!(u(&t, "moe_intermediate_size").is_multiple_of(tp as u64));
    let attn = |k: &str| {
        [".attn.wq_b.", ".attn.wo_a.", ".attn.wo_b."]
            .iter()
            .any(|p| k.contains(p))
    };
    tensors()
        .iter()
        .filter(|(k, _)| !k.contains(".engram.embed."))
        .map(|(k, (d, shape))| {
            let b = stored_bytes(d, shape);
            if k.contains(".experts.*.") {
                b * experts / tp as u64
            } else if k.contains(".shared_experts.")
                || k.starts_with("embed.")
                || k.starts_with("head.")
            {
                b / tp as u64
            } else if attn(k) {
                b * owned / groups as u64
            } else {
                b
            }
        })
        .sum()
}

const GIB: f64 = (1u64 << 30) as f64;

/// 2026-10-18: The gb10 box: its visible memory (kernels/DEVICES.toml) and its class's driver
/// terms and util ceiling (kernels/gb10/HARDWARE.toml).
fn gb10() -> (u64, DriverTerms) {
    let reg = parse_devices(&common::read("kernels/DEVICES.toml")).expect("devices");
    let d = reg.devices.iter().find(|d| d.id == "gb10").expect("gb10");
    let terms = DriverTerms::parse(&d.class, &common::read("kernels/gb10/HARDWARE.toml"))
        .expect("driver terms");
    (d.memory_bytes as u64, terms)
}

/// 2026-10-18: Rank 0 of the three-box split, everything resident at the declared formats:
/// the plan boots at the class's util ceiling, refuses any util above it, and holds 16
/// sequences at a 64k context but not at 128k (where a lever must free room first).
#[test]
fn the_three_box_rank_plan_boots_at_the_ceiling_and_holds_16_sequences_to_64k() {
    let (device, driver) = gb10();
    let above = driver.util_ceiling + 0.05;
    assert_eq!(
        Util::new(above, &driver),
        Err(BoxPlanError::Util {
            util: above,
            ceiling: driver.util_ceiling
        })
    );
    let gib = |g: f64| (g * GIB) as u64;
    let inputs = BoxInputs {
        device_bytes: device,
        util: Util::new(driver.util_ceiling, &driver).expect("the ceiling is allowed"),
        driver,
        weights: rank_weights(3, 0),
        // 2026-10-18: The plan's caps for the terms no checkpoint here sizes: the drafter's
        // rank share (its weights are not in this fixture), capped collective buffers, an
        // unmeasured workspace cap and the Engram staging rows.
        drafter: gib(2.63),
        comm: gib(0.30),
        workspace: gib(2.00),
        staging: gib(0.10),
    };
    let plan = BoxPlan::new(&inputs).expect("rank 0 fits");
    // 2026-10-18: Rank 0 owns three of the eight output groups; rank 2 owns two and is lighter.
    assert!(rank_weights(3, 2) < plan.term(BoxTerm::Weights));
    assert_eq!(rank_weights(3, 1), plan.term(BoxTerm::Weights));
    let f = footprint(&circuit(), &instance().policy.settings).expect("footprint");
    let per_seq = |ctx: u64| (f.kv_per_token * ctx as f64 + f.state_per_seq) as u64;
    assert!(plan.sequences(per_seq(64 * 1024)).expect("non-zero") >= 16);
    assert!(plan.sequences(per_seq(128 * 1024)).expect("non-zero") < 16);
    // 2026-10-18: A workspace that eats the room is refused at build, not at the first request.
    let over = BoxInputs {
        workspace: plan.kv_room + inputs.workspace + 1,
        ..inputs
    };
    assert!(matches!(
        BoxPlan::new(&over),
        Err(BoxPlanError::Over { .. })
    ));
}

fn edited_config(edit: impl Fn(&mut Value)) -> String {
    let mut v: Value = serde_json::from_str(&common::read(CONFIG)).expect("config parses");
    edit(&mut v);
    v.to_string()
}

/// 2026-10-18: The checkpoint alone picks this circuit: the config map gives the instance's
/// shape, carries the math parameters, and the declared plan instantiates the same circuit.
#[test]
fn the_config_map_reproduces_the_instance_shape() {
    let config = common::read(CONFIG);
    let m = metrale_circuit::map_checkpoint(&config).expect("the config maps");
    assert_eq!(m.arch, "deepseek_v41");
    assert_eq!(m.shape, instance().shape);
    for (k, v) in [
        ("swiglu_limit", "10.0"),
        ("routed_scaling_factor", "1.5"),
        ("compress_rope_theta", "160000"),
        ("rope_scaling.rope_type", "\"yarn\""),
        ("engram_compressed_vocab_size", "99092"),
    ] {
        assert_eq!(m.params.get(k).map(String::as_str), Some(v), "{k}");
    }
    let r = metrale_circuit::resolve_checkpoint(
        &config,
        metrale_circuit::QuantMetadata::default(),
        &metrale_circuit::ServePrecision::Declared,
    )
    .expect("the checkpoint resolves");
    // 2026-10-18: Every weight the checkpoint stores with a scale takes the instance's format
    // from `quantization_config` alone. The block cannot say which linears the publisher kept
    // 16-bit (the reference code fixes the router, head, compressor and indexer projections at
    // bf16), so the unscaled ones are not compared here.
    let t = tensors();
    let scaled = |n: &metrale_circuit::ir::Node| {
        n.binding
            .first()
            .is_some_and(|b| t.contains_key(&format!("{b}.scale")))
    };
    let formats = |c: &Circuit| {
        c.nodes
            .iter()
            .filter(|n| n.weight.is_some() && scaled(n))
            .map(|n| (n.id.clone(), n.weight))
            .collect::<BTreeMap<_, _>>()
    };
    let (got, want) = (formats(&r.circuit), formats(&circuit()));
    assert!(
        want.get("l3.moe.experts_down") == Some(&Some(Format::Mxfp4 { group: 32 })),
        "{want:?}"
    );
    assert_eq!(got, want);
}

#[test]
fn the_config_map_refuses_what_the_circuit_does_not_model() {
    let refused = |edit: &dyn Fn(&mut Value), want: &str| {
        let e = metrale_circuit::map_checkpoint(&edited_config(edit))
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(e.contains(want), "{want}: {e}");
    };
    // 2026-10-18: The draft tail must be exactly `num_nextn_predict_layers` entries.
    refused(
        &|v| {
            v["text_config"]["compress_ratios"]
                .as_array_mut()
                .expect("a list")
                .pop();
        },
        "40 text layers + 3 draft layers",
    );
    refused(
        &|v| v["text_config"]["num_nextn_predict_layers"] = 2.into(),
        "40 text layers + 2 draft layers",
    );
    // 2026-10-18: A ratio-2 layer after the first ratio-1 layer would take the HCA template.
    refused(
        &|v| v["text_config"]["compress_ratios"][25] = 2.into(),
        "must all be 1",
    );
    refused(
        &|v| {
            for r in v["text_config"]["compress_ratios"]
                .as_array_mut()
                .expect("a list")
            {
                if r == 1 {
                    *r = 2.into();
                }
            }
        },
        "the first entry equal to 1",
    );
    refused(
        &|v| v["text_config"]["compress_ratios"][3] = 4.into(),
        "a layer kind this arch maps",
    );
    refused(
        &|v| v["text_config"]["kv_source_layer_ids"][1] = 9.into(),
        "kv_source_layer_ids",
    );
    refused(
        &|v| v["text_config"]["engram_n_heads"] = 16.into(),
        "engram_n_heads",
    );
    refused(
        &|v| v["text_config"]["scoring_func"] = "sigmoid".into(),
        "scoring_func",
    );
    refused(
        &|v| v["text_config"]["mla_rope_scaling"] = 1.into(),
        "mla_rope_scaling",
    );
    refused(&|v| v["mtp_layout"] = 1.into(), "mtp_layout");
}
