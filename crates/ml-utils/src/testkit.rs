// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Toy checkpoints for this crate's tests: the checked-in configs of the three
//! reference checkpoints (FP8 block MoE, ModelOpt NVFP4 MoE, compressed-tensors NVFP4/FP8
//! dense), cut to 8 layers and small dims, with a tensor index laid out as each real checkpoint
//! lays out its tensors (names, dtypes, scale shapes).
//!
//! Owner: metrale-ml-utils.
//! Invariants: the layouts follow the real checkpoints' headers (2026-10-03 survey): FP8 block
//! scales are BF16 `[N/128, K/128]`; ModelOpt NVFP4 has scalar `weight_scale_2` and
//! `input_scale`; compressed-tensors has `weight_packed`, `weight_global_scale [1]`,
//! `input_global_scale [1]` and BF16 `[N, 1]` FP8 channel scales.

use serde_json::{Value, json};

use crate::index::{Dtype, TensorEntry, TensorIndex};
use crate::spec::MockSpec;

pub const LAYERS: usize = 8;
pub const HIDDEN: u64 = 256;
pub const VOCAB: u64 = 512;
pub const EXPERTS: u64 = 8;
pub const TOP_K: u64 = 2;
/// 2026-10-03: A strongly skewed 8-expert load (selection counts).
pub const SKEWED: [u64; 8] = [400, 250, 150, 80, 50, 40, 20, 10];
const MOE_INTER: u64 = 128;
const DENSE_INTER: u64 = 512;
const HEADS: u64 = 2;
const KV_HEADS: u64 = 1;
const HEAD_DIM: u64 = 128;
const LK_HEADS: u64 = 2;
const LV_HEADS: u64 = 4;
const L_DIM: u64 = 64;

fn fixture(name: &str) -> Value {
    let text = match name {
        "moe_fp8" => include_str!(
            "../../circuit/tests/fixtures/checkpoints/Qwen--Qwen3.6-35B-A3B-FP8/config.json"
        ),
        "moe_nvfp4" => include_str!(
            "../../circuit/tests/fixtures/checkpoints/nvidia--Qwen3.6-35B-A3B-NVFP4/config.json"
        ),
        "dense_nvfp4" => include_str!(
            "../../circuit/tests/fixtures/checkpoints/unsloth--Qwen3.8-27B-NVFP4/config.json"
        ),
        _ => unreachable!("fixture {name}"),
    };
    serde_json::from_str(text).expect("fixture parses")
}

fn shrink(c: &mut Value, moe: bool, layers: usize) {
    let t = c.get_mut("text_config").expect("text_config");
    let types: Vec<Value> = t["layer_types"].as_array().unwrap()[..layers].to_vec();
    t["layer_types"] = Value::Array(types);
    t["num_hidden_layers"] = json!(layers);
    t["hidden_size"] = json!(HIDDEN);
    t["vocab_size"] = json!(VOCAB);
    t["num_attention_heads"] = json!(HEADS);
    t["num_key_value_heads"] = json!(KV_HEADS);
    t["head_dim"] = json!(HEAD_DIM);
    t["linear_num_key_heads"] = json!(LK_HEADS);
    t["linear_num_value_heads"] = json!(LV_HEADS);
    t["linear_key_head_dim"] = json!(L_DIM);
    t["linear_value_head_dim"] = json!(L_DIM);
    if moe {
        t["num_experts"] = json!(EXPERTS);
        t["num_experts_per_tok"] = json!(TOP_K);
        t["moe_intermediate_size"] = json!(MOE_INTER);
        t["shared_expert_intermediate_size"] = json!(MOE_INTER);
    } else {
        t["intermediate_size"] = json!(DENSE_INTER);
    }
    if let Some(root) = c.as_object_mut() {
        for k in ["head_dim", "num_attention_heads", "num_key_value_heads"] {
            if root.contains_key(k) {
                root.remove(k);
            }
        }
    }
}

/// 2026-10-03: How the toy stores a linear weight.
#[derive(Clone, Copy)]
pub enum Store {
    Bf16,
    Fp8Block,
    Fp8Tensor,
    Fp8Channel,
    Nvfp4ModelOpt,
    Nvfp4Ct,
}

fn e(name: String, dtype: Dtype, shape: Vec<u64>) -> TensorEntry {
    TensorEntry {
        name,
        dtype,
        shape,
        shard: "model.safetensors".into(),
    }
}

/// 2026-10-03: The tensors of linear `m` (`[n, k]`) stored as `s`.
pub fn linear(out: &mut Vec<TensorEntry>, m: &str, n: u64, k: u64, s: Store) {
    use Dtype::*;
    match s {
        Store::Bf16 => out.push(e(format!("{m}.weight"), Bf16, vec![n, k])),
        Store::Fp8Block => {
            out.push(e(format!("{m}.weight"), F8E4m3, vec![n, k]));
            out.push(e(
                format!("{m}.weight_scale_inv"),
                Bf16,
                vec![n.div_ceil(128), k.div_ceil(128)],
            ));
        }
        Store::Fp8Tensor => {
            out.push(e(format!("{m}.weight"), F8E4m3, vec![n, k]));
            out.push(e(format!("{m}.weight_scale"), F32, vec![]));
            out.push(e(format!("{m}.input_scale"), F32, vec![]));
        }
        Store::Fp8Channel => {
            out.push(e(format!("{m}.weight"), F8E4m3, vec![n, k]));
            out.push(e(format!("{m}.weight_scale"), Bf16, vec![n, 1]));
        }
        Store::Nvfp4ModelOpt => {
            out.push(e(format!("{m}.weight"), U8, vec![n, k / 2]));
            out.push(e(format!("{m}.weight_scale"), F8E4m3, vec![n, k / 16]));
            out.push(e(format!("{m}.weight_scale_2"), F32, vec![]));
            out.push(e(format!("{m}.input_scale"), F32, vec![]));
        }
        Store::Nvfp4Ct => {
            out.push(e(format!("{m}.weight_packed"), U8, vec![n, k / 2]));
            out.push(e(format!("{m}.weight_scale"), F8E4m3, vec![n, k / 16]));
            out.push(e(format!("{m}.weight_global_scale"), F32, vec![1]));
            out.push(e(format!("{m}.input_global_scale"), F32, vec![1]));
        }
    }
}

/// 2026-10-03: One hybrid layer (`full` attention or GDN) without its FFN.
fn mixer(out: &mut Vec<TensorEntry>, p: &str, full: bool, s: Store) {
    use Dtype::*;
    out.push(e(format!("{p}.input_layernorm.weight"), Bf16, vec![HIDDEN]));
    out.push(e(
        format!("{p}.post_attention_layernorm.weight"),
        Bf16,
        vec![HIDDEN],
    ));
    if full {
        let a = format!("{p}.self_attn");
        linear(out, &format!("{a}.q_proj"), HEADS * HEAD_DIM * 2, HIDDEN, s);
        linear(out, &format!("{a}.k_proj"), KV_HEADS * HEAD_DIM, HIDDEN, s);
        linear(out, &format!("{a}.v_proj"), KV_HEADS * HEAD_DIM, HIDDEN, s);
        linear(out, &format!("{a}.o_proj"), HIDDEN, HEADS * HEAD_DIM, s);
        out.push(e(format!("{a}.q_norm.weight"), Bf16, vec![HEAD_DIM]));
        out.push(e(format!("{a}.k_norm.weight"), Bf16, vec![HEAD_DIM]));
    } else {
        let g = format!("{p}.linear_attn");
        let qkv = LK_HEADS * L_DIM * 2 + LV_HEADS * L_DIM;
        linear(out, &format!("{g}.in_proj_qkv"), qkv, HIDDEN, s);
        linear(out, &format!("{g}.in_proj_z"), LV_HEADS * L_DIM, HIDDEN, s);
        linear(out, &format!("{g}.out_proj"), HIDDEN, LV_HEADS * L_DIM, s);
        out.push(e(
            format!("{g}.in_proj_a.weight"),
            Bf16,
            vec![LV_HEADS, HIDDEN],
        ));
        out.push(e(
            format!("{g}.in_proj_b.weight"),
            Bf16,
            vec![LV_HEADS, HIDDEN],
        ));
        out.push(e(format!("{g}.A_log"), Bf16, vec![LV_HEADS]));
        out.push(e(format!("{g}.dt_bias"), Bf16, vec![LV_HEADS]));
        out.push(e(format!("{g}.conv1d.weight"), Bf16, vec![qkv, 1, 4]));
        out.push(e(format!("{g}.norm.weight"), Bf16, vec![L_DIM]));
    }
}

fn moe_ffn(out: &mut Vec<TensorEntry>, p: &str, experts: Store, shared: Store) {
    use Dtype::*;
    out.push(e(
        format!("{p}.mlp.gate.weight"),
        Bf16,
        vec![EXPERTS, HIDDEN],
    ));
    out.push(e(
        format!("{p}.mlp.shared_expert_gate.weight"),
        Bf16,
        vec![1, HIDDEN],
    ));
    for x in 0..EXPERTS {
        let m = format!("{p}.mlp.experts.{x}");
        linear(out, &format!("{m}.gate_proj"), MOE_INTER, HIDDEN, experts);
        linear(out, &format!("{m}.up_proj"), MOE_INTER, HIDDEN, experts);
        linear(out, &format!("{m}.down_proj"), HIDDEN, MOE_INTER, experts);
    }
    let m = format!("{p}.mlp.shared_expert");
    linear(out, &format!("{m}.gate_proj"), MOE_INTER, HIDDEN, shared);
    linear(out, &format!("{m}.up_proj"), MOE_INTER, HIDDEN, shared);
    linear(out, &format!("{m}.down_proj"), HIDDEN, MOE_INTER, shared);
}

fn globals(out: &mut Vec<TensorEntry>, head: Store) {
    use Dtype::*;
    out.push(e(
        "model.language_model.embed_tokens.weight".into(),
        Bf16,
        vec![VOCAB, HIDDEN],
    ));
    out.push(e(
        "model.language_model.norm.weight".into(),
        Bf16,
        vec![HIDDEN],
    ));
    linear(out, "lm_head", VOCAB, HIDDEN, head);
    out.push(e("mtp.fc.weight".into(), Bf16, vec![HIDDEN, 2 * HIDDEN]));
    out.push(e("mtp.norm.weight".into(), Bf16, vec![HIDDEN]));
}

fn full(i: usize) -> bool {
    (i + 1).is_multiple_of(4)
}

/// 2026-10-03: The FP8-block MoE toy (Qwen/Qwen3.6-35B-A3B-FP8's layout).
pub fn moe_fp8() -> (String, TensorIndex) {
    let mut c = fixture("moe_fp8");
    shrink(&mut c, true, LAYERS);
    let mut v = Vec::new();
    for i in 0..LAYERS {
        let p = format!("model.language_model.layers.{i}");
        mixer(&mut v, &p, full(i), Store::Fp8Block);
        moe_ffn(&mut v, &p, Store::Fp8Block, Store::Fp8Block);
    }
    mixer(&mut v, "mtp.layers.0", true, Store::Fp8Block);
    moe_ffn(&mut v, "mtp.layers.0", Store::Fp8Block, Store::Fp8Block);
    globals(&mut v, Store::Bf16);
    (
        c.to_string(),
        TensorIndex::from_entries(v).expect("toy index"),
    )
}

/// 2026-10-03: The ModelOpt NVFP4 MoE toy (nvidia/Qwen3.6-35B-A3B-NVFP4's layout); its
/// `quantized_layers` keep only the toy's layers. Returns (config, sidecar, index).
pub fn moe_nvfp4() -> (String, String, TensorIndex) {
    let mut c = fixture("moe_nvfp4");
    shrink(&mut c, true, LAYERS);
    let keep = |k: &str| {
        !k.starts_with("model.language_model.layers.")
            || k.split('.')
                .nth(3)
                .and_then(|n| n.parse::<usize>().ok())
                .is_some_and(|n| n < LAYERS)
    };
    if let Some(ql) = c["quantization_config"]["quantized_layers"].as_object_mut() {
        ql.retain(|k, _| keep(k));
    }
    let side_text = include_str!(
        "../../circuit/tests/fixtures/checkpoints/nvidia--Qwen3.6-35B-A3B-NVFP4/hf_quant_config.json"
    );
    let mut side: Value = serde_json::from_str(side_text).expect("sidecar");
    if let Some(ql) = side["quantization"]["quantized_layers"].as_object_mut() {
        ql.retain(|k, _| keep(k));
    }
    let mut v = Vec::new();
    for i in 0..LAYERS {
        let p = format!("model.language_model.layers.{i}");
        mixer(&mut v, &p, full(i), Store::Fp8Tensor);
        moe_ffn(&mut v, &p, Store::Nvfp4ModelOpt, Store::Nvfp4ModelOpt);
    }
    mixer(&mut v, "mtp.layers.0", true, Store::Bf16);
    moe_ffn(&mut v, "mtp.layers.0", Store::Bf16, Store::Bf16);
    globals(&mut v, Store::Nvfp4ModelOpt);
    (
        c.to_string(),
        side.to_string(),
        TensorIndex::from_entries(v).expect("toy index"),
    )
}

/// 2026-10-03: Layers of the dense toy: three periods, so a mock drops the middle one.
pub const DENSE_LAYERS: usize = 12;

/// 2026-10-03: The compressed-tensors dense toy (unsloth/Qwen3.8-27B-NVFP4's layout): FFN NVFP4
/// in layers 0-7 and FP8 per channel in 8-11, named by a literal-index pattern, as the real
/// checkpoint names 56-63.
pub fn dense_ct() -> (String, TensorIndex) {
    let mut c = fixture("dense_nvfp4");
    shrink(&mut c, false, DENSE_LAYERS);
    let targets = &mut c["quantization_config"]["config_groups"]["group_0"]["targets"];
    for t in targets.as_array_mut().expect("targets") {
        if t.as_str().is_some_and(|s| s.contains("56|57")) {
            *t = json!("re:.*layers\\.(8|9|10|11)\\.mlp\\.(gate|up|down)_proj$");
        }
    }
    let mut v = Vec::new();
    for i in 0..DENSE_LAYERS {
        let p = format!("model.language_model.layers.{i}");
        mixer(&mut v, &p, full(i), Store::Fp8Channel);
        let ffn = if i < 8 {
            Store::Nvfp4Ct
        } else {
            Store::Fp8Channel
        };
        linear(
            &mut v,
            &format!("{p}.mlp.gate_proj"),
            DENSE_INTER,
            HIDDEN,
            ffn,
        );
        linear(
            &mut v,
            &format!("{p}.mlp.up_proj"),
            DENSE_INTER,
            HIDDEN,
            ffn,
        );
        linear(
            &mut v,
            &format!("{p}.mlp.down_proj"),
            HIDDEN,
            DENSE_INTER,
            ffn,
        );
        if full(i) {
            v.push(e(format!("{p}.self_attn.k_scale"), Dtype::Bf16, vec![1]));
            v.push(e(format!("{p}.self_attn.v_scale"), Dtype::Bf16, vec![1]));
        }
    }
    mixer(&mut v, "mtp.layers.0", true, Store::Bf16);
    globals(&mut v, Store::Fp8Channel);
    (
        c.to_string(),
        TensorIndex::from_entries(v).expect("toy index"),
    )
}

/// 2026-10-03: A spec with `per_signature` and `routing` lines substituted.
pub fn spec(per_signature: &str, routing: &str) -> MockSpec {
    let text = format!(
        "schema = 1\nseed = 11\n[layers]\nper_signature = {per_signature}\n[experts]\nkeep = \"all\"\n\
         [vocab]\nkeep = \"all\"\n[mtp]\nkeep = true\n[vision]\nkeep = true\n[capacity]\nkv = \"free\"\n\
         [routing]\n{routing}\n[values]\nmode = \"init\"\n[speculative]\naccept = \"natural\"\n"
    );
    MockSpec::parse(&text).expect("toy spec")
}
