// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: Tests for `DeclaredPrecisionPlan::weight_storage` against real safetensors
//! headers: the DeepSeek-V4.1 publisher's `config.json` and the dtype and shape of every
//! tensor of layer 0 (routed experts 0 and 383 only), the layer-2 compressor and indexer, and
//! the layer-1 Engram (`tests/fixtures/deepseek_v41_flash/headers.json`, trimmed from the
//! header bytes of the published shards; no weights).
//!
//! Owner: config (quantization).
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use super::*;
use crate::parse_config;
use crate::tests::deepseek_v41::CONFIG;

const HEADERS: &str = include_str!("tests/fixtures/deepseek_v41_flash/headers.json");

#[derive(serde::Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<u64>,
}

fn element_dtype(e: StoredElement) -> &'static [&'static str] {
    match e {
        StoredElement::E4m3 => &["F8_E4M3"],
        StoredElement::E2m1Pair => &["I8", "U8"],
    }
}

fn scale_dtype(s: StoredScale) -> &'static str {
    match s {
        StoredScale::F32 => "F32",
        StoredScale::E8m0 => "F8_E8M0",
    }
}

/// 2026-10-18: Each quantized linear's logical `[out, in]`, from the config alone; the
/// header must then hold exactly the tensors the plan derives from it.
#[test]
fn declared_storage_matches_every_scaled_tensor_in_the_published_headers() {
    let c = parse_config(CONFIG).expect("parse V4.1 config.json");
    let plan = c.quantization_config.as_ref().unwrap().precision.clone();
    let headers: BTreeMap<String, Entry> = serde_json::from_str(HEADERS).unwrap();

    let (hidden, heads, hd) = (
        c.hidden_size as u64,
        c.num_attention_heads as u64,
        c.head_dim as u64,
    );
    let (q_lora, o_lora, og) = (
        c.q_lora_rank as u64,
        c.o_lora_rank as u64,
        c.o_groups as u64,
    );
    let inter = c.moe_intermediate_size as u64;
    let shared = c.shared_expert_intermediate_size as u64;
    let index_q = (c.index_n_heads * c.index_head_dim) as u64;
    // 2026-10-18: The Engram `wkv` reads one `head_dim` row per (n-gram order, head) column
    // (orders 2..=max) and writes `hc_mult` keys plus one value.
    let engram_in = ((c.engram_max_ngram_size - 1) * c.engram_n_heads * c.engram_head_dim) as u64;
    let engram_out = (c.hc_mult as u64 + 1) * hidden;

    let mut linears: Vec<(String, u64, u64)> = vec![
        ("layers.0.attn.wq_a".into(), q_lora, hidden),
        ("layers.0.attn.wq_b".into(), heads * hd, q_lora),
        ("layers.0.attn.wkv".into(), hd, hidden),
        ("layers.0.attn.wo_a".into(), og * o_lora, heads * hd / og),
        ("layers.0.attn.wo_b".into(), hidden, og * o_lora),
        ("layers.0.ffn.shared_experts.w1".into(), shared, hidden),
        ("layers.0.ffn.shared_experts.w2".into(), hidden, shared),
        ("layers.0.ffn.shared_experts.w3".into(), shared, hidden),
        ("layers.2.attn.indexer.wq_b".into(), index_q, q_lora),
        ("layers.1.engram.wkv".into(), engram_out, engram_in),
    ];
    for e in [0, c.num_experts - 1] {
        let p = format!("layers.0.ffn.experts.{e}");
        linears.push((format!("{p}.w1"), inter, hidden));
        linears.push((format!("{p}.w2"), hidden, inter));
        linears.push((format!("{p}.w3"), inter, hidden));
    }

    for (module, rows, cols) in &linears {
        let s = plan
            .weight_storage(module, *rows, *cols)
            .unwrap()
            .unwrap_or_else(|| panic!("{module} declared unquantized"));
        let w = &headers[&format!("{module}.weight")];
        let sc = &headers[&format!("{module}.scale")];
        assert!(
            element_dtype(s.element).contains(&w.dtype.as_str()),
            "{module}: {}",
            w.dtype
        );
        assert_eq!(w.shape, s.shape, "{module} weight shape");
        assert_eq!(sc.dtype, scale_dtype(s.scale), "{module} scale dtype");
        assert_eq!(sc.shape, s.scale_shape, "{module} scale shape");
    }

    // 2026-10-18: Every scaled linear in the fixture is checked above; the Engram embedding
    // table is not a linear layer (its scales are per row of 32, not per 32x32 block).
    let checked: Vec<&str> = linears.iter().map(|(m, ..)| m.as_str()).collect();
    for name in headers.keys().filter_map(|k| k.strip_suffix(".scale")) {
        assert!(
            checked.contains(&name) || name.ends_with(".engram.embed"),
            "{name} has scales but no expectation"
        );
    }
}

#[test]
fn storage_refuses_what_it_does_not_model_and_reports_unquantized() {
    let plan = DeclaredPrecisionPlan::from_quantization_config(&serde_json::json!({
        "quant_method": "fp8", "weight_block_size": [128, 128], "expert_dtype": "fp4",
        "modules_to_not_convert": ["lm_head"]
    }))
    .unwrap();
    // 2026-10-18: No `scale_fmt`: block scales are F32, ceil-divided at the edge.
    let s = plan
        .weight_storage("model.layers.0.self_attn.o_proj", 300, 129)
        .unwrap()
        .unwrap();
    assert_eq!((s.scale, s.scale_shape), (StoredScale::F32, [3, 2]));
    assert_eq!(plan.weight_storage("lm_head", 8, 8).unwrap(), None);
    let odd = plan.weight_storage("model.layers.0.mlp.experts.3.down_proj", 8, 33);
    assert!(odd.unwrap_err().to_string().contains("even K"));

    let nvfp4 = DeclaredPrecisionPlan::from_quantization_config(&serde_json::json!({
        "quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16
    }))
    .unwrap();
    let e = nvfp4
        .weight_storage("model.layers.0.mlp.down_proj", 8, 32)
        .unwrap_err();
    assert!(e.to_string().contains("not modelled"), "{e:#}");
}
