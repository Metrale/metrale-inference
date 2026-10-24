// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The safetensors names against the publisher's own parameter names (the tracked
//! golden, generated from the publisher's model), both ways; the GGUF names against the
//! spellings the GGUF loader has always read.

use std::collections::BTreeSet;

use super::*;
use crate::deepseek_v41_ref::GOLDEN_JSON;

fn keys(v: &serde_json::Value, out: &mut BTreeSet<String>) {
    match v {
        serde_json::Value::Object(m) => {
            for (k, x) in m {
                out.insert(k.clone());
                keys(x, out);
            }
        }
        serde_json::Value::Array(a) => a.iter().for_each(|x| keys(x, out)),
        _ => {}
    }
}

/// 2026-10-19: Every per-layer name the publisher's model holds is one of ours, and every name
/// of ours on a layer that holds it is the publisher's. Routed experts and the engram row
/// tables are read by their own readers.
#[test]
fn safetensors_names_are_the_publishers_per_layer_parameters() {
    let v: serde_json::Value = serde_json::from_str(GOLDEN_JSON).unwrap();
    let mut all = BTreeSet::new();
    keys(&v, &mut all);
    let layer_keys: Vec<&String> = all
        .iter()
        .filter(|k| k.starts_with("layers."))
        .filter(|k| !k.contains(".ffn.experts.") && !k.contains(".engram.embed."))
        .collect();
    assert!(layer_keys.len() > 100, "{} layer keys", layer_keys.len());
    let n_layers = 1 + layer_keys
        .iter()
        .map(|k| k.split('.').nth(1).unwrap().parse::<usize>().unwrap())
        .max()
        .unwrap();
    let ours: BTreeSet<String> = (0..n_layers)
        .flat_map(|l| ROLES.iter().map(move |&t| name(Format::Safetensors, l, t)))
        .collect();
    for k in &layer_keys {
        assert!(ours.contains(*k), "the publisher's {k} has no role");
    }
    for l in 0..n_layers {
        for t in [
            T::WqA,
            T::WoB,
            T::Sink,
            T::GateBias,
            T::SharedW2,
            T::FfnNorm,
        ] {
            let n = name(Format::Safetensors, l, t);
            assert!(all.contains(&n), "{n} is not the publisher's");
        }
    }
    let covered = ROLES
        .iter()
        .filter(|&&t| (0..n_layers).any(|l| all.contains(&name(Format::Safetensors, l, t))))
        .count();
    assert_eq!(covered, 31, "every role is some layer's parameter");
}

#[test]
fn gguf_names_are_the_spellings_the_gguf_loader_reads() {
    let got: Vec<String> = ROLES.iter().map(|&t| name(Format::Gguf, 7, t)).collect();
    let want = [
        "attn.wq_a.weight",
        "attn.q_norm.weight",
        "attn.wq_b.weight",
        "attn.wkv.weight",
        "attn.kv_norm.weight",
        "attn.wo_a.weight",
        "attn.wo_b.weight",
        "attn.attn_sink",
        "compressor.norm.weight",
        "compressor.wkv.weight",
        "compressor.wgate.weight",
        "indexer.wq_b.weight",
        "indexer.proj.weight",
        "indexer.wk.weight",
        "indexer.k_norm.weight",
        "ffn.gate.weight",
        "ffn.gate.e_score_correction_bias",
        "ffn.shared_experts.w1",
        "ffn.shared_experts.w2",
        "ffn.shared_experts.w3",
        "hc_attn_fn",
        "hc_attn_base",
        "hc_attn_scale",
        "hc_ffn_fn",
        "hc_ffn_base",
        "hc_ffn_scale",
        "attn_norm.weight",
        "ffn_norm.weight",
        "engram.wq",
        "engram.wk",
        "engram.wkv",
    ]
    .map(|s| format!("model.layers.7.{s}"));
    assert_eq!(got, want);
}

/// 2026-10-19: Every routed-expert and engram-row parameter the publisher's model holds is one of
/// the names the safetensors readers build, and the scale names are the checkpoint's siblings.
#[test]
fn expert_and_engram_row_names_are_the_publishers() {
    let v: serde_json::Value = serde_json::from_str(GOLDEN_JSON).unwrap();
    let mut all = BTreeSet::new();
    keys(&v, &mut all);
    let experts: Vec<&String> = all.iter().filter(|k| k.contains(".ffn.experts.")).collect();
    let rows: Vec<&String> = all
        .iter()
        .filter(|k| k.contains(".engram.embed."))
        .collect();
    assert!(!experts.is_empty() && !rows.is_empty());
    let field = |k: &str, i: usize| k.split('.').nth(i).unwrap().parse::<usize>().unwrap();
    for k in experts {
        let (l, e) = (field(k, 1), field(k, 4));
        let ours: Vec<String> = (0..6).map(|t| expert_name(l, e, t)).collect();
        assert!(
            ours.contains(k),
            "the publisher's {k} is not an expert table"
        );
    }
    for k in rows {
        let l = field(k, 1);
        let w = engram_rows_name(l);
        assert!(*k == w || *k == scale_of(&w).unwrap(), "{k}");
    }
    assert_eq!(
        scale_of("layers.0.attn.wq_b.weight").unwrap(),
        "layers.0.attn.wq_b.scale"
    );
    assert_eq!(
        scale_of(&expert_name(3, 9, 4)).unwrap(),
        expert_name(3, 9, 5)
    );
    assert!(scale_of("layers.0.attn.attn_sink").is_err());
}
