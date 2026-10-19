// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

use serde_json::Value;

use super::super::{ModelCfg, Semantics};
use super::REFERENCE_GATE_TEMP;

/// 2026-10-19: The snapshot ships two configs: the Hugging Face `config.json` (what the engine
/// parses) and `inference/config.json` (the reference's `ModelArgs` keys). Every `ModelCfg`
/// field derived from the first must equal the matching key of the second, so a parser
/// mapping slip (a swapped theta, a dropped source list) shows here.
#[test]
#[ignore = "needs METRALE_DSV41_CKPT: a DeepSeek-V4.1-Flash snapshot directory"]
fn the_engine_config_agrees_with_the_reference_model_args() {
    let dir = std::path::PathBuf::from(
        std::env::var("METRALE_DSV41_CKPT").expect("METRALE_DSV41_CKPT names the snapshot"),
    );
    let read = |p: &str| std::fs::read_to_string(dir.join(p)).expect("snapshot file");
    let parsed = metrale_config::parse_config(&read("config.json")).expect("engine parse");
    let c = ModelCfg::from_model_config(&parsed, 4096, Semantics::REVISION_1017);
    let a: Value = serde_json::from_str(&read("inference/config.json")).expect("ModelArgs json");
    let u = |k: &str| a[k].as_u64().unwrap_or_else(|| panic!("{k}")) as usize;
    let f = |k: &str| a[k].as_f64().unwrap_or_else(|| panic!("{k}")) as f32;
    let list = |k: &str| -> Vec<usize> {
        a[k].as_array()
            .unwrap_or_else(|| panic!("{k}"))
            .iter()
            .map(|v| v.as_u64().expect("int") as usize)
            .collect()
    };
    let n = u("n_layers");
    let pairs: [(&str, usize, usize); 21] = [
        ("dim", c.dim, u("dim")),
        ("hc_mult", c.hc, u("hc_mult")),
        ("hc_sinkhorn_iters", c.iters, u("hc_sinkhorn_iters")),
        ("n_layers", c.n_layers, n),
        ("vocab_size", c.vocab, u("vocab_size")),
        ("n_heads", c.n_heads, u("n_heads")),
        ("head_dim", c.head_dim, u("head_dim")),
        ("rope_head_dim", c.rope_dim, u("rope_head_dim")),
        ("q_lora_rank", c.q_rank, u("q_lora_rank")),
        ("o_lora_rank", c.o_rank, u("o_lora_rank")),
        ("o_groups", c.groups, u("o_groups")),
        ("window_size", c.window, u("window_size")),
        (
            "candidate_topk_blocks",
            c.cand_topk_blocks,
            u("candidate_topk_blocks"),
        ),
        (
            "candidate_block_size",
            c.cand_block,
            u("candidate_block_size"),
        ),
        ("index_n_heads", c.index_heads, u("index_n_heads")),
        ("index_head_dim", c.index_hd, u("index_head_dim")),
        ("index_topk", c.index_topk, u("index_topk")),
        ("original_seq_len", c.orig_seq, u("original_seq_len")),
        ("moe_inter_dim", c.moe.inter, u("moe_inter_dim")),
        ("n_routed_experts", c.moe.n_routed, u("n_routed_experts")),
        ("n_activated_experts", c.moe.topk, u("n_activated_experts")),
    ];
    for (k, got, want) in pairs {
        assert_eq!(got, want, "{k}");
    }
    let fpairs: [(&str, f32, f32); 10] = [
        ("hc_eps", c.hc_eps, f("hc_eps")),
        ("norm_eps", c.eps, f("norm_eps")),
        ("rope_theta", c.rope_theta, f("rope_theta")),
        (
            "compress_rope_theta",
            c.compress_rope_theta,
            f("compress_rope_theta"),
        ),
        ("rope_factor", c.rope_factor, f("rope_factor")),
        ("beta_fast", c.beta_fast, f("beta_fast")),
        ("beta_slow", c.beta_slow, f("beta_slow")),
        ("route_scale", c.moe.route_scale, f("route_scale")),
        ("swiglu_limit", c.moe.swiglu_limit, f("swiglu_limit")),
        ("dim/moe.dim", c.moe.dim as f32, f("dim")),
    ];
    for (k, got, want) in fpairs {
        assert_eq!(got.to_bits(), want.to_bits(), "{k}");
    }
    assert_eq!(c.ratios, list("compress_ratios")[..n].to_vec());
    assert_eq!(c.kv_sources, list("kv_source_layers"));
    assert_eq!(c.index_sources, list("index_source_layers"));
    assert_eq!(
        c.cand_src,
        a["candidate_source_layer"].as_i64().expect("int")
    );
    // 2026-10-19: Neither file names these, so both resolve to the `ModelArgs` declarations.
    assert!(a.get("gate_temp").is_none() && a.get("norm_topk_prob").is_none());
    assert_eq!(c.moe.gate_temp, REFERENCE_GATE_TEMP);
    assert!(c.moe.norm_topk_prob);
}
