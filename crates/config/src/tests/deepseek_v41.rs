// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: Tests for parsing the DeepSeek-V4.1 Hugging Face `config.json`, against the
//! publisher's file (`fixtures/deepseek_v41_flash/config.json`, unmodified).
//!
//! Owner: config (model parsers).
//! Invariants: none beyond the types.

use super::*;
use crate::precision_plan::{Granularity, Operand};

pub(crate) const CONFIG: &str = include_str!("fixtures/deepseek_v41_flash/config.json");

fn without(path: &[&str]) -> String {
    let mut v: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    let (last, parents) = path.split_last().unwrap();
    let mut obj = &mut v;
    for p in parents {
        obj = obj.get_mut(*p).unwrap();
    }
    assert!(
        obj.as_object_mut().unwrap().remove(*last).is_some(),
        "{path:?}"
    );
    v.to_string()
}

fn with(path: &[&str], value: serde_json::Value) -> String {
    let mut v: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    let (last, parents) = path.split_last().unwrap();
    let mut obj = &mut v;
    for p in parents {
        obj = obj.get_mut(*p).unwrap();
    }
    obj[*last] = value;
    v.to_string()
}

#[test]
fn publisher_config_maps_every_field_the_model_reads() {
    let c = parse_config(CONFIG).expect("parse V4.1 config.json");
    assert_eq!(c.model_type, "deepseek_v41");
    assert!(c.nested_config);
    assert_eq!(c.weight_prefix, "model");
    assert_eq!(
        (c.hidden_size, c.num_hidden_layers, c.vocab_size),
        (5120, 40, 129280)
    );
    assert_eq!((c.num_attention_heads, c.num_key_value_heads), (64, 1));
    assert_eq!((c.head_dim, c.kv_lora_rank, c.v_head_dim), (512, 512, 512));
    assert_eq!(
        (c.qk_rope_head_dim, c.qk_nope_head_dim, c.rotary_dim),
        (64, 448, 64)
    );
    assert_eq!((c.q_lora_rank, c.o_lora_rank, c.o_groups), (1280, 1024, 8));
    assert_eq!(
        (c.num_experts, c.n_routed_experts, c.num_experts_per_tok),
        (384, 384, 6)
    );
    assert_eq!(
        (c.moe_intermediate_size, c.shared_expert_intermediate_size),
        (2304, 2304)
    );
    assert!(c.norm_topk_prob && c.use_routing_bias);
    assert_eq!(c.scoring_func, "sqrtsoftplus");
    assert_eq!(c.routed_scaling_factor, 1.5);
    assert_eq!(c.swiglu_limit, 10.0);
    assert_eq!(c.sliding_window, 128);
    assert_eq!(c.rms_norm_eps, 1e-20);
    assert_eq!(c.compress_ratios.len(), 43);
    // 2026-10-18: Ratio 0 (layers 0-1) is window-only; every other layer also reads the
    // indexed compressed KV.
    let mut kinds = vec![LayerType::SparseAttention; 40];
    kinds[..2].fill(LayerType::SlidingAttention);
    assert_eq!(c.layer_types, kinds);
    // 2026-10-18: Both rope bases are kept; the compressed one does not replace the other.
    assert_eq!((c.rope_theta, c.compress_rope_theta), (10000.0, 160000.0));
    assert_eq!(
        (
            c.yarn_factor,
            c.yarn_beta_fast,
            c.yarn_beta_slow,
            c.yarn_original_max_position_embeddings
        ),
        (16.0, 32.0, 1.0, 65536)
    );
    assert_eq!(c.max_position_embeddings, 1_048_576);
    assert_eq!((c.hc_mult, c.hc_sinkhorn_iters, c.hc_eps), (4, 20, 1e-6));
    assert_eq!(
        (c.index_n_heads, c.index_head_dim, c.index_topk),
        (32, 128, 512)
    );
    assert_eq!(c.kv_source_layer_ids, vec![2, 8, 14, 20]);
    assert_eq!(c.index_source_layer_ids, vec![2, 8, 14, 20, 24, 28, 32, 36]);
    assert_eq!(c.candidate_source_layer_id, Some(20));
    assert_eq!((c.candidate_topk_blocks, c.candidate_block_size), (2048, 8));
    assert_eq!(c.engram_layer_ids, vec![1, 14]);
    assert_eq!(c.engram_num_embeddings, vec![384006168, 384016682]);
    assert_eq!(
        (c.engram_max_ngram_size, c.engram_n_heads, c.engram_head_dim),
        (4, 8, 256)
    );
    assert_eq!(c.engram_pad_token_id, 2);
    assert_eq!(
        (c.engram_vocab_size, c.engram_compressed_vocab_size),
        (16_000_000, 99092)
    );
    // 2026-10-18: Derived at parse time (`parsers/deepseek_v41/engram.rs` holds the golden).
    assert_eq!(
        (
            c.engram_multipliers.len(),
            c.engram_primes.len(),
            c.engram_offsets.len()
        ),
        (8, 48, 48)
    );
    assert_eq!(c.engram_primes[0], 16_000_057);
    assert_eq!((c.num_mtp_modules, c.mtp_num_hidden_layers), (3, 3));
    assert_eq!(c.dspark_target_layer_ids, vec![37, 38, 39]);
    assert_eq!((c.bos_token_id, c.eos_token_id), (0, 1));
    assert!(c.is_eos(1));
    assert!(c.vision.is_none());
}

#[test]
fn publisher_quantization_declares_mxfp4_experts_and_32x32_fp8_elsewhere() {
    let c = parse_config(CONFIG).expect("parse V4.1 config.json");
    let plan = &c
        .quantization_config
        .expect("quantization_config")
        .precision;
    let expert = plan.resolve("layers.0.ffn.experts.17.w2");
    assert_eq!(expert.weight, Some(Operand::MXFP4));
    let dense = plan.resolve("layers.0.ffn.shared_experts.w2");
    assert_eq!(
        dense.weight.unwrap().granularity,
        Granularity::Block(32, 32)
    );
    for p in [expert, dense] {
        let a = p.activation.unwrap();
        assert!(a.is_fp8());
        assert_eq!(a.granularity, Granularity::Group(32));
    }
}

#[test]
fn each_required_text_field_is_refused_when_absent() {
    for key in [
        "head_dim",
        "o_groups",
        "compress_rope_theta",
        "rope_theta",
        "sliding_window",
        "candidate_source_layer_id",
        "kv_source_layer_ids",
        "engram_layer_ids",
        "swiglu_limit",
        "scoring_func",
        "rms_norm_eps",
    ] {
        let e = parse_config(&without(&["text_config", key])).expect_err(key);
        assert!(
            e.to_string()
                .contains(&format!("no `{key}`; refusing to guess")),
            "{key}: {e:#}"
        );
    }
    let e = parse_config(&without(&["text_config", "engram_pad_token_id"])).unwrap_err();
    assert!(e.to_string().contains("engram_pad_token_id"), "{e:#}");
    let e = parse_config(&without(&["text_config"])).unwrap_err();
    assert!(e.to_string().contains("no `text_config`"), "{e:#}");
    let e = parse_config(&without(&["bos_token_id"])).unwrap_err();
    assert!(e.to_string().contains("bos_token_id"), "{e:#}");
}

#[test]
fn unimplemented_or_inconsistent_values_are_refused() {
    let cases = [
        (
            with(&["text_config", "scoring_func"], "sigmoid".into()),
            "scoring_func",
        ),
        (
            with(&["text_config", "topk_method"], "greedy".into()),
            "topk_method",
        ),
        (
            with(&["text_config", "num_key_value_heads"], 2.into()),
            "one latent KV head",
        ),
        (
            with(&["text_config", "o_groups"], 7.into()),
            "multiple of o_groups",
        ),
        (
            with(
                &["text_config", "compress_ratios"],
                serde_json::json!([0, 4]),
            ),
            "fewer than num_hidden_layers",
        ),
        (
            with(&["text_config", "candidate_source_layer_id"], 40.into()),
            "candidate",
        ),
        (
            with(&["text_config", "engram_layer_ids"], serde_json::json!([1])),
            "engram_num_embeddings has 2 entries for 1",
        ),
        (
            with(
                &["text_config", "engram_num_embeddings"],
                serde_json::json!([384006168, 384016681]),
            ),
            "differ from engram_num_embeddings",
        ),
        (
            with(&["text_config", "engram_compressed_vocab_size"], 0.into()),
            "compressed_vocab_size must be > 0",
        ),
        (
            with(
                &["text_config", "rope_scaling", "rope_type"],
                "linear".into(),
            ),
            "not yarn",
        ),
        (
            with(&["quantization_config", "expert_dtype"], "int4".into()),
            "expert_dtype",
        ),
        (
            with(&["quantization_config", "scale_fmt"], "e4m3".into()),
            "scale_fmt",
        ),
    ];
    for (json, want) in cases {
        let e = parse_config(&json).expect_err(want);
        assert!(format!("{e:#}").contains(want), "{want}: {e:#}");
    }
}
