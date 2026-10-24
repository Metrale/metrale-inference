// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: Tests for the embedding-model parse and the Llama-3 RoPE table.
//!
//! Owner: config.
//! Invariants: none beyond the types.

use super::*;

const FIXTURE: &str = include_str!("tests/fixtures/llama_nemotron_embed_vl_1b_v2.config.json");

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).unwrap()
}

fn parse(v: &Value) -> Result<crate::ModelConfig> {
    crate::parse_config(&v.to_string())
}

/// 2026-10-24: The checkpoint's own config parses into the text tower's shape, with the
/// embedding semantics and the llama3 rope it declares.
#[test]
fn parses_the_checkpoint_text_tower() {
    let c = parse(&fixture()).unwrap();
    assert_eq!(c.model_type, "llama_nemotron_vl");
    assert_eq!(
        (c.hidden_size, c.num_hidden_layers, c.intermediate_size),
        (2048, 16, 8192)
    );
    assert_eq!(
        (c.num_attention_heads, c.num_key_value_heads, c.head_dim),
        (32, 8, 64)
    );
    assert_eq!(c.vocab_size, 128266);
    assert_eq!(c.bos_token_id, 128000);
    assert!(c.tie_word_embeddings);
    assert_eq!(c.rope_theta, 500000.0);
    assert!((c.rms_norm_eps - 1e-5).abs() < 1e-12);
    assert_eq!(c.weight_prefix, "language_model");
    assert!(c.vision.is_none());
    assert_eq!(
        c.rope_llama3,
        Some(Llama3Rope {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192.0,
        })
    );
    let e = c.embedding.clone().unwrap();
    assert_eq!(e.pooling, Pooling::Mean);
    assert_eq!(e.attention, AttentionMask::Bidirectional);
    assert_eq!((e.query_max_tokens, e.document_max_tokens), (512, 4096));
    assert!(!c.capabilities().generates);
}

/// 2026-10-24: A generative config keeps `generates` (the capability is derived, not declared).
#[test]
fn a_generative_config_generates() {
    let json = r#"{"model_type":"mistral_x","hidden_size":64,"num_hidden_layers":1,
        "num_attention_heads":1,"num_key_value_heads":1,"vocab_size":8}"#;
    let c = crate::parse_config(json).unwrap();
    assert!(c.embedding.is_none());
    assert!(c.capabilities().generates);
}

/// 2026-10-24: Every key that changes the math and is not mapped is refused, never ignored.
#[test]
fn unmapped_math_keys_are_refused() {
    let cases: [(&str, Value); 7] = [
        ("/llm_config/rope_scaling/rope_type", Value::from("yarn")),
        ("/llm_config/attention_bias", Value::from(true)),
        ("/llm_config/mlp_bias", Value::from(true)),
        ("/llm_config/hidden_act", Value::from("gelu")),
        (
            "/llm_config/architectures",
            serde_json::json!(["LlamaForCausalLM"]),
        ),
        ("/pooling", Value::from("last")),
        ("/llm_config/tie_word_embeddings", Value::from(false)),
    ];
    for (ptr, v) in cases {
        let mut raw = fixture();
        *raw.pointer_mut(ptr).unwrap() = v;
        assert!(parse(&raw).is_err(), "{ptr} must be refused");
    }
}

/// 2026-10-24: A missing declaration is an error, never a default.
#[test]
fn missing_declarations_are_refused() {
    for (parent, key) in [
        ("/llm_config/rope_scaling", "factor"),
        (
            "/llm_config/rope_scaling",
            "original_max_position_embeddings",
        ),
        ("", "q_max_length"),
        ("", "p_max_length"),
        ("/llm_config", "rope_scaling"),
        ("", "llm_config"),
    ] {
        let mut raw = fixture();
        let obj = if parent.is_empty() {
            &mut raw
        } else {
            raw.pointer_mut(parent).unwrap()
        };
        obj.as_object_mut().unwrap().remove(key);
        assert!(
            parse(&raw).is_err(),
            "{parent}/{key} missing must be refused"
        );
    }
}

/// 2026-10-24: The table's three bands: wavelengths below `orig / high_freq_factor` keep their
/// frequency, those above `orig / low_freq_factor` are divided by `factor`, and the band
/// between blends smoothly and stays between the two.
#[test]
fn llama3_table_bands() {
    let r = Llama3Rope {
        factor: 32.0,
        low_freq_factor: 1.0,
        high_freq_factor: 4.0,
        original_max_position_embeddings: 8192.0,
    };
    let (theta, dim) = (500000.0f64, 64usize);
    let t = r.inv_freq(theta, dim);
    assert_eq!(t.len(), 32);
    let (mut high, mut low, mut mid) = (0, 0, 0);
    for (i, &got) in t.iter().enumerate() {
        let f = 1.0 / theta.powf((2 * i) as f64 / dim as f64);
        let wl = 2.0 * std::f64::consts::PI / f;
        if wl < 2048.0 {
            assert_eq!(got, f as f32, "pair {i} must keep its frequency");
            high += 1;
        } else if wl > 8192.0 {
            assert_eq!(got, (f / 32.0) as f32, "pair {i} must be divided by factor");
            low += 1;
        } else {
            let g = f64::from(got);
            assert!(g > f / 32.0 && g < f, "pair {i} must blend: {g} vs {f}");
            mid += 1;
        }
    }
    assert!(high > 0 && low > 0 && mid > 0, "{high}/{mid}/{low}");
    // 2026-10-24: Strictly decreasing, as plain RoPE: no band boundary may reorder pairs.
    assert!(t.windows(2).all(|w| w[1] < w[0]));
}

/// 2026-10-24: The prompts file needs both roles.
#[test]
fn prompts_need_both_roles() {
    let ok = parse_sentence_transformers_prompts(
        r#"{"prompts":{"document":"passage: ","query":"query: "}}"#,
    )
    .unwrap();
    assert_eq!(ok.document, "passage: ");
    assert_eq!(ok.query, "query: ");
    assert!(parse_sentence_transformers_prompts(r#"{"prompts":{"query":"q"}}"#).is_err());
    assert!(parse_sentence_transformers_prompts(r#"{"default_prompt_name":"x"}"#).is_err());
}

/// 2026-10-24: `role` refuses until the prompts are set, then gives each role its own limit.
#[test]
fn role_reads_prompt_and_limit() {
    let mut e = parse(&fixture()).unwrap().embedding.unwrap();
    assert!(e.role(EmbedRole::Query).is_err());
    e.prompts = Some(EmbeddingPrompts {
        query: "query: ".into(),
        document: "passage: ".into(),
    });
    assert_eq!(e.role(EmbedRole::Query).unwrap(), ("query: ", 512));
    assert_eq!(e.role(EmbedRole::Document).unwrap(), ("passage: ", 4096));
}

/// 2026-10-24: An `input_type` names a role only when that role's prompt is `"<input_type>: "`,
/// so the echoed prefix is the applied one: `document` is refused for a checkpoint whose
/// document prompt is `passage: `, and follows the prompts when they change.
#[test]
fn input_types_come_from_the_prompts() {
    let mut e = parse(&fixture()).unwrap().embedding.unwrap();
    assert!(e.role_for_input_type("query").is_err());
    e.prompts = Some(EmbeddingPrompts {
        query: "query: ".into(),
        document: "passage: ".into(),
    });
    assert_eq!(
        e.role_for_input_type("query").unwrap(),
        Some(EmbedRole::Query)
    );
    assert_eq!(
        e.role_for_input_type("passage").unwrap(),
        Some(EmbedRole::Document)
    );
    for refused in ["document", "search", "passage:", "", "query: "] {
        assert_eq!(e.role_for_input_type(refused).unwrap(), None, "{refused:?}");
    }
    assert_eq!(e.input_types().unwrap(), vec!["query", "passage"]);
    e.prompts = Some(EmbeddingPrompts {
        query: "Represent the query: ".into(),
        document: "document: ".into(),
    });
    assert_eq!(
        e.role_for_input_type("document").unwrap(),
        Some(EmbedRole::Document)
    );
    assert_eq!(e.role_for_input_type("query").unwrap(), None);
    assert_eq!(
        e.input_types().unwrap(),
        vec!["Represent the query", "document"]
    );
}
