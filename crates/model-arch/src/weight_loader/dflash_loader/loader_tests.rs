// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Parser tests for the DFlash drafter's `config.json`.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use super::*;

const SHIPPED_CONFIG: &str = r#"{
    "hidden_size": 2048,
    "num_hidden_layers": 8,
    "intermediate_size": 6144,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "vocab_size": 248320,
    "draft_vocab_size": 248320,
    "tie_word_embeddings": false,
    "block_size": 16,
    "rope_theta": 10000000.0,
    "rope_scaling": null,
    "rms_norm_eps": 1e-6,
    "dflash_config": {
        "mask_token_id": 248070,
        "target_layer_ids": [1, 10, 19, 28, 37]
    }
}"#;

#[test]
fn shipped_qwen3_6_fields_reach_the_runtime_config() {
    let config = parse_dflash_config(SHIPPED_CONFIG).expect("parse drafter config");
    assert_eq!(config.num_hidden_layers, 8);
    assert_eq!(config.hidden_size, 2048);
    assert_eq!(config.intermediate_size, 6144);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_key_value_heads, 4);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.vocab_size, 248320);
    assert_eq!(config.draft_vocab_size, Some(248320));
    assert!(!config.tie_word_embeddings);
    assert_eq!(config.block_size, 16);
    assert_eq!(config.rope_theta, Some(10_000_000.0));
    assert_eq!(config.effective_rope_theta(), Ok(10_000_000.0));
    assert!(config.rope_scaling.is_none());
    assert_eq!(config.effective_rms_norm_eps(), Ok(1e-6));
    let sub = config.dflash_config.expect("dflash_config present");
    assert_eq!(sub.mask_token_id, 248070);
    assert_eq!(sub.target_layer_ids, vec![1, 10, 19, 28, 37]);
}

#[test]
fn omitted_optional_fields_use_runtime_defaults() {
    let config = parse_dflash_config(
        r#"{
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "intermediate_size": 128,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "vocab_size": 256
        }"#,
    )
    .unwrap();

    assert_eq!(config.block_size, 16);
    assert_eq!(config.rope_theta, None);
    assert_eq!(config.effective_rope_theta(), Ok(10_000_000.0));
    assert!(!config.tie_word_embeddings);
    assert!(config.draft_vocab_size.is_none());
    assert!(config.dflash_config.is_none());
    assert!(config.rope_scaling.is_none());
    assert!(
        config.effective_rms_norm_eps().is_err(),
        "a drafter that does not state its norm epsilon is refused, not given 1e-6"
    );
}

#[test]
fn malformed_runtime_field_is_rejected_with_parser_context() {
    let malformed =
        SHIPPED_CONFIG.replace("\"mask_token_id\": 248070", "\"mask_token_id\": \"bad\"");
    let error = parse_dflash_config(&malformed).unwrap_err();
    assert_eq!(error.to_string(), "Parsing DFlash drafter config.json");
    assert!(format!("{error:#}").contains("invalid type: string \"bad\""));
}

/// 2026-09-25: With `block_size` only inside `dflash_config`, the top-level
/// field takes its default of 16 and `effective_block_size` returns the
/// sub-config's 8.
#[test]
fn effective_block_size_prefers_the_drafters_own_value() {
    let json = r#"{
        "hidden_size": 5120, "num_hidden_layers": 5,
        "num_attention_heads": 32, "num_key_value_heads": 8,
        "intermediate_size": 17408, "vocab_size": 248320, "head_dim": 128,
        "dflash_config": {
            "block_size": 8, "mask_token_id": 248070,
            "target_layer_ids": [1, 10, 19, 28, 37]
        }
    }"#;
    let cfg = parse_dflash_config(json).expect("parses");
    assert_eq!(cfg.block_size, 16, "top-level default is still 16");
    assert_eq!(
        cfg.effective_block_size(),
        8,
        "the drafter's own block_size must win over the top-level default"
    );
}

/// 2026-09-25: With `block_size` only at the top level,
/// `effective_block_size` returns it.
#[test]
fn effective_block_size_falls_back_to_the_top_level() {
    let json = r#"{
        "hidden_size": 2048, "num_hidden_layers": 8,
        "num_attention_heads": 32, "num_key_value_heads": 4,
        "intermediate_size": 6144, "vocab_size": 248320, "head_dim": 128,
        "block_size": 16
    }"#;
    let cfg = parse_dflash_config(json).expect("parses");
    assert_eq!(cfg.effective_block_size(), 16);
}

/// 2026-10-08: A transformers-5 drafter config with θ only inside `rope_parameters` (the GLM-5.3
/// Flash DFlash2 drafter: 10,000). Before this was read, the head silently used 10,000,000.
const NESTED_THETA_CONFIG: &str = r#"{
    "hidden_size": 64, "num_hidden_layers": 1, "intermediate_size": 128,
    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32, "vocab_size": 256,
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}, "rms_norm_eps": 1e-05
}"#;

#[test]
fn rope_theta_inside_rope_parameters_is_used() {
    let config = parse_dflash_config(NESTED_THETA_CONFIG).unwrap();
    assert_eq!(config.rope_theta, None);
    assert_eq!(config.effective_rope_theta(), Ok(10_000.0));
    // 2026-10-08: The GLM-5.3 Flash DFlash2 drafter's 1e-5, not the 1e-6 every head used.
    assert_eq!(config.effective_rms_norm_eps(), Ok(1e-5));
}

#[test]
fn rope_theta_stated_twice_with_different_values_is_refused() {
    let config = parse_dflash_config(&NESTED_THETA_CONFIG.replacen(
        "\"rope_parameters\"",
        "\"rope_theta\": 500000.0, \"rope_parameters\"",
        1,
    ))
    .unwrap();
    assert!(config.effective_rope_theta().is_err());
}

#[test]
fn rope_theta_stated_twice_with_the_same_value_is_accepted() {
    let config = parse_dflash_config(&NESTED_THETA_CONFIG.replacen(
        "\"rope_parameters\"",
        "\"rope_theta\": 10000.0, \"rope_parameters\"",
        1,
    ))
    .unwrap();
    assert_eq!(config.effective_rope_theta(), Ok(10_000.0));
}

/// 2026-10-09: The trained window is read, an off switch hides it, and only a different
/// serve window draws the warning.
#[test]
fn a_serve_window_other_than_the_trained_one_is_reported() {
    let with = |extra: &str| {
        parse_dflash_config(&format!(
            r#"{{"hidden_size": 64, "num_hidden_layers": 1, "intermediate_size": 128,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
                "vocab_size": 256{extra}}}"#
        ))
        .unwrap()
    };
    let glm = with(r#", "sliding_window": 2048, "use_sliding_window": true"#);
    assert_eq!(glm.trained_window(), Some(2048));
    assert_eq!(glm.window_mismatch(2048), None);
    assert!(
        glm.window_mismatch(4096)
            .unwrap()
            .contains("--dflash-window-size 2048")
    );
    assert!(glm.window_mismatch(0).is_some());
    let off = with(r#", "sliding_window": 2048, "use_sliding_window": false"#);
    assert_eq!(off.trained_window(), None);
    assert_eq!(off.window_mismatch(4096), None);
    assert_eq!(with("").window_mismatch(4096), None);
    assert_eq!(glm.attention_window(Some(2048)), Some(2048));
    assert_eq!(
        glm.attention_window(Some(4096)),
        None,
        "a differing flag keeps no window"
    );
    assert_eq!(glm.attention_window(None), None);
    assert_eq!(off.attention_window(Some(2048)), None);
}
