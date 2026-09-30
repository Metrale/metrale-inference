// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Tests of the attention positional-encoding resolution: Nemotron-H declares
//! none, other families keep RoPE, and a contradictory config is refused.
//!
//! Owner: config.
//! Invariants: none beyond the types.

use super::*;
use crate::parse_config;

/// 2026-09-29: A two-layer Nemotron-H config (one Mamba-2 layer, one attention layer) with
/// the checkpoint's inert rope keys; `extra` adds or overrides keys.
fn nemotron_h(model_type: &str, extra: serde_json::Value) -> String {
    let mut v = serde_json::json!({
        "model_type": model_type,
        "hidden_size": 2688,
        "num_hidden_layers": 2,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "vocab_size": 131072,
        "hybrid_override_pattern": "M*",
        "mamba_num_heads": 64,
        "mamba_head_dim": 64,
        "ssm_state_size": 128,
        "n_groups": 8,
        "conv_kernel": 4,
        "rope_theta": 10000,
        "partial_rotary_factor": 1.0
    });
    for (k, x) in extra.as_object().unwrap() {
        v[k] = x.clone();
    }
    v.to_string()
}

#[test]
fn nemotron_h_attention_has_no_position_encoding() {
    let cfg = parse_config(&nemotron_h("nemotron_h", serde_json::json!({}))).unwrap();
    assert_eq!(
        cfg.attn_position_encoding().unwrap(),
        AttnPositionEncoding::None
    );
    assert_eq!(cfg.rotary_dim(), 0, "no dims are rotated");
}

/// 2026-09-29: Path B: a RoPE family through the same entry point keeps RoPE and its
/// rotary dims (Qwen3-Next: head_dim 256 x partial_rotary_factor 0.25 = 64).
#[test]
fn a_rope_family_keeps_rope_and_its_rotary_dims() {
    let json = serde_json::json!({
        "model_type": "qwen3_next",
        "hidden_size": 2048,
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "partial_rotary_factor": 0.25,
        "vocab_size": 151936
    })
    .to_string();
    let cfg = parse_config(&json).unwrap();
    assert_eq!(
        cfg.attn_position_encoding().unwrap(),
        AttnPositionEncoding::Rope
    );
    assert_eq!(cfg.rotary_dim(), 64);
    assert!(
        !cfg.has_mamba2_layers(),
        "no mamba_* heads, so no Mamba-2 layers"
    );
}

/// 2026-09-29: Path C: a Nemotron-H config that also sets a RoPE variant parameter is
/// refused, and the error names the parameter.
#[test]
fn no_position_encoding_with_rope_parameters_is_refused() {
    for (extra, named) in [
        (serde_json::json!({"rotary_dim": 64}), "rotary_dim"),
        (serde_json::json!({"yarn_factor": 4.0}), "yarn_factor"),
    ] {
        let err = parse_config(&nemotron_h("nemotron_h", extra.clone()))
            .expect_err(&format!("{extra} must be refused"))
            .to_string();
        assert!(err.contains("no positional encoding"), "{err}");
        assert!(err.contains(named), "{err}");
    }
}

/// 2026-09-29: Path C: a config that bypassed `parse_config` has no resolved encoding, and
/// reading it is an error rather than a guess.
#[test]
fn an_unresolved_config_is_an_error() {
    let cfg: ModelConfig = serde_json::from_str(r#"{"hidden_size": 64}"#).unwrap();
    let err = cfg.attn_position_encoding().unwrap_err().to_string();
    assert!(err.contains("never resolved"), "{err}");
}

/// 2026-09-29: Path C: MRoPE fields are set by parsers, not JSON, so the refusal is checked on
/// the resolver directly; the same config without them resolves.
#[test]
fn no_position_encoding_with_mrope_is_refused() {
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    cfg.attn_position_encoding = Some(AttnPositionEncoding::None);
    resolve_attn_position_encoding(&mut cfg).unwrap();
    cfg.mrope_section = [16, 24, 24];
    let err = resolve_attn_position_encoding(&mut cfg)
        .unwrap_err()
        .to_string();
    assert!(err.contains("mrope_section"), "{err}");
    cfg.mrope_section = [0, 0, 0];
    cfg.mrope_interleaved = true;
    let err = resolve_attn_position_encoding(&mut cfg)
        .unwrap_err()
        .to_string();
    assert!(err.contains("mrope_interleaved"), "{err}");
}
