// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `PolicyPrecision` over a small compressed-tensors plan: FP8 W8A8 attention,
//! NVFP4 W4A4 MLP, one ignored module, under both tiers and with and without the W8A8 caps.
//!
//! Owner: metrale-circuit.
//! Invariants: none beyond the types.

use super::*;

const QC: &str = r#"{
  "quant_method": "compressed-tensors",
  "format": "mixed-precision",
  "ignore": ["model.layers.0.self_attn.o_proj"],
  "config_groups": {
    "group_0": {
      "format": "float-quantized",
      "targets": ["re:.*self_attn\\.(q|k|v|o)_proj$"],
      "weights": {"num_bits": 8, "type": "float", "strategy": "channel", "dynamic": false, "symmetric": true},
      "input_activations": {"num_bits": 8, "type": "float", "strategy": "token", "dynamic": true, "symmetric": true}
    },
    "group_1": {
      "format": "nvfp4-pack-quantized",
      "targets": ["re:.*mlp\\.(gate|up|down)_proj$"],
      "weights": {"num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16, "dynamic": false, "symmetric": true},
      "input_activations": {"num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16, "dynamic": "local", "symmetric": true}
    }
  }
}"#;

fn plan() -> DeclaredPrecisionPlan {
    DeclaredPrecisionPlan::from_quantization_config(&serde_json::from_str(QC).unwrap()).unwrap()
}

fn f(s: &str) -> Format {
    Format::parse(s).unwrap()
}

fn formats(w: &str, a: &str) -> LinearFormats {
    LinearFormats {
        weight: f(w),
        activation: f(a),
    }
}

fn ask(
    tier: &str,
    caps: &[&str],
    engine: &[(String, LinearFormats)],
    module: &str,
) -> LinearFormats {
    let plan = plan();
    let caps: Vec<String> = caps.iter().map(|c| c.to_string()).collect();
    let policy =
        WeightQuantPolicy::new(tier_named(tier).unwrap(), &plan, caps_named(&caps).unwrap());
    PolicyPrecision::new(policy, engine).linear(module)
}

const Q: &str = "model.layers.3.self_attn.q_proj";
const MLP: &str = "model.layers.3.mlp.down_proj";
const IGNORED: &str = "model.layers.0.self_attn.o_proj";
const PLAIN: &str = "model.layers.3.linear_attn.in_proj_b";

#[test]
fn the_nvfp4_tier_serves_every_quantized_module_w4a16() {
    for m in [Q, MLP] {
        assert_eq!(
            ask("nvfp4", &[], &[], m),
            formats("nvfp4/g16", "bf16"),
            "{m}"
        );
    }
    for m in [IGNORED, PLAIN] {
        assert_eq!(ask("nvfp4", &[], &[], m), formats("bf16", "bf16"), "{m}");
    }
}

#[test]
fn the_declared_tier_follows_the_checkpoint_up_to_the_kernels_present() {
    assert_eq!(ask("declared", &[], &[], Q), formats("fp8/channel", "bf16"));
    assert_eq!(
        ask("declared", &["w8a8_decode"], &[], Q),
        formats("fp8/channel", "fp8/token")
    );
    assert_eq!(
        ask("declared", &[], &[], MLP),
        formats("nvfp4/g16", "nvfp4/g16")
    );
    assert_eq!(
        ask("declared", &["w8a8_decode"], &[], IGNORED),
        formats("bf16", "bf16")
    );
}

#[test]
fn engine_formats_answer_first_and_in_order() {
    let engine = vec![
        ("*.self_attn.q_proj".to_string(), formats("bf16", "bf16")),
        (
            "*.self_attn.*".to_string(),
            formats("nvfp4/g16", "nvfp4/g16"),
        ),
    ];
    assert_eq!(
        ask("declared", &["w8a8_decode"], &engine, Q),
        formats("bf16", "bf16")
    );
    assert_eq!(
        ask("declared", &[], &engine, "model.layers.3.self_attn.k_proj"),
        formats("nvfp4/g16", "nvfp4/g16")
    );
    assert_eq!(
        ask("declared", &[], &engine, MLP),
        formats("nvfp4/g16", "nvfp4/g16")
    );
}

#[test]
fn unknown_names_and_malformed_fixtures_are_refused() {
    assert!(tier_named("fp8").is_err());
    assert!(caps_named(&["w9a9".to_string()]).is_err());
    assert!(
        CheckpointPlan::parse("schema = 2\ncheckpoint = \"c\"\nquantization_config = \"{}\"")
            .is_err()
    );
    assert!(
        CheckpointPlan::parse("schema = 1\ncheckpoint = \"c\"\nquantization_config = \"{\"")
            .is_err()
    );
    let ok = format!("schema = 1\ncheckpoint = \"c\"\nquantization_config = '''{QC}'''");
    assert_eq!(CheckpointPlan::parse(&ok).unwrap().checkpoint, "c");
}
