// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The declared formats: FP8 channel/block/tensor weights with token or static
//! tensor activations, NVFP4 W4A4 and W4A16, 16-bit when undeclared, and a refusal for a
//! format the circuit has none for.

use metrale_config::DeclaredPrecisionPlan;

use super::DeclaredPrecision;
use crate::format::{Format, Scale};
use crate::precision::EdgePrecision;

fn plan(qc: serde_json::Value) -> DeclaredPrecisionPlan {
    DeclaredPrecisionPlan::from_quantization_config(&qc).unwrap()
}

fn group(
    weights: serde_json::Value,
    acts: serde_json::Value,
    targets: &[&str],
) -> serde_json::Value {
    serde_json::json!({
        "quant_method": "compressed-tensors",
        "config_groups": { "group_0": { "weights": weights, "input_activations": acts, "targets": targets } },
        "ignore": ["lm_head"],
    })
}

#[test]
fn fp8_channel_weights_with_dynamic_token_activations_are_w8a8() {
    let p = plan(group(
        serde_json::json!({"num_bits": 8, "type": "float", "strategy": "channel", "dynamic": false}),
        serde_json::json!({"num_bits": 8, "type": "float", "strategy": "token", "dynamic": true}),
        &["Linear"],
    ));
    let d = DeclaredPrecision::new(&p);
    let f = d.linear("model.layers.0.self_attn.q_proj");
    assert_eq!(
        f.weight,
        Format::Fp8E4m3 {
            scale: Scale::PerChannel
        }
    );
    assert_eq!(
        f.activation,
        Format::Fp8E4m3 {
            scale: Scale::PerToken
        }
    );
    let head = d.linear("lm_head");
    assert_eq!((head.weight, head.activation), (Format::Bf16, Format::Bf16));
    assert!(d.refusals().is_empty());
}

#[test]
fn nvfp4_w4a4_and_an_integer_scheme_is_refused() {
    let p = plan(group(
        serde_json::json!({"num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16}),
        serde_json::json!({"num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16, "dynamic": "local"}),
        &["Linear"],
    ));
    let d = DeclaredPrecision::new(&p);
    let f = d.linear("model.layers.3.mlp.down_proj");
    assert_eq!(
        (f.weight, f.activation),
        (Format::Nvfp4 { group: 16 }, Format::Nvfp4 { group: 16 })
    );
    let int = plan(group(
        serde_json::json!({"num_bits": 8, "type": "int", "strategy": "channel"}),
        serde_json::Value::Null,
        &["Linear"],
    ));
    let d = DeclaredPrecision::new(&int);
    let _ = d.linear("model.layers.0.mlp.up_proj");
    assert_eq!(d.refusals().len(), 1, "{:?}", d.refusals());
}

/// 2026-10-02: ModelOpt names the fused routed-experts module; its per-expert projections
/// inherit it unless they are ignored, and no other undeclared module inherits anything.
#[test]
fn routed_expert_projections_inherit_the_fused_experts_module() {
    let fp4 = serde_json::json!({"num_bits": 4, "type": "float", "group_size": 16});
    let p = plan(serde_json::json!({
        "quant_method": "modelopt",
        "config_groups": { "group_0": {
            "weights": fp4, "input_activations": fp4,
            "targets": ["model.layers.0.mlp.experts", "model.layers.1.mlp.experts"],
        } },
        "quantized_layers": {
            "model.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
            "model.layers.1.mlp.experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
        },
        "ignore": ["model.layers.1.mlp.experts.0.down_proj"],
    }));
    let d = DeclaredPrecision::new(&p);
    let up = d.linear("model.layers.0.mlp.experts.0.gate_proj");
    assert_eq!(
        (up.weight, up.activation),
        (Format::Nvfp4 { group: 16 }, Format::Bf16)
    );
    assert_eq!(
        d.linear("model.layers.1.mlp.experts.0.down_proj").weight,
        Format::Bf16,
        "an ignored projection stays 16-bit"
    );
    assert_eq!(
        d.linear("model.layers.0.mlp.shared_expert.gate_proj")
            .weight,
        Format::Bf16,
        "only a routed expert's projection inherits"
    );
}
