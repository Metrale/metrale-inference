// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Packed-int admission against the `quantization_config` of
//! `poolside/Laguna-XS-2.1-INT4` @ 4b7e28abdc0a8b121def816b89d631750bc53c92 (config.json
//! verbatim in fixtures/), and one refusal per unsupported variant, each a single-field
//! mutation of that block so a refusal cannot come from another field.

use serde_json::{Value, json};

use super::{PACKED_INT_GROUP_SIZE, PackedIntScheme, admit_packed_int};
use crate::precision_plan::{DeclaredPrecisionPlan, Granularity, NumKind, Operand, ScaleTiming};

fn laguna_qc() -> Value {
    let config: Value =
        serde_json::from_str(include_str!("fixtures/poolside_laguna_xs_2_1_int4.json"))
            .expect("fixture parses");
    config["quantization_config"].clone()
}

/// 2026-10-07: Apply `edit` to the real block and return the refusal text.
fn refusal(edit: impl FnOnce(&mut Value)) -> String {
    let mut qc = laguna_qc();
    edit(&mut qc);
    format!(
        "{:#}",
        admit_packed_int(&qc).expect_err("variant must be refused")
    )
}

fn g0_weights(qc: &mut Value) -> &mut Value {
    &mut qc["config_groups"]["group_0"]["weights"]
}

#[test]
fn laguna_int4_block_admits_int4_then_int8() {
    assert_eq!(
        admit_packed_int(&laguna_qc()).expect("admitted"),
        [PackedIntScheme::INT4_G128, PackedIntScheme::INT8_G128]
    );
}

#[test]
fn float_and_foreign_blocks_are_not_packed_int() {
    let nvfp4 = json!({
        "quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized",
        "config_groups": { "group_0": { "targets": ["Linear"],
            "weights": { "num_bits": 4, "type": "float", "group_size": 16 } } }
    });
    assert!(admit_packed_int(&nvfp4).unwrap().is_empty());
    assert!(
        admit_packed_int(&json!({"quant_method": "modelopt", "quant_algo": "NVFP4"}))
            .unwrap()
            .is_empty()
    );
    assert!(admit_packed_int(&json!({})).unwrap().is_empty());
}

/// 2026-10-07: The plan routes Laguna's modules exactly as compressed-tensors does: experts
/// of layers 1-30 INT4, 31-39 INT8; the dense layer 0, attention, the router, the shared
/// expert and lm_head unquantized (the shared expert matches group_0's pattern but is ignored).
#[test]
fn laguna_plan_targets_only_routed_experts() {
    let plan = DeclaredPrecisionPlan::from_quantization_config(&laguna_qc()).expect("plan");
    let scheme = |m: &str| {
        let p = plan.resolve(m);
        assert!(p.activation.is_none(), "{m} declares activations");
        p.weight
            .map(|w| PackedIntScheme::from_operand(&w).expect("admitted operand"))
    };
    for layer in 1..=39 {
        let want = if layer <= 30 {
            PackedIntScheme::INT4_G128
        } else {
            PackedIntScheme::INT8_G128
        };
        for e in [0, 17, 255] {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                let m = format!("model.layers.{layer}.mlp.experts.{e}.{proj}");
                assert_eq!(scheme(&m), Some(want), "{m}");
            }
        }
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let m = format!("model.layers.{layer}.mlp.shared_expert.{proj}");
            assert_eq!(scheme(&m), None, "{m}");
        }
        for m in ["mlp.gate", "self_attn.q_proj", "self_attn.k_proj"]
            .into_iter()
            .chain(["self_attn.v_proj", "self_attn.o_proj", "self_attn.g_proj"])
        {
            let m = format!("model.layers.{layer}.{m}");
            assert_eq!(scheme(&m), None, "{m}");
        }
    }
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        assert_eq!(scheme(&format!("model.layers.0.mlp.{proj}")), None);
    }
    assert_eq!(scheme("lm_head"), None);
}

#[test]
fn operand_mapping_admits_only_static_group_128_int4_int8() {
    let int = |bits, granularity, timing| Operand {
        kind: NumKind::Int,
        bits,
        granularity,
        timing,
    };
    let g = Granularity::Group(PACKED_INT_GROUP_SIZE);
    assert_eq!(
        PackedIntScheme::from_operand(&int(4, g, ScaleTiming::Static)),
        Some(PackedIntScheme::INT4_G128)
    );
    assert_eq!(
        PackedIntScheme::from_operand(&int(8, g, ScaleTiming::Static)),
        Some(PackedIntScheme::INT8_G128)
    );
    for refused in [
        int(2, g, ScaleTiming::Static),
        int(4, Granularity::Group(64), ScaleTiming::Static),
        int(4, Granularity::Channel, ScaleTiming::Static),
        int(4, g, ScaleTiming::Dynamic),
        Operand::NVFP4,
    ] {
        assert_eq!(PackedIntScheme::from_operand(&refused), None, "{refused:?}");
    }
    assert_eq!(PackedIntScheme::INT4_G128.codes_per_word(), 8);
    assert_eq!(PackedIntScheme::INT8_G128.codes_per_word(), 4);
    assert_eq!(PackedIntScheme::INT4_G128.code_offset(), 8);
    assert_eq!(PackedIntScheme::INT8_G128.code_offset(), 128);
}

#[test]
fn refuses_asymmetric_weights() {
    let e = refusal(|qc| g0_weights(qc)["symmetric"] = json!(false));
    assert!(e.contains("weights.symmetric false"), "{e}");
}

#[test]
fn refuses_zero_points() {
    let e = refusal(|qc| g0_weights(qc)["zp_dtype"] = json!("torch.int8"));
    assert!(e.contains("weights.zp_dtype"), "{e}");
}

#[test]
fn refuses_activation_ordering() {
    for order in [json!("group"), json!("weight"), json!(true)] {
        let e = refusal(|qc| g0_weights(qc)["actorder"] = order.clone());
        assert!(e.contains("weights.actorder"), "{order}: {e}");
    }
}

#[test]
fn refuses_other_group_sizes() {
    for size in [32, 64, 256] {
        let e = refusal(|qc| g0_weights(qc)["group_size"] = json!(size));
        assert!(
            e.contains(&format!("weights.group_size Some({size})")),
            "{e}"
        );
    }
    let e = refusal(|qc| g0_weights(qc)["group_size"] = Value::Null);
    assert!(e.contains("weights.group_size None"), "{e}");
}

#[test]
fn refuses_other_strategies() {
    for strategy in ["channel", "tensor", "tensor_group", "block"] {
        let e = refusal(|qc| g0_weights(qc)["strategy"] = json!(strategy));
        assert!(e.contains("weights.strategy"), "{strategy}: {e}");
    }
}

#[test]
fn refuses_other_widths() {
    for bits in [2, 3, 16] {
        let e = refusal(|qc| g0_weights(qc)["num_bits"] = json!(bits));
        assert!(e.contains("weights.num_bits"), "{bits}: {e}");
    }
}

#[test]
fn refuses_dynamic_block_and_scale_dtype_variants() {
    let e = refusal(|qc| g0_weights(qc)["dynamic"] = json!(true));
    assert!(e.contains("weights.dynamic"), "{e}");
    let e = refusal(|qc| g0_weights(qc)["block_structure"] = json!([128, 128]));
    assert!(e.contains("weights.block_structure"), "{e}");
    let e = refusal(|qc| g0_weights(qc)["scale_dtype"] = json!("torch.float16"));
    assert!(e.contains("weights.scale_dtype"), "{e}");
}

#[test]
fn refuses_quantized_activations() {
    let act = json!({"num_bits": 8, "type": "int", "strategy": "token", "dynamic": true});
    let e = refusal(|qc| qc["config_groups"]["group_0"]["input_activations"] = act.clone());
    assert!(e.contains("group_0.input_activations"), "{e}");
    let e = refusal(|qc| qc["config_groups"]["group_1"]["output_activations"] = act);
    assert!(e.contains("group_1.output_activations"), "{e}");
}

#[test]
fn refuses_other_formats() {
    for format in ["int-quantized", "float-quantized", "naive-quantized"] {
        let e = refusal(|qc| qc["config_groups"]["group_0"]["format"] = json!(format));
        assert!(e.contains("group_0.format"), "{format}: {e}");
    }
    // 2026-10-07: A group without its own format inherits the block's.
    let e = refusal(|qc| {
        qc["config_groups"]["group_1"]["format"] = Value::Null;
        qc["format"] = json!("int-quantized");
    });
    assert!(e.contains("group_1.format"), "{e}");
}

#[test]
fn refuses_mixed_integer_and_float_groups() {
    let e = refusal(|qc| qc["config_groups"]["group_1"]["weights"]["type"] = json!("float"));
    assert!(e.contains("mixes integer and float"), "{e}");
}

#[test]
fn refuses_uncompressed_status_and_sparsity() {
    let e = refusal(|qc| qc["quantization_status"] = json!("frozen"));
    assert!(e.contains("quantization_status"), "{e}");
    let e = refusal(|qc| qc["sparsity_config"] = json!({"format": "sparse-24-bitmask"}));
    assert!(e.contains("sparsity_config"), "{e}");
}

#[test]
fn refuses_online_transforms() {
    for location in ["input", "output", "k_cache", "q_attn"] {
        let e = refusal(|qc| {
            qc["transform_config"]["config_groups"]["R1"]["apply"][1]["location"] = json!(location);
        });
        assert!(e.contains("apply[1].location"), "{location}: {e}");
    }
}
