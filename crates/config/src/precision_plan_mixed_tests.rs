// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The declared precision plan against a ModelOpt `MIXED_PRECISION` block: the
//! fused-experts container fallback and the scale-encoding check.

use super::*;

/// 2026-10-17: GLM-5.3-Flash FP4-Local (ModelOpt `MIXED_PRECISION`, `quantized_layers` only;
/// fixture trimmed to the head, layers 0/3/5/45 and part of the vision tower): the fused
/// `...mlp.experts` entry declares every per-expert projection the loaders query (NVFP4 W4A4 on
/// the text layers, W4A16 on the MTP layer 45); the MXFP8 and block-128 FP8 entries are
/// algorithms the plan does not know, so those modules stay 16-bit. The pattern is anchored to
/// its own layer: layer 35 is absent from the fixture and resolves unquantized.
#[test]
fn fp4local_fused_expert_entries_declare_their_projections() {
    let p = plan("fp4local");
    assert_eq!(p.source, PlanSource::ModelOpt);
    for m in [
        format!("{L}.3.mlp.experts"),
        format!("{L}.3.mlp.experts.0.gate_proj"),
        format!("{L}.5.mlp.experts.287.down_proj"),
    ] {
        let got = p.resolve(&m);
        assert_eq!(got.label(), "W4A4", "{m}");
        let a = got.activation.expect("activation");
        assert!(a.is_fp4(), "{m}");
        assert_eq!(a.granularity, Granularity::TensorGroup(16), "{m}");
    }
    for m in [
        format!("{L}.45.mlp.experts.0.gate_proj"),
        "model.visual.merger.up_proj".to_string(),
    ] {
        assert_eq!(p.resolve(&m).label(), "W4A16", "{m}");
    }
    for m in [
        "lm_head".to_string(),
        format!("{L}.0.mlp.gate_proj"),
        format!("{L}.5.self_attn.q_proj"),
        format!("{L}.3.self_attn.o_proj"),
        format!("{L}.3.mlp.shared_experts.gate_proj"),
        format!("{L}.45.self_attn.q_a_proj"),
        format!("{L}.3.mlp.gate"),
        format!("{L}.35.mlp.experts.0.gate_proj"),
        format!("{L}.30.mlp.experts.0.gate_proj"),
    ] {
        assert_eq!(p.resolve(&m), LayerPrecision::UNQUANTIZED, "{m}");
    }
}

/// 2026-10-17: An expert projection no rule names takes its fused-experts entry; an exact entry
/// for one projection outranks the container, an excluded projection stays 16-bit, and a name
/// that is not an expert projection does not reach the container.
#[test]
fn an_expert_projection_falls_back_to_its_container_entry() {
    let qc = serde_json::json!({
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            "m.experts": {"quant_algo": "NVFP4", "group_size": 16},
            "m.experts.0.gate_proj": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
        },
        "exclude_modules": ["m.experts.2.up_proj"],
    });
    let p = DeclaredPrecisionPlan::from_quantization_config(&qc).expect("plan");
    assert_eq!(p.resolve("m.experts.0.gate_proj").label(), "W4A16");
    assert_eq!(p.resolve("m.experts.1.gate_proj").label(), "W4A4");
    assert_eq!(
        p.resolve("m.experts.2.up_proj"),
        LayerPrecision::UNQUANTIZED
    );
    assert_eq!(p.resolve("m.experts.2.down_proj").label(), "W4A4");
    for none in [
        "m.expertsX",
        "m.experts.x.gate_proj",
        "m.shared_expert.gate_proj",
    ] {
        assert_eq!(p.resolve(none), LayerPrecision::UNQUANTIZED, "{none}");
    }
}

/// 2026-10-17: `weight_scale_encoding` must name the one encoding the loaders decode; another
/// name or a non-string is refused with the entry named, never read as raw scales.
#[test]
fn an_unknown_scale_encoding_is_refused() {
    let entry = format!("{L}.3.mlp.experts");
    for bad in [serde_json::json!("csf_64x1_v9"), serde_json::json!(7)] {
        let mut raw = fixture("fp4local");
        raw["quantized_layers"][&entry]["weight_scale_encoding"] = bad.clone();
        let err = DeclaredPrecisionPlan::from_quantization_config(&raw).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains(&entry) && msg.contains("weight_scale_encoding"),
            "{bad}: {msg}"
        );
    }
    // 2026-10-17: Positive control: the fixture's own encoding and an absent one both parse.
    let mut raw = fixture("fp4local");
    assert!(DeclaredPrecisionPlan::from_quantization_config(&raw).is_ok());
    raw["quantized_layers"][&entry]
        .as_object_mut()
        .expect("entry")
        .remove("weight_scale_encoding");
    assert!(DeclaredPrecisionPlan::from_quantization_config(&raw).is_ok());
}
