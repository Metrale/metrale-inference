// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Whole plans on the three toy layouts (Path A), their determinism and the
//! independence of a layer's bytes from which other layers are kept, and the refusals (Path B).
//!
//! Owner: metrale-ml-utils.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use serde_json::Value;

use super::*;
use crate::index::TensorEntry;
use crate::testkit;

fn plan(config: &str, side: Option<&str>, index: &TensorIndex, spec: &MockSpec) -> MockPlan {
    plan_mock(&MockInputs {
        source_id: "toy/model",
        revision: Some("abc"),
        config_json: config,
        hf_quant_config: side,
        index,
        spec,
        routing: None,
    })
    .expect("plan")
}

fn all_bytes(p: &MockPlan) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    for u in 0..p.units.len() {
        for (t, b) in synthesize(p, u).expect("synthesize") {
            out.insert(p.tensors[t].name.clone(), b);
        }
    }
    out
}

fn text_config(p: &MockPlan) -> Value {
    let c: Value = serde_json::from_str(&p.config_json).unwrap();
    c["text_config"].clone()
}

#[test]
fn the_fp8_moe_keeps_one_period_and_renumbers_nothing() {
    let (config, index) = testkit::moe_fp8();
    let p = plan(
        &config,
        None,
        &index,
        &testkit::spec("1", "mode = \"uniform\""),
    );
    assert_eq!(p.selection.signatures.len(), 1);
    assert_eq!(p.selection.kept_layers(), vec![0, 1, 2, 3]);
    let t = text_config(&p);
    assert_eq!(t["num_hidden_layers"], 4);
    assert_eq!(t["layer_types"].as_array().unwrap().len(), 4);
    assert_eq!(t["layer_types"][3], "full_attention");
    assert!(
        p.tensors
            .iter()
            .all(|o| !o.name.contains("layers.4.") && !o.name.contains("layers.7."))
    );
    assert!(
        p.tensors
            .iter()
            .any(|o| o.name == "mtp.layers.0.mlp.gate.weight")
    );
    let kept = index
        .iter()
        .filter(|e| {
            !(4..8).any(|l| {
                e.name
                    .starts_with(&format!("model.language_model.layers.{l}."))
            })
        })
        .count();
    assert_eq!(p.tensors.len(), kept, "every kept tensor appears once");
    let bytes = all_bytes(&p);
    assert_eq!(bytes.len(), p.tensors.len());
}

#[test]
fn the_dense_ct_mock_drops_the_middle_period_and_pins_the_renumbered_fp8_layers() {
    let (config, index) = testkit::dense_ct();
    let p = plan(
        &config,
        None,
        &index,
        &testkit::spec("1", "mode = \"uniform\""),
    );
    assert_eq!(
        p.selection.signatures.len(),
        2,
        "NVFP4-FFN and FP8-FFN periods"
    );
    assert_eq!(p.selection.kept_layers(), vec![0, 1, 2, 3, 8, 9, 10, 11]);
    assert_eq!(
        p.pinned, 12,
        "4 layers x gate/up/down renumbered 8..11 -> 4..7"
    );
    let names: Vec<&str> = p.tensors.iter().map(|o| o.name.as_str()).collect();
    assert!(
        names.contains(&"model.language_model.layers.4.mlp.gate_proj.weight"),
        "FP8 layer 8 is now 4"
    );
    assert!(names.contains(&"model.language_model.layers.0.mlp.gate_proj.weight_packed"));
    let src = metrale_config::DeclaredPrecisionPlan::from_quantization_config(
        &serde_json::from_str::<Value>(&config).unwrap()["quantization_config"],
    )
    .unwrap();
    let mock_qc =
        serde_json::from_str::<Value>(&p.config_json).unwrap()["quantization_config"].clone();
    let mock = metrale_config::DeclaredPrecisionPlan::from_quantization_config(&mock_qc).unwrap();
    for (from, to) in [(8, 4), (11, 7), (0, 0), (3, 3)] {
        let s = src.resolve(&format!("model.language_model.layers.{from}.mlp.down_proj"));
        let m = mock.resolve(&format!("model.language_model.layers.{to}.mlp.down_proj"));
        assert_eq!(s, m, "layer {from} -> {to}");
    }
    assert_eq!(all_bytes(&p).len(), p.tensors.len());
}

#[test]
fn the_modelopt_moe_renames_quantized_layer_keys_in_both_files() {
    let (config, side, index) = testkit::moe_nvfp4();
    let p = plan(
        &config,
        Some(&side),
        &index,
        &testkit::spec("1", "mode = \"uniform\""),
    );
    let side_out: Value = serde_json::from_str(p.hf_quant_config.as_deref().unwrap()).unwrap();
    let ql = side_out["quantization"]["quantized_layers"]
        .as_object()
        .unwrap();
    assert!(
        ql.keys()
            .all(|k| !k.contains("layers.4.") && !k.contains("layers.7."))
    );
    assert!(ql.contains_key("model.language_model.layers.3.mlp.experts"));
    assert_eq!(p.pinned, 0);
    assert_eq!(all_bytes(&p).len(), p.tensors.len());
}

#[test]
fn plans_are_deterministic_and_a_layer_has_the_same_bytes_in_every_mock() {
    let (config, index) = testkit::dense_ct();
    let one = plan(
        &config,
        None,
        &index,
        &testkit::spec("1", "mode = \"uniform\""),
    );
    let again = plan(
        &config,
        None,
        &index,
        &testkit::spec("1", "mode = \"uniform\""),
    );
    assert_eq!(one.digest, again.digest);
    assert_eq!(all_bytes(&one), all_bytes(&again));
    let wide = plan(
        &config,
        None,
        &index,
        &testkit::spec("[2, 1]", "mode = \"uniform\""),
    );
    assert_eq!(wide.selection.kept_layers(), (0..12).collect::<Vec<_>>());
    assert_ne!(one.digest, wide.digest);
    let (a, b) = (all_bytes(&one), all_bytes(&wide));
    // 2026-10-03: Source layer 8 is mock layer 4 in `one` and layer 8 in `wide`.
    let n = |l: usize| format!("model.language_model.layers.{l}.mlp.down_proj.weight");
    let np = |l: usize| format!("model.language_model.layers.{l}.mlp.down_proj.weight_packed");
    assert_eq!(a[&n(4)], b[&n(8)]);
    assert_eq!(a[&np(0)], b[&np(0)]);
    assert_ne!(
        a[&np(0)],
        a[&np(1)],
        "different tensors draw different values"
    );
}

#[test]
fn the_resolved_spec_names_the_source_and_the_derivation_but_no_path() {
    let (config, index) = testkit::moe_fp8();
    let p = plan(
        &config,
        None,
        &index,
        &testkit::spec("1", "mode = \"uniform\""),
    );
    for want in [
        "id = \"toy/model\"",
        "revision = \"abc\"",
        "layers_kept = [0, 1, 2, 3]",
        "units_full = 2",
    ] {
        assert!(p.resolved.contains(want), "{want} in\n{}", p.resolved);
    }
    assert_eq!(p.digest.len(), 64);
}

#[test]
fn refusals_name_their_cause() {
    let (config, index) = testkit::moe_fp8();
    let too_many = plan_mock(&MockInputs {
        source_id: "toy/model",
        revision: None,
        config_json: &config,
        hf_quant_config: None,
        index: &index,
        spec: &testkit::spec("3", "mode = \"uniform\""),
        routing: None,
    })
    .unwrap_err();
    assert!(too_many.to_string().contains("has 2 units"), "{too_many}");
    let (dense, dindex) = testkit::dense_ct();
    let wrong_len = plan_mock(&MockInputs {
        source_id: "toy/model",
        revision: None,
        config_json: &dense,
        hf_quant_config: None,
        index: &dindex,
        spec: &testkit::spec("[1, 1, 1]", "mode = \"uniform\""),
        routing: None,
    })
    .unwrap_err();
    assert!(
        wrong_len.to_string().contains("2 layer signatures"),
        "{wrong_len}"
    );
    let mut missing: Vec<TensorEntry> = index.iter().cloned().collect();
    missing.retain(|e| e.name != "model.language_model.layers.3.self_attn.q_proj.weight_scale_inv");
    let missing = TensorIndex::from_entries(missing).unwrap();
    let err = plan_mock(&MockInputs {
        source_id: "toy/model",
        revision: None,
        config_json: &config,
        hf_quant_config: None,
        index: &missing,
        spec: &testkit::spec("1", "mode = \"uniform\""),
        routing: None,
    })
    .unwrap_err();
    assert!(err.to_string().contains("q_proj.weight"), "{err}");
}
