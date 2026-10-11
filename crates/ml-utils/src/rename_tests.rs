// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Names under kept layers are renumbered, under dropped layers dropped, elsewhere
//! untouched; metadata keys and list entries follow; patterns stay; config schedule keys shrink.
//!
//! Owner: metrale-ml-utils.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use serde_json::json;

use super::*;
use crate::testkit;

fn schedule() -> LayerSchedule {
    metrale_circuit::layer_schedule(&testkit::dense_ct().0).unwrap()
}

#[test]
fn names_are_renumbered_dropped_or_kept() {
    let s = schedule();
    let renumber = BTreeMap::from([(0, 0), (8, 1)]);
    let r = Renamer::new(&s, &renumber);
    assert_eq!(
        r.name("model.language_model.layers.8.mlp.up_proj.weight")
            .as_deref(),
        Some("model.language_model.layers.1.mlp.up_proj.weight")
    );
    assert_eq!(
        r.name("model.language_model.layers.5.mlp.up_proj.weight"),
        None
    );
    assert_eq!(r.name("lm_head.weight").as_deref(), Some("lm_head.weight"));
    assert_eq!(
        r.name("model.language_model.layers.80.mlp").as_deref(),
        None
    );
}

#[test]
fn metadata_keys_and_list_entries_follow_and_patterns_stay() {
    let s = schedule();
    let renumber = BTreeMap::from([(8, 0)]);
    let r = Renamer::new(&s, &renumber);
    let mut v = json!({
        "quantized_layers": {
            "model.language_model.layers.8.mlp.down_proj": {"quant_algo": "FP8"},
            "model.language_model.layers.2.mlp.down_proj": {"quant_algo": "FP8"},
            "lm_head": {"quant_algo": "FP8"}
        },
        "ignore": ["model.language_model.layers.2.mlp.gate", "model.language_model.layers.8.mlp.gate", "re:.*layers\\.8\\..*", "model.visual.blocks.0"]
    });
    r.rewrite_metadata(&mut v);
    let ql = v["quantized_layers"].as_object().unwrap();
    assert_eq!(
        ql.keys().collect::<Vec<_>>(),
        ["model.language_model.layers.0.mlp.down_proj", "lm_head"]
    );
    assert_eq!(
        v["ignore"],
        json!([
            "model.language_model.layers.0.mlp.gate",
            "re:.*layers\\.8\\..*",
            "model.visual.blocks.0"
        ])
    );
}

#[test]
fn config_schedule_keys_shrink_and_a_mismatch_is_refused() {
    let (config, _) = testkit::dense_ct();
    let s = schedule();
    let renumber = BTreeMap::from([(0, 0), (1, 1), (2, 2), (3, 3)]);
    let r = Renamer::new(&s, &renumber);
    let c: serde_json::Value = serde_json::from_str(&config).unwrap();
    let out = r.rewrite_config(&c).unwrap();
    assert_eq!(out["text_config"]["num_hidden_layers"], 4);
    assert_eq!(
        out["text_config"]["layer_types"].as_array().unwrap().len(),
        4
    );
    let keys_in: Vec<&String> = c.as_object().unwrap().keys().collect();
    let keys_out: Vec<&String> = out.as_object().unwrap().keys().collect();
    assert_eq!(keys_in, keys_out, "key order is kept");
    let mut bad = c.clone();
    bad["text_config"]["num_hidden_layers"] = json!(13);
    assert!(
        r.rewrite_config(&bad)
            .unwrap_err()
            .to_string()
            .contains("disagrees")
    );
}
