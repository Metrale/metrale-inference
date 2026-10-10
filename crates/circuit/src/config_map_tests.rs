// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The config-map schema's own refusals, and the rules the checkpoint tests do not
//! reach: a dim outside its `allowed` values, a nested-object key with no rule, and a map
//! whose `variant` or `[root]` does not fit its `model_types` or `nest`.

use serde_json::json;

use super::{ConfigMap, ConfigMapError, map_config};

const MAP: &str = r#"
schema = 1
arch = "toy"
model_types = ["toy"]
[layers]
count = "num_hidden_layers"
uniform = "full_attention"
[dims]
hidden = "hidden_size"
mtp = { key = "num_nextn_predict_layers", default = 0, allowed = [0, 1] }
[keys]
model_type = "ignore"
rope_scaling = { param = "rope_scaling", object = { rope_type = { param = true, require = ["llama3"] } } }
"#;

fn map(config: serde_json::Value) -> Result<super::MappedConfig, ConfigMapError> {
    map_config(&ConfigMap::parse(MAP).unwrap(), &config)
}

#[test]
fn allowed_dims_nested_objects_and_defaults() {
    let base = json!({"model_type": "toy", "num_hidden_layers": 2, "hidden_size": 8});
    let m = map(base.clone()).unwrap();
    assert_eq!(m.shape.dims["mtp"], 0, "the stated default");
    assert_eq!(m.shape.layer_kinds.len(), 2);
    let mut two = base.clone();
    two["num_nextn_predict_layers"] = 2.into();
    assert!(
        matches!(map(two), Err(ConfigMapError::Refused { key, .. }) if key == "num_nextn_predict_layers")
    );
    let mut rope = base.clone();
    rope["rope_scaling"] = json!({"rope_type": "llama3"});
    let m = map(rope.clone()).unwrap();
    assert_eq!(m.params["rope_scaling.rope_type"], "\"llama3\"");
    rope["rope_scaling"]["beta_fast"] = 32.into();
    assert!(
        matches!(map(rope), Err(ConfigMapError::UnmappedKey { key, .. }) if key == "rope_scaling.beta_fast")
    );
    let mut missing = base;
    missing.as_object_mut().unwrap().remove("hidden_size");
    assert!(
        matches!(map(missing), Err(ConfigMapError::MissingKey { key, .. }) if key == "hidden_size")
    );
}

#[test]
fn a_malformed_map_is_refused() {
    let bad_variant = MAP.replace("[keys]", "[variant.other.keys]\nx = \"ignore\"\n[keys]");
    assert!(
        matches!(ConfigMap::parse(&bad_variant), Err(ConfigMapError::Schema(m)) if m.contains("variant `other`"))
    );
    let nest_without_root = MAP.replace(
        "model_types = [\"toy\"]",
        "model_types = [\"toy\"]\nnest = \"text_config\"",
    );
    assert!(
        matches!(ConfigMap::parse(&nest_without_root), Err(ConfigMapError::Schema(m)) if m.contains("[root]"))
    );
}

/// 2026-10-10: Integer list entries map by their decimal spelling, and a `trailing` key's
/// count of extra entries (MTP layers) must be listed, must map, and is dropped.
#[test]
fn integer_entries_map_and_trailing_entries_are_checked_then_dropped() {
    let text = MAP.replace(
        "uniform = \"full_attention\"",
        "sources = [{ key = \"ratios\", values = { \"0\" = \"sliding_attention\", \"4\" = \
         \"compressed_sparse_attention\" }, trailing = \"num_nextn_predict_layers\" }]",
    );
    let m = |cfg: serde_json::Value| map_config(&ConfigMap::parse(&text).unwrap(), &cfg);
    let base = json!({"model_type": "toy", "num_hidden_layers": 2, "hidden_size": 8,
        "num_nextn_predict_layers": 1, "ratios": [4, 0, 0]});
    let kinds = m(base.clone()).unwrap().shape.layer_kinds;
    assert_eq!(
        kinds,
        [
            crate::ir::LayerKind::CompressedSparseAttention,
            crate::ir::LayerKind::SlidingAttention
        ]
    );
    let refused = |cfg, want: &str| match m(cfg) {
        Err(ConfigMapError::Refused { key, .. }) => assert_eq!(key, want),
        other => panic!("{other:?}"),
    };
    let mut short = base.clone();
    short["ratios"] = json!([4, 0]);
    refused(short, "num_hidden_layers");
    let mut unmapped_trailing = base.clone();
    unmapped_trailing["ratios"] = json!([4, 0, 8]);
    refused(unmapped_trailing, "ratios");
    let mut no_trailing = base.clone();
    no_trailing["num_nextn_predict_layers"] = 0.into();
    refused(no_trailing, "num_hidden_layers");
    let mut absent = base;
    absent
        .as_object_mut()
        .unwrap()
        .remove("num_nextn_predict_layers");
    assert!(
        matches!(m(absent), Err(ConfigMapError::MissingKey { key, .. }) if key == "num_nextn_predict_layers")
    );
}
