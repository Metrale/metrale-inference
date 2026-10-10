// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The config-map schema's own refusals, and the rules the checkpoint tests do not
//! reach: a dim outside its `allowed` values, a nested-object key with no rule, and a map
//! whose `variant` or `[root]` does not fit its `model_types` or `nest`. 2026-10-10: A presence
//! switch whose value no rule classifies.

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

/// 2026-10-10: A presence switch leaves its key to be classified: with a param rule the switch
/// is set and the value carried; with none the key is refused as unmapped, never dropped.
#[test]
fn a_presence_switch_leaves_its_value_to_be_classified() {
    let switch = MAP.replace(
        "[keys]",
        "cap = { key = \"final_logit_softcapping\", bool_present = true }\n[keys]",
    );
    let with_param = switch.replace(
        "model_type = \"ignore\"",
        "model_type = \"ignore\"\nfinal_logit_softcapping = { param = true }",
    );
    let base = json!({"model_type": "toy", "num_hidden_layers": 2, "hidden_size": 8});
    let run =
        |map: &str, config: &serde_json::Value| map_config(&ConfigMap::parse(map).unwrap(), config);
    let mut capped = base.clone();
    capped["final_logit_softcapping"] = 30.0.into();
    let m = run(&with_param, &capped).unwrap();
    assert_eq!(m.shape.dims["cap"], 1);
    assert_eq!(m.params["final_logit_softcapping"], "30.0");
    let m = run(&with_param, &base).unwrap();
    assert_eq!(m.shape.dims["cap"], 0);
    assert!(!m.params.contains_key("final_logit_softcapping"));
    assert!(
        matches!(run(&switch, &capped), Err(ConfigMapError::UnmappedKey { key, .. }) if key == "final_logit_softcapping")
    );
}
