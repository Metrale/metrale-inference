// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: A renumbered module that a literal-index pattern no longer covers is pinned to the
//! group with its scheme; an unquantized one goes to `ignore`; a ModelOpt change is refused; a
//! pin that would not restore the precision is refused (the detection control: without the pin
//! the precision differs).
//!
//! Owner: metrale-ml-utils.
//! Invariants: none beyond the types.

use serde_json::json;

use super::*;

fn ct() -> Value {
    json!({
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {
            "group_0": {"targets": ["re:.*layers\\.(8|9)\\.mlp\\.down_proj$"],
                "weights": {"num_bits": 8, "type": "float", "strategy": "channel", "dynamic": false, "symmetric": true}},
            "group_1": {"targets": ["re:.*mlp\\.down_proj$"],
                "weights": {"num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16, "dynamic": false, "symmetric": true}}
        },
        "ignore": ["re:.*layers\\.9\\.mlp\\.down_proj$"]
    })
}

fn pair(s: usize, m: usize) -> ModulePair {
    ModulePair {
        source: format!("model.layers.{s}.mlp.down_proj"),
        mock: format!("model.layers.{m}.mlp.down_proj"),
    }
}

#[test]
fn a_pattern_that_no_longer_matches_is_pinned_by_exact_name() {
    let src = ct();
    let mut mock = ct();
    let pairs = [pair(8, 1), pair(0, 0)];
    let before = DeclaredPrecisionPlan::from_quantization_config(&mock).unwrap();
    let want = DeclaredPrecisionPlan::from_quantization_config(&src).unwrap();
    assert_ne!(
        before.resolve(&pairs[0].mock),
        want.resolve(&pairs[0].source),
        "control: it differs unpinned"
    );
    assert_eq!(reconcile(&src, &mut mock, &pairs).unwrap(), 1);
    assert!(
        mock["config_groups"]["group_0"]["targets"]
            .as_array()
            .unwrap()
            .contains(&json!("model.layers.1.mlp.down_proj"))
    );
}

#[test]
fn an_ignored_module_is_pinned_into_ignore() {
    let src = ct();
    let mut mock = ct();
    assert_eq!(reconcile(&src, &mut mock, &[pair(9, 2)]).unwrap(), 1);
    assert!(
        mock["ignore"]
            .as_array()
            .unwrap()
            .contains(&json!("model.layers.2.mlp.down_proj"))
    );
}

#[test]
fn a_changed_modelopt_block_is_refused() {
    let src = json!({"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {"model.layers.8.mlp.down_proj": {"quant_algo": "FP8"}}});
    let mut mock = json!({"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {"model.layers.0.mlp.down_proj": {"quant_algo": "FP8"}}});
    let err = reconcile(&src, &mut mock, &[pair(8, 1)]).unwrap_err();
    assert!(
        err.to_string().contains("would change declared precision"),
        "{err}"
    );
}
