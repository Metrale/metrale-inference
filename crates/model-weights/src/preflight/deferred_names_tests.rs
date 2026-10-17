// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: Preflight sees the tensors a defer hook kept off the device. With
//! `METRALE_GLM_LAYER_STREAM=1` every text layer's non-expert tensors are deferred, so a walk of
//! the resident names alone found no text layer: the correction-bias shape check checked nothing
//! and the layer count passed only on the MTP layer's index.
//!
//! Owner: metrale-model-weights preflight.
//! Invariants: none beyond the types.

use std::collections::HashMap;

use metrale_config::{ModelConfig, parse_config};
use metrale_gpu_runtime::gpu::DevicePtr;

use super::{check_correction_bias_shape, check_layer_count};
use crate::weights::{DeferredTensor, WeightDtype, WeightStore, WeightTensor};

const LAYERS: usize = 3;
const EXPERTS: usize = 4;

/// 2026-10-17: A checked-in fixture config cut to three text layers, four routed experts and no
/// zero experts; only those fields and `model_type` are read by the two checks.
fn config() -> ModelConfig {
    let mut c = parse_config(include_str!(
        "../../../../docs/k3/fixtures/Kimi-K3-0.40B-config.json"
    ))
    .unwrap();
    c.num_hidden_layers = LAYERS;
    c.num_experts = EXPERTS;
    c.zero_expert_num = 0;
    c
}

fn layer(i: usize, leaf: &str) -> String {
    format!("model.language_model.layers.{i}.{leaf}")
}

/// 2026-10-17: The store a streamed load leaves: the embedding resident, the MTP layer (index
/// `LAYERS`) resident when `mtp`, and `deferred` (name, shape) kept off the device. Only names
/// and shapes are read, so the resident pointers are placeholders.
fn store(deferred: &[(String, Vec<usize>)], mtp: bool) -> WeightStore {
    let resident = |shape: Vec<usize>| WeightTensor {
        ptr: DevicePtr(0),
        shape,
        dtype: WeightDtype::BF16,
    };
    let mut map = HashMap::from([(
        "model.language_model.embed_tokens.weight".to_string(),
        resident(vec![128, 64]),
    )]);
    if mtp {
        map.insert(
            layer(LAYERS, "mlp.gate.e_score_correction_bias"),
            resident(vec![EXPERTS]),
        );
    }
    let mut s = WeightStore::from_map(map);
    for (name, shape) in deferred {
        let d = DeferredTensor {
            path: "unused.safetensors".into(),
            offset: 0,
            shape: shape.clone(),
            dtype: WeightDtype::BF16,
        };
        s.defer(name.clone(), d);
    }
    s
}

/// 2026-10-16: Every text layer's attention weight and router bias, deferred; `bias_len` is
/// layer 1's bias length.
fn streamed_text_layers(bias_len: usize) -> Vec<(String, Vec<usize>)> {
    (0..LAYERS)
        .flat_map(|i| {
            let n = if i == 1 { bias_len } else { EXPERTS };
            [
                (layer(i, "self_attn.q_proj.weight"), vec![64, 64]),
                (layer(i, "mlp.gate.e_score_correction_bias"), vec![n]),
            ]
        })
        .collect()
}

#[test]
fn a_deferred_router_bias_of_the_wrong_length_fails_preflight() {
    let c = config();
    // 2026-10-16: The positive control: the same store with a correct deferred bias passes, so
    // the failure below is the bias length and nothing else.
    check_correction_bias_shape(&store(&streamed_text_layers(EXPERTS), true), &c).unwrap();
    let err = check_correction_bias_shape(&store(&streamed_text_layers(EXPERTS + 1), true), &c)
        .expect_err("a deferred bias with one logit too many must fail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains(&layer(1, "mlp.gate.e_score_correction_bias")),
        "{msg}"
    );
    assert!(msg.contains("has 5 elements"), "{msg}");
}

#[test]
fn the_layer_count_reads_the_deferred_text_layers() {
    let c = config();
    // 2026-10-17: `check_layer_count` returns the highest layer index. With the MTP layer
    // resident it is the MTP layer's either way.
    assert_eq!(
        check_layer_count(&store(&streamed_text_layers(EXPERTS), true), &c).unwrap(),
        LAYERS
    );
    // 2026-10-16: With only deferred text layers (no MTP layer resident), the resident-only walk
    // saw no layer at all and failed; the walk over both passes on the deferred layers.
    let only_text = store(&streamed_text_layers(EXPERTS), false);
    assert_eq!(check_layer_count(&only_text, &c).unwrap(), LAYERS - 1);
    // 2026-10-16: The negative: a deferred checkpoint one text layer short fails.
    let short: Vec<_> = streamed_text_layers(EXPERTS)
        .into_iter()
        .filter(|(n, _)| !n.contains(&format!(".layers.{}.", LAYERS - 1)))
        .collect();
    let err = check_layer_count(&store(&short, false), &c)
        .expect_err("a missing last text layer must fail");
    assert!(format!("{err:#}").contains("has layers 0..2"), "{err:#}");
}
