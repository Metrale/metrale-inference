// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Tests for the FP8 KV scale-source classification.

use std::collections::HashMap;

use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_weights::weights::{WeightDtype, WeightStore, WeightTensor};

use super::{Fp8KvScaleSource, fp8_kv_scale_source};

/// 2026-09-28: A store with only names: the census reads no tensor data.
fn store_named(names: impl IntoIterator<Item = String>) -> WeightStore {
    WeightStore::from_map(
        names
            .into_iter()
            .map(|n| {
                let t = WeightTensor {
                    ptr: DevicePtr::NULL,
                    shape: vec![1],
                    dtype: WeightDtype::BF16,
                };
                (n, t)
            })
            .collect::<HashMap<_, _>>(),
    )
}

/// 2026-09-28: The unsloth Qwen3.8-27B layout (16 full-attention layers, bare
/// `self_attn.k_scale`) classifies as checkpoint-sourced on all 16. Before the
/// shared resolver the count said 16 while every layer ran at 1.0.
#[test]
fn the_unsloth_27b_layout_is_checkpoint_on_every_layer() {
    let store = store_named((0..16).flat_map(|i| {
        let p = format!("model.language_model.layers.{}.self_attn", 4 * i + 3);
        [format!("{p}.k_scale"), format!("{p}.v_scale")]
    }));
    let census = store.kv_scale_census().expect("census");
    assert_eq!(census.spellings(), ["k_scale"]);
    assert_eq!(
        fp8_kv_scale_source(0, census.len(), 16),
        Fp8KvScaleSource::Checkpoint { layers: 16 }
    );
}

/// 2026-09-28: One layer short of the attention-layer count is PARTIAL (a warned
/// state), never reported as checkpoint-sourced.
#[test]
fn one_missing_layer_is_partial_not_checkpoint() {
    assert_eq!(
        fp8_kv_scale_source(0, 15, 16),
        Fp8KvScaleSource::Partial {
            from_checkpoint: 15,
            attention_layers: 16
        }
    );
    assert_eq!(
        fp8_kv_scale_source(0, 0, 16),
        Fp8KvScaleSource::Unscaled {
            attention_layers: 16
        }
    );
}

/// 2026-09-28: Calibration replaces checkpoint scales on every FP8 layer, so it
/// wins whatever the checkpoint ships, and the shadowed count is kept for the
/// log. An extra scaled head (MTP) beyond the count is still every layer.
#[test]
fn calibration_wins_and_extra_scaled_heads_are_not_partial() {
    assert_eq!(
        fp8_kv_scale_source(256, 16, 16),
        Fp8KvScaleSource::Calibration {
            tokens: 256,
            shadowed: 16
        }
    );
    assert_eq!(
        fp8_kv_scale_source(0, 17, 16),
        Fp8KvScaleSource::Checkpoint { layers: 17 }
    );
}
