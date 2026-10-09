// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The layer schedule of checked-in checkpoint configs (Path A), and the layer-index
//! parse of module paths, including the near misses it must refuse (Path B).
//!
//! Owner: metrale-circuit.
//! Invariants: the expected values are read off the fixture configs by hand.

use super::*;

const MOE_FP8: &str =
    include_str!("../../tests/fixtures/checkpoints/Qwen--Qwen3.6-35B-A3B-FP8/config.json");
const DENSE_NVFP4: &str =
    include_str!("../../tests/fixtures/checkpoints/unsloth--Qwen3.8-27B-NVFP4/config.json");

#[test]
fn the_moe_hybrid_schedule_is_read_from_its_map_and_circuit() {
    let s = layer_schedule(MOE_FP8).expect("schedule");
    assert_eq!(s.arch, "qwen3_6_moe");
    assert_eq!(s.layer_module, "model.language_model.layers.{i}");
    assert_eq!(s.draft_module.as_deref(), Some("mtp.layers.0"));
    assert_eq!(s.layout, LayoutRule::Interval { period: 4 });
    assert_eq!(s.layer_kinds.len(), 40);
    assert_eq!(s.layer_kinds[3], LayerKind::FullAttention);
    assert_eq!(s.layer_kinds[4], LayerKind::LinearAttention);
    assert_eq!(s.nest.as_deref(), Some("text_config"));
    assert_eq!(s.count_key, "num_hidden_layers");
    assert_eq!(s.source_keys, vec!["layer_types".to_string()]);
}

#[test]
fn the_dense_hybrid_has_sixty_four_layers() {
    let s = layer_schedule(DENSE_NVFP4).expect("schedule");
    assert_eq!(s.arch, "qwen3_5");
    assert_eq!(s.layer_kinds.len(), 64);
    assert_eq!(s.module_of(17), "model.language_model.layers.17");
}

#[test]
fn layer_of_reads_only_whole_index_segments() {
    let s = layer_schedule(MOE_FP8).expect("schedule");
    assert_eq!(
        s.layer_of("model.language_model.layers.12.mlp.gate.weight"),
        Some(12)
    );
    assert_eq!(s.layer_of("model.language_model.layers.0"), Some(0));
    assert_eq!(s.layer_of("model.language_model.layers.12x.mlp"), None);
    assert_eq!(s.layer_of("model.language_model.layers..mlp"), None);
    assert_eq!(s.layer_of("mtp.layers.0.mlp.gate.weight"), None);
    assert_eq!(s.layer_of("model.language_model.embed_tokens.weight"), None);
}

#[test]
fn an_unserved_model_type_is_refused() {
    let err = layer_schedule(r#"{"model_type": "no_such_model"}"#).unwrap_err();
    assert!(
        matches!(err, CheckpointError::UnknownModelType { .. }),
        "{err}"
    );
}
