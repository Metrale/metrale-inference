// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The vision-tower bind decision of `Glm5NextWeightLoader`, and the Qwen family
//! (2026-10-13).
//!
//! Owner: model-arch weight loader (GLM-5.3).
//! Invariants: none beyond the types.

use super::Glm5NextWeightLoader;
use crate::weight_loader::ModelWeightLoader;
use crate::weight_loader::qwen4_exp::Qwen4ExpWeightLoader;
use crate::weight_loader::qwen35::Qwen35WeightLoader;
use metrale_config::ModelConfig;

/// 2026-09-25: The loader binds the tower exactly when
/// `metrale_config::glm_vision_enabled()` says so, and an unset
/// `METRALE_GLM_VISION` means off.
///
/// Both states are tested through `glm_vision_enabled_from`, because setting
/// the variable here would race the other tests in the binary.
#[test]
fn the_vision_tower_is_off_unless_the_operator_asks() {
    assert_eq!(
        Glm5NextWeightLoader.binds_vision_encoder(&ModelConfig::qwen3_next_80b_nvfp4()),
        metrale_config::glm_vision_enabled(),
        "the loader's withhold decision must be the SAME gate the config \
             parser reads, or the two disagree and the loader asks for tensors \
             that were never uploaded"
    );
    assert!(
        !metrale_config::glm_vision_enabled_from(None),
        "unset must mean off: binding the tower costs 1.05 GiB/rank"
    );
    assert!(metrale_config::glm_vision_enabled_from(Some("1")));
}

#[test]
fn a_qwen_family_loader_binds_the_tower_exactly_when_the_config_declares_one() {
    // 2026-10-13: `Qwen35WeightLoader::load_vision_encoder` returns `None` without a
    // `vision_config`; the bind decision must agree, or a text-only view uploads the tower
    // only to free it (0.84 GiB of peak on Qwen3.8-Flash-Next) and a multimodal one would
    // withhold tensors its loader reads.
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.vision = None;
    assert!(!Qwen35WeightLoader.binds_vision_encoder(&config));
    assert!(!Qwen4ExpWeightLoader.binds_vision_encoder(&config));
    config.vision = Some(metrale_config::VisionConfig::default());
    assert!(Qwen35WeightLoader.binds_vision_encoder(&config));
    assert!(Qwen4ExpWeightLoader.binds_vision_encoder(&config));
}
