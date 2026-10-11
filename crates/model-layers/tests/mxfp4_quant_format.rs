// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Metadata-only native layout and legacy mapping regression controls.
use metrale_config::ModelConfig;
use metrale_model_layers::quant_format::*;
use metrale_model_layers::weight_map;
use metrale_model_layers::weight_map::Nvfp4Variant;
use metrale_model_weights::weights::WeightStore;
#[path = "../src/quant_format/quant_format_tests.rs"]
mod legacy;

mod tests {
    use super::*;
    use metrale_gpu_runtime::gpu::DevicePtr;
    use metrale_model_weights::weights::{WeightDtype, WeightTensor};
    use std::collections::HashMap;
    fn fixture() -> (ModelConfig, HashMap<String, WeightTensor>) {
        let config = metrale_config::parse_config(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../circuit/tests/fixtures/checkpoints/openai--gpt-oss-20b/config.json"
        )))
        .unwrap();
        let mut map = HashMap::new();
        for layer in 0..24 {
            for (projection, rows) in [("gate_up_proj", 5760), ("down_proj", 2880)] {
                for (suffix, shape) in [
                    ("blocks", vec![32, rows, 90, 16]),
                    ("scales", vec![32, rows, 90]),
                ] {
                    map.insert(
                        format!("model.layers.{layer}.mlp.experts.{projection}_{suffix}"),
                        WeightTensor {
                            ptr: DevicePtr(0x1000),
                            dtype: WeightDtype::UInt8,
                            shape,
                        },
                    );
                }
            }
        }
        (config, map)
    }
    #[test]
    fn packed_layout_has_no_nvfp4_mapping() {
        let (config, map) = fixture();
        let format = detect_quant_format(&config, &WeightStore::from_map(map)).unwrap();
        assert_eq!(format.name(), "mxfp4-e2m1-e8m0-native");
        assert_eq!(format.base_variant(), None);
        assert_eq!(
            format.variant_for("model.layers.0.mlp.experts.gate_up_proj"),
            None
        );
        assert_eq!(format.variant_for("lm_head"), Some(Nvfp4Variant::Bf16Raw));
    }
    #[test]
    fn packed_format_refuses_missing_and_malformed_metadata() {
        for control in 0..5 {
            let (config, mut map) = fixture();
            let key = "model.layers.23.mlp.experts.down_proj_scales";
            match control {
                0 => {
                    map.remove(key);
                }
                1 => map.get_mut(key).unwrap().dtype = WeightDtype::FP8E8M0,
                2 => map.get_mut(key).unwrap().shape.swap(1, 2),
                3 => map.get_mut(key).unwrap().ptr = DevicePtr(0),
                _ => map.get_mut(key).unwrap().ptr = DevicePtr(u64::MAX - 1),
            }
            assert!(Mxfp4Format::from_checkpoint(&config, &WeightStore::from_map(map)).is_err());
        }
        let (mut config, map) = fixture();
        config.model_type = "qwen3".into();
        assert!(Mxfp4Format::from_checkpoint(&config, &WeightStore::from_map(map)).is_err());
    }
}
