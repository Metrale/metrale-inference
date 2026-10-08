// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Which Laguna MoE layers take the packed-int path, and the tensor checks of
//! that load, over the pinned Laguna-XS-2.1-INT4 config and a mock store. The real shape is
//! narrowed (4 experts, hidden 256, intermediate 128) so the mock allocations stay small.
//!
//! Owner: model-arch weight loader (Laguna).
//! Invariants: none beyond the types.

use std::collections::HashMap;

use metrale_config::ModelConfig;
use metrale_config::precision_plan::packed_int::PackedIntScheme;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_model_layers::layers::FfnComponent;
use metrale_model_weights::weights::{WeightDtype, WeightStore, WeightTensor};

use super::{load_packed_int_moe, packed_int_scheme};

const INT4_CONFIG: &str =
    include_str!("../../../../config/src/precision_plan/fixtures/poolside_laguna_xs_2_1_int4.json");

const E: usize = 4;
const H: usize = 256;
const I: usize = 128;

fn config() -> ModelConfig {
    let mut c = metrale_config::parse_config(INT4_CONFIG).unwrap();
    c.num_experts = E;
    c.num_experts_per_tok = 2;
    c.hidden_size = H;
    c.moe_intermediate_size = I;
    c.shared_expert_intermediate_size = I;
    c
}

/// 2026-10-07: Tensors of one MoE layer under `mlp`, laid out for `bits` (0 = an NVFP4-style
/// U8 `weight_packed`), each a fresh mock allocation.
fn layer_tensors(gpu: &dyn GpuBackend, mlp: &str, bits: usize) -> HashMap<String, WeightTensor> {
    let mut m = HashMap::new();
    let mut put = |name: String, dtype: WeightDtype, shape: Vec<usize>| {
        let bytes = shape.iter().product::<usize>() * dtype.byte_size();
        let ptr = gpu.alloc(bytes.max(1)).unwrap();
        m.insert(name, WeightTensor { ptr, shape, dtype });
    };
    for e in 0..E {
        for (proj, n, k) in [("gate_proj", I, H), ("up_proj", I, H), ("down_proj", H, I)] {
            let p = format!("{mlp}.experts.{e}.{proj}");
            if bits == 0 {
                put(
                    format!("{p}.weight_packed"),
                    WeightDtype::UInt8,
                    vec![n, k / 2],
                );
            } else {
                put(
                    format!("{p}.weight_packed"),
                    WeightDtype::Int32,
                    vec![n, k * bits / 32],
                );
            }
            put(
                format!("{p}.weight_scale"),
                WeightDtype::BF16,
                vec![n, k / 128],
            );
        }
    }
    put(format!("{mlp}.gate.weight"), WeightDtype::BF16, vec![E, H]);
    put(
        format!("{mlp}.experts.e_score_correction_bias"),
        WeightDtype::FP32,
        vec![E],
    );
    for (proj, n, k) in [("gate_proj", I, H), ("up_proj", I, H), ("down_proj", H, I)] {
        put(
            format!("{mlp}.shared_expert.{proj}.weight"),
            WeightDtype::BF16,
            vec![n, k],
        );
    }
    m
}

#[test]
fn layers_1_to_30_are_int4_and_31_to_39_int8() {
    let gpu = MockGpuBackend::new();
    let c = config();
    let mut all = HashMap::new();
    for (layer, bits) in [(1, 4), (30, 4), (31, 8), (39, 8)] {
        all.extend(layer_tensors(
            &gpu,
            &format!("model.layers.{layer}.mlp"),
            bits,
        ));
    }
    let store = WeightStore::from_map(all);
    for (layer, want) in [
        (1, PackedIntScheme::INT4_G128),
        (30, PackedIntScheme::INT4_G128),
        (31, PackedIntScheme::INT8_G128),
        (39, PackedIntScheme::INT8_G128),
    ] {
        let mlp = format!("model.layers.{layer}.mlp");
        assert_eq!(
            packed_int_scheme(&store, &c, &mlp).unwrap(),
            Some(want),
            "{mlp}"
        );
    }
}

#[test]
fn nvfp4_u8_experts_keep_the_nvfp4_path() {
    let gpu = MockGpuBackend::new();
    let mlp = "model.layers.1.mlp";
    let store = WeightStore::from_map(layer_tensors(&gpu, mlp, 0));
    assert_eq!(packed_int_scheme(&store, &config(), mlp).unwrap(), None);
    assert_eq!(
        packed_int_scheme(&store, &config(), "model.layers.7.mlp").unwrap(),
        None
    );
}

#[test]
fn i32_experts_without_a_declared_scheme_are_refused() {
    let gpu = MockGpuBackend::new();
    let mlp = "model.layers.1.mlp";
    let store = WeightStore::from_map(layer_tensors(&gpu, mlp, 4));
    let mut c = config();
    c.quantization_config = None;
    let err = packed_int_scheme(&store, &c, mlp).unwrap_err();
    assert!(
        err.to_string().contains("declares no quantization"),
        "{err}"
    );
}

#[test]
fn a_well_formed_layer_builds_the_packed_int_moe() {
    let gpu = MockGpuBackend::new();
    let mlp = "model.layers.31.mlp";
    let store = WeightStore::from_map(layer_tensors(&gpu, mlp, 8));
    let c = config();
    let scheme = packed_int_scheme(&store, &c, mlp).unwrap().unwrap();
    match load_packed_int_moe(&store, &c, &gpu, mlp, scheme).unwrap() {
        FfnComponent::PackedIntMoe(layer) => assert_eq!(layer.scheme(), PackedIntScheme::INT8_G128),
        _ => panic!("expected the packed-int MoE"),
    }
}

#[test]
fn malformed_tensors_are_refused() {
    let gpu = MockGpuBackend::new();
    let mlp = "model.layers.1.mlp";
    let c = config();
    let cases: [(&str, WeightDtype, Vec<usize>, &str); 5] = [
        (
            "experts.2.down_proj.weight_scale",
            WeightDtype::FP32,
            vec![H, I / 128],
            "BF16",
        ),
        (
            "experts.0.up_proj.weight_packed",
            WeightDtype::Int32,
            vec![I, H / 4],
            "weight_packed shape",
        ),
        (
            "experts.3.gate_proj.weight_scale",
            WeightDtype::BF16,
            vec![I, 1],
            "weight_scale shape",
        ),
        (
            "experts.e_score_correction_bias",
            WeightDtype::BF16,
            vec![E],
            "e_score_correction_bias",
        ),
        (
            "shared_expert.down_proj.weight",
            WeightDtype::BF16,
            vec![I, H],
            "shared_expert.down_proj",
        ),
    ];
    for (name, dtype, shape, want) in cases {
        let mut t = layer_tensors(&gpu, mlp, 4);
        let key = format!("{mlp}.{name}");
        let ptr = t[&key].ptr;
        t.insert(key, WeightTensor { ptr, shape, dtype });
        let store = WeightStore::from_map(t);
        let err = load_packed_int_moe(&store, &c, &gpu, mlp, PackedIntScheme::INT4_G128)
            .err()
            .unwrap_or_else(|| panic!("{name} accepted"));
        assert!(format!("{err:#}").contains(want), "{name}: {err:#}");
    }
}
