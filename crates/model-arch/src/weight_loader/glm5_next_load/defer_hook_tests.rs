// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests of the defer hook: which tensors `Glm5NextWeightLoader`
//! keeps off the device, what it never defers, and that a deferred expert
//! quantises to the same bytes as a resident one.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_model_weights::weights::{DeferredTensor, WeightDtype, WeightStore, WeightTensor};

use super::{bind_expert, is_full_width_mtp_expert};
use crate::weight_loader::ModelWeightLoader;

const LAYERS: usize = 45;

fn qualified(layer: usize, leaf: &str) -> String {
    format!("model.language_model.layers.{layer}.{leaf}")
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
        .collect()
}

/// 2026-09-25: `n` sign-mixed, evenly spaced values, so a swapped nibble or a
/// dropped scale changes the quantised bytes.
fn ramp(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 - n as f32 / 2.0) * 0.125)
        .collect()
}

#[test]
fn only_the_mtp_layers_full_width_routed_experts_are_deferred() {
    let yes = |n: &str| is_full_width_mtp_expert(n, WeightDtype::BF16, LAYERS);
    for p in ["gate_proj", "up_proj", "down_proj"] {
        assert!(yes(&qualified(
            LAYERS,
            &format!("mlp.experts.0.{p}.weight")
        )));
        assert!(yes(&qualified(
            LAYERS,
            &format!("mlp.experts.287.{p}.weight")
        )));
    }

    // 2026-09-25: The MTP layer's shared expert, router and other non-expert
    // tensors are not deferred.
    for leaf in [
        "mlp.shared_experts.gate_proj.weight",
        "mlp.gate.weight",
        "mlp.gate.e_score_correction_bias",
        "self_attn.q_proj.weight",
        "eh_proj.weight",
    ] {
        assert!(!yes(&qualified(LAYERS, leaf)), "{leaf}");
    }

    // 2026-09-25: Text-layer experts are not deferred.
    assert!(!yes(&qualified(44, "mlp.experts.0.gate_proj.weight")));
    assert!(!yes(&qualified(0, "mlp.experts.0.gate_proj.weight")));

    assert!(!yes("lm_head.weight"));
    assert!(!yes(
        "model.language_model.layers.x.mlp.experts.0.up_proj.weight"
    ));
}

/// 2026-09-25: A routed expert of the MTP layer stored as U8, FP8 or FP32 is
/// not deferred, and neither is a `weight_scale` sibling.
#[test]
fn a_packed_expert_is_never_deferred_whatever_the_layer() {
    for dtype in [WeightDtype::UInt8, WeightDtype::FP8E4M3, WeightDtype::FP32] {
        assert!(!is_full_width_mtp_expert(
            &qualified(LAYERS, "mlp.experts.0.gate_proj.weight"),
            dtype,
            LAYERS
        ));
    }
    assert!(!is_full_width_mtp_expert(
        &qualified(LAYERS, "mlp.experts.0.gate_proj.weight_scale"),
        WeightDtype::BF16,
        LAYERS
    ));
}

/// 2026-09-25: The deferred layer is `num_hidden_layers` of the config given to
/// `defer_predicate`.
#[test]
fn the_deferred_layer_comes_from_the_config_not_a_literal() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.num_hidden_layers = 7;
    let hook = super::Glm5NextWeightLoader
        .defer_predicate(&config)
        .expect("glm5_next declares a defer predicate");

    assert!(hook(
        &qualified(7, "mlp.experts.3.down_proj.weight"),
        WeightDtype::BF16
    ));
    assert!(!hook(
        &qualified(45, "mlp.experts.3.down_proj.weight"),
        WeightDtype::BF16
    ));
}

/// 2026-09-25: A loader that does not override `defer_predicate` gets `None`,
/// which defers nothing.
#[test]
fn a_loader_that_does_not_override_the_hook_defers_nothing() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    assert!(
        crate::weight_loader::qwen35::Qwen35WeightLoader
            .defer_predicate(&config)
            .is_none()
    );
}

/// 2026-09-25: Write `bytes` into a scratch shard at `offset`, after `offset`
/// filler bytes.
fn stage_shard(tag: &str, offset: u64, bytes: &[u8]) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("metrale-glm5next-defer-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{tag}.safetensors"));
    let mut blob = vec![0xAAu8; offset as usize];
    blob.extend_from_slice(bytes);
    std::fs::write(&path, &blob).unwrap();
    path
}

/// 2026-09-25: An expert read from its shard and the same expert read back
/// from the device quantise to identical NVFP4 bytes and global scales.
#[test]
fn a_deferred_expert_quantises_to_the_same_bytes_as_a_resident_one() {
    let values = ramp(32);
    let (rows, cols) = (2usize, 16usize);
    let bytes = bf16_bytes(&values);

    let gpu_a = MockGpuBackend::new();
    let mut map = std::collections::HashMap::new();
    for p in ["gate_proj", "up_proj", "down_proj"] {
        let ptr = gpu_a.alloc(bytes.len()).unwrap();
        gpu_a.copy_h2d(&bytes, ptr).unwrap();
        map.insert(
            qualified(LAYERS, &format!("mlp.experts.0.{p}.weight")),
            WeightTensor {
                ptr,
                shape: vec![rows, cols],
                dtype: WeightDtype::BF16,
            },
        );
    }
    let resident = bind_expert(&gpu_a, &WeightStore::from_map(map), LAYERS, 0).unwrap();

    let path = stage_shard("expert", 137, &bytes);
    let gpu_b = MockGpuBackend::new();
    let mut store = WeightStore::from_map(std::collections::HashMap::new());
    for p in ["gate_proj", "up_proj", "down_proj"] {
        store.defer(
            qualified(LAYERS, &format!("mlp.experts.0.{p}.weight")),
            DeferredTensor {
                path: path.clone(),
                offset: 137,
                shape: vec![rows, cols],
                dtype: WeightDtype::BF16,
            },
        );
    }
    let deferred = bind_expert(&gpu_b, &store, LAYERS, 0).unwrap();

    for (a, b, what) in [
        (&resident.gate_proj, &deferred.gate_proj, "gate"),
        (&resident.up_proj, &deferred.up_proj, "up"),
        (&resident.down_proj, &deferred.down_proj, "down"),
    ] {
        assert_eq!(
            gpu_a.read_alloc(a.packed).unwrap(),
            gpu_b.read_alloc(b.packed).unwrap(),
            "{what}: packed codes differ between the shard and device paths"
        );
        assert_eq!(
            gpu_a.read_alloc(a.scale).unwrap(),
            gpu_b.read_alloc(b.scale).unwrap(),
            "{what}: block scales differ"
        );
        assert_eq!(a.scale_2, b.scale_2, "{what}: global scale differs");
    }

    // 2026-09-25: The deferred arm adopted only the NVFP4 buffers: per
    // projection, packed `[2, 8]` and scales `[2, 1]`.
    assert_eq!(store.derived().len(), 6);
    assert_eq!(store.derived().bytes(), 3 * (rows * cols / 2 + rows));
    assert!(
        store.derived().bytes() < 3 * bytes.len(),
        "the point of deferring is that less reaches the device than is on disk"
    );

    let _ = std::fs::remove_file(&path);
}

/// 2026-09-25: A deferred expert that is not BF16 is refused, with an error that
/// says it was deferred.
#[test]
fn a_deferred_expert_at_an_unsupported_width_is_refused() {
    let path = stage_shard("odd-width", 0, &[0x21u8; 16]);
    let gpu = MockGpuBackend::new();
    let mut store = WeightStore::from_map(std::collections::HashMap::new());
    store.defer(
        qualified(LAYERS, "mlp.experts.0.gate_proj.weight"),
        DeferredTensor {
            path: path.clone(),
            offset: 0,
            shape: vec![2, 8],
            dtype: WeightDtype::UInt8,
        },
    );
    let err = bind_expert(&gpu, &store, LAYERS, 0)
        .unwrap_err()
        .to_string();
    assert!(err.contains("deferred"), "{err}");

    let _ = std::fs::remove_file(&path);
}

/// 2026-10-08: The hook also keeps every text layer's F32 activation scale off the device, and
/// nothing else that ends in a scale: not `weight_scale_2`, not a 16-bit spelling, not a tensor
/// outside the text layers.
#[test]
fn text_layer_activation_scales_are_deferred_and_nothing_else_new() {
    let hook = super::Glm5NextWeightLoader
        .defer_predicate(&ModelConfig::qwen3_next_80b_nvfp4())
        .expect("glm5_next declares a defer predicate");
    for leaf in [
        "mlp.experts.7.down_proj.input_scale",
        "mlp.gate_proj.input_scale",
    ] {
        assert!(hook(&qualified(5, leaf), WeightDtype::FP32), "{leaf}");
    }
    for (name, dtype) in [
        (
            qualified(5, "mlp.experts.7.down_proj.weight_scale_2"),
            WeightDtype::FP32,
        ),
        (
            qualified(5, "mlp.experts.7.down_proj.input_scale"),
            WeightDtype::BF16,
        ),
        (
            "model.visual.blocks.0.mlp.input_scale".to_string(),
            WeightDtype::FP32,
        ),
        (
            qualified(5, "mlp.experts.7.down_proj.weight"),
            WeightDtype::UInt8,
        ),
    ] {
        assert!(!hook(&name, dtype), "{name} {dtype:?}");
    }
}

/// 2026-10-08: A resident U8 expert: per projection, packed codes, E4M3 block scales and a
/// scalar `weight_scale_2`, with the given `input_scale` bytes deferred to a staged shard (or no
/// `input_scale` at all for `None`).
fn packed_expert_store(tag: &str, gpu: &MockGpuBackend, scales: Option<[&[u8]; 3]>) -> WeightStore {
    let mut map = std::collections::HashMap::new();
    let mut put = |name: String, bytes: &[u8], shape: Vec<usize>, dtype: WeightDtype| {
        let ptr = gpu.alloc(bytes.len()).unwrap();
        gpu.copy_h2d(bytes, ptr).unwrap();
        map.insert(name, WeightTensor { ptr, shape, dtype });
    };
    let projs = ["gate_proj", "up_proj", "down_proj"];
    for p in projs {
        let base = format!("mlp.experts.0.{p}");
        put(
            qualified(4, &format!("{base}.weight")),
            &[0x21; 16],
            vec![2, 8],
            WeightDtype::UInt8,
        );
        put(
            qualified(4, &format!("{base}.weight_scale")),
            &[0x38; 2],
            vec![2, 1],
            WeightDtype::FP8E4M3,
        );
        put(
            qualified(4, &format!("{base}.weight_scale_2")),
            &0.5f32.to_le_bytes(),
            vec![],
            WeightDtype::FP32,
        );
    }
    let mut store = WeightStore::from_map(map);
    if let Some(scales) = scales {
        for (p, bytes) in projs.iter().zip(scales) {
            let path = stage_shard(&format!("input-scale-{tag}-{p}-{}", bytes.len()), 9, bytes);
            store.defer(
                qualified(4, &format!("mlp.experts.0.{p}.input_scale")),
                DeferredTensor {
                    path,
                    offset: 9,
                    shape: if bytes.len() == 4 {
                        vec![]
                    } else {
                        vec![bytes.len() / 4]
                    },
                    dtype: WeightDtype::FP32,
                },
            );
        }
    }
    store
}

/// 2026-10-08: A packed expert binds each projection's own deferred activation scale, read from
/// its shard (gate, up and down distinct here so a crossed wire shows), and none when the
/// checkpoint has none.
#[test]
fn a_packed_expert_binds_its_deferred_activation_scales() {
    let gpu = MockGpuBackend::new();
    let (g, u, d) = (
        0.25f32.to_le_bytes(),
        0.5f32.to_le_bytes(),
        0.0372f32.to_le_bytes(),
    );
    let store = packed_expert_store("bind", &gpu, Some([&g, &u, &d]));
    let e = bind_expert(&gpu, &store, 4, 0).unwrap();
    assert_eq!(e.gate_proj.input_scale, Some(0.25));
    assert_eq!(e.up_proj.input_scale, Some(0.5));
    assert_eq!(e.down_proj.input_scale, Some(0.0372));
    assert_eq!(
        e.down_proj.scale_2, 0.5,
        "weight_scale_2 still binds beside it"
    );

    let gpu = MockGpuBackend::new();
    let none = bind_expert(&gpu, &packed_expert_store("none", &gpu, None), 4, 0).unwrap();
    assert_eq!(none.gate_proj.input_scale, None);
}

/// 2026-10-08: An activation scale that is zero, not finite, or not one scalar is refused.
#[test]
fn a_malformed_activation_scale_is_refused() {
    let ok = 0.25f32.to_le_bytes();
    let two: Vec<u8> = [0.25f32, 0.5]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    for (bad, want) in [
        (0.0f32.to_le_bytes().to_vec(), "finite and positive"),
        (f32::NAN.to_le_bytes().to_vec(), "finite and positive"),
        (two, "one scalar"),
    ] {
        let gpu = MockGpuBackend::new();
        let store = packed_expert_store(&format!("bad{}", bad.len()), &gpu, Some([&ok, &ok, &bad]));
        let err = bind_expert(&gpu, &store, 4, 0).unwrap_err().to_string();
        assert!(err.contains(want), "{err}");
    }
}
