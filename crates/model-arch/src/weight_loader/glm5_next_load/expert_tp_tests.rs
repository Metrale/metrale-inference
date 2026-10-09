// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Tests of the tp-layout expert bind on the mock device: deferred packed experts
//! read back as each rank's slices (rows by ranged read, columns whole), with the per-tensor
//! scales carried; a BF16 expert sliced from its whole quantization; and the defer hook's
//! extra family under the tp layout only.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use metrale_config::{ModelConfig, MoeExpertLayout};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_model_weights::weights::{DeferredTensor, WeightDtype, WeightStore, WeightTensor};

use super::{bind_routed_expert, is_routed_expert_tensor};
use crate::glm5next_mlp::expert_tp::{
    ExpertShard, ExpertSlice, expert_slice, slice_expert_cols, slice_expert_rows,
};
use crate::weight_loader::ModelWeightLoader;

const LAYER: usize = 4;
const HIDDEN: usize = 32;

fn qualified(leaf: &str) -> String {
    format!("model.language_model.layers.{LAYER}.{leaf}")
}

/// 2026-10-09: Bytes that name their position, never zero, so a crossed range shows.
fn tagged(len: usize, salt: usize) -> Vec<u8> {
    (0..len)
        .map(|i| ((i * 29 + salt) % 255 + 1) as u8)
        .collect()
}

/// 2026-10-09: Every tensor written into one scratch shard, each deferred at its offset.
struct Shard {
    store: WeightStore,
    path: std::path::PathBuf,
}

impl Drop for Shard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn shard(tag: &str, tensors: &[(String, WeightDtype, Vec<usize>, Vec<u8>)]) -> Shard {
    let dir = std::env::temp_dir().join(format!("metrale-glm5next-etp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{tag}.safetensors"));
    let mut blob = vec![0xA5u8; 11];
    let mut store = WeightStore::from_map(std::collections::HashMap::new());
    for (name, dtype, shape, bytes) in tensors {
        store.defer(
            name.clone(),
            DeferredTensor {
                path: path.clone(),
                offset: blob.len() as u64,
                shape: shape.clone(),
                dtype: *dtype,
            },
        );
        blob.extend_from_slice(bytes);
    }
    std::fs::write(&path, &blob).unwrap();
    Shard { store, path }
}

/// 2026-10-09: A packed expert of width `full`: gate/up `[full, HIDDEN]`, down `[HIDDEN, full]`,
/// each with scales, `weight_scale_2` = 0.5 / 0.25 / 0.125 and `input_scale` = 1 / 1 / 2.
fn packed_expert(full: usize) -> Vec<(String, WeightDtype, Vec<usize>, Vec<u8>)> {
    let mut t = Vec::new();
    for (i, (p, n, k)) in [
        ("gate_proj", full, HIDDEN),
        ("up_proj", full, HIDDEN),
        ("down_proj", HIDDEN, full),
    ]
    .into_iter()
    .enumerate()
    {
        let base = format!("mlp.experts.7.{p}");
        let w = |leaf: &str| qualified(&format!("{base}.{leaf}"));
        t.push((
            w("weight"),
            WeightDtype::UInt8,
            vec![n, k / 2],
            tagged(n * k / 2, i),
        ));
        t.push((
            w("weight_scale"),
            WeightDtype::FP8E4M3,
            vec![n, k / 16],
            tagged(n * k / 16, 100 + i),
        ));
        let s2 = [0.5f32, 0.25, 0.125][i].to_le_bytes().to_vec();
        t.push((w("weight_scale_2"), WeightDtype::FP32, vec![], s2));
        let a = [1.0f32, 1.0, 2.0][i].to_le_bytes().to_vec();
        t.push((w("input_scale"), WeightDtype::FP32, vec![], a));
    }
    t
}

fn bytes_of(t: &[(String, WeightDtype, Vec<usize>, Vec<u8>)], leaf: &str) -> Vec<u8> {
    t.iter().find(|x| x.0 == qualified(leaf)).unwrap().3.clone()
}

/// 2026-10-09: Every rank's bound slice is exactly the slicer's cut of the whole tensor (gate
/// and up from a ranged read), with the expert's own global and activation scales, on an even
/// and a padded width.
#[test]
fn deferred_packed_expert_binds_each_ranks_slice() {
    for (full, tp) in [(640usize, 3usize), (336, 3)] {
        let t = packed_expert(full);
        let sh = shard(&format!("packed-{full}"), &t);
        for r in 0..tp {
            let s = expert_slice(full, tp, r).unwrap();
            let gpu = MockGpuBackend::new();
            let before = sh.store.derived().bytes();
            let e = bind_routed_expert(&gpu, &sh.store, LAYER, 7, ExpertShard::Sliced(s)).unwrap();
            // 2026-10-09: Only the slices reached the device: three projections of `s.len`
            // rows or columns, codes plus scales.
            assert_eq!(
                sh.store.derived().bytes() - before,
                3 * (s.len * HIDDEN / 2 + s.len * HIDDEN / 16)
            );
            let check = |proj: &crate::glm5next_mlp::weights::Nvfp4Proj, p: &str, rows: bool| {
                let (wp, ws) = (
                    bytes_of(&t, &format!("mlp.experts.7.{p}.weight")),
                    bytes_of(&t, &format!("mlp.experts.7.{p}.weight_scale")),
                );
                let want = if rows {
                    slice_expert_rows(&wp, &ws, HIDDEN, &s).unwrap()
                } else {
                    slice_expert_cols(&wp, &ws, HIDDEN, &s).unwrap()
                };
                assert_eq!(
                    gpu.read_alloc(proj.packed).unwrap(),
                    want.0,
                    "{p} r{r} codes"
                );
                assert_eq!(
                    gpu.read_alloc(proj.scale).unwrap(),
                    want.1,
                    "{p} r{r} scales"
                );
            };
            check(&e.gate_proj, "gate_proj", true);
            check(&e.up_proj, "up_proj", true);
            check(&e.down_proj, "down_proj", false);
            assert_eq!(
                [e.gate_proj.scale_2, e.up_proj.scale_2, e.down_proj.scale_2],
                [0.5, 0.25, 0.125]
            );
            assert_eq!(e.down_proj.input_scale, Some(2.0));
            assert_eq!(e.gate_proj.input_scale, Some(1.0));
        }
    }
}

/// 2026-10-09: The whole layout still binds through `bind_expert` (here: a resident expert of
/// the store's own pointers, which a sliced bind would have copied).
#[test]
fn whole_shard_binds_the_store_pointers_zero_copy() {
    let gpu = MockGpuBackend::new();
    let mut map = std::collections::HashMap::new();
    for p in ["gate_proj", "up_proj", "down_proj"] {
        for (leaf, dtype, shape, bytes) in [
            ("weight", WeightDtype::UInt8, vec![2, 8], vec![0x21u8; 16]),
            (
                "weight_scale",
                WeightDtype::FP8E4M3,
                vec![2, 1],
                vec![0x38; 2],
            ),
            (
                "weight_scale_2",
                WeightDtype::FP32,
                vec![],
                0.5f32.to_le_bytes().to_vec(),
            ),
        ] {
            let ptr = gpu.alloc(bytes.len()).unwrap();
            gpu.copy_h2d(&bytes, ptr).unwrap();
            map.insert(
                qualified(&format!("mlp.experts.0.{p}.{leaf}")),
                WeightTensor { ptr, shape, dtype },
            );
        }
    }
    let store = WeightStore::from_map(map);
    let e = bind_routed_expert(&gpu, &store, LAYER, 0, ExpertShard::Whole).unwrap();
    let want = store
        .get(&qualified("mlp.experts.0.down_proj.weight"))
        .unwrap()
        .ptr;
    assert_eq!(e.down_proj.packed, want);
    assert_eq!(store.derived().len(), 0);
}

/// 2026-10-09: A BF16 expert (the MTP layer's export form) is quantized whole and sliced: each
/// rank's bytes are the slicer's cut of the whole-expert quantization `bind_expert` produces.
#[test]
fn bf16_expert_slices_its_whole_quantization() {
    let full = 256usize;
    let ramp = |n: usize, salt: f32| -> Vec<u8> {
        (0..n)
            .flat_map(|i| {
                let v = ((i as f32 * 0.37 + salt).sin()) * 3.0;
                half::bf16::from_f32(v).to_le_bytes()
            })
            .collect()
    };
    let t: Vec<_> = [
        ("gate_proj", vec![full, HIDDEN], 0.1),
        ("up_proj", vec![full, HIDDEN], 0.7),
        ("down_proj", vec![HIDDEN, full], 1.3),
    ]
    .into_iter()
    .map(|(p, shape, salt)| {
        (
            qualified(&format!("mlp.experts.2.{p}.weight")),
            WeightDtype::BF16,
            shape,
            ramp(full * HIDDEN, salt),
        )
    })
    .collect();
    let sh = shard("bf16", &t);
    let gpu_w = MockGpuBackend::new();
    let whole = super::bind_expert(&gpu_w, &sh.store, LAYER, 2).unwrap();
    for r in 0..2 {
        let s: ExpertSlice = expert_slice(full, 2, r).unwrap();
        let gpu = MockGpuBackend::new();
        let e = bind_routed_expert(&gpu, &sh.store, LAYER, 2, ExpertShard::Sliced(s)).unwrap();
        for (w, got, rows) in [
            (&whole.gate_proj, &e.gate_proj, true),
            (&whole.down_proj, &e.down_proj, false),
        ] {
            let (wp, ws) = (
                gpu_w.read_alloc(w.packed).unwrap(),
                gpu_w.read_alloc(w.scale).unwrap(),
            );
            let want = if rows {
                slice_expert_rows(&wp, &ws, HIDDEN, &s).unwrap()
            } else {
                slice_expert_cols(&wp, &ws, HIDDEN, &s).unwrap()
            };
            assert_eq!(gpu.read_alloc(got.packed).unwrap(), want.0, "r{r} codes");
            assert_eq!(gpu.read_alloc(got.scale).unwrap(), want.1, "r{r} scales");
            assert_eq!(
                got.scale_2, w.scale_2,
                "r{r}: the whole tensor's global scale"
            );
            assert_eq!(got.input_scale, None);
        }
    }
}

/// 2026-10-09: A packed expert whose gate rows do not match the slice's checkpoint width is
/// refused before anything is read.
#[test]
fn a_packed_expert_of_another_width_is_refused() {
    let t = packed_expert(512);
    let sh = shard("wrong-width", &t);
    let s = expert_slice(640, 3, 0).unwrap();
    let gpu = MockGpuBackend::new();
    let e = bind_routed_expert(&gpu, &sh.store, LAYER, 7, ExpertShard::Sliced(s)).unwrap_err();
    assert!(e.to_string().contains("does not fit"), "{e}");
}

/// 2026-10-09: The routed-expert family: every `mlp.experts.*` tensor of a numbered layer, and
/// nothing else.
#[test]
fn routed_expert_family() {
    for n in [
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight",
        "model.language_model.layers.45.mlp.experts.287.down_proj.weight_scale_2",
        "model.language_model.layers.10.mlp.experts.5.up_proj.input_scale",
    ] {
        assert!(is_routed_expert_tensor(n), "{n}");
    }
    for n in [
        "model.language_model.layers.3.mlp.shared_experts.gate_proj.weight",
        "model.language_model.layers.3.mlp.gate.weight",
        "model.language_model.layers.x.mlp.experts.0.gate_proj.weight",
        "model.visual.blocks.0.mlp.experts.0.weight",
        "lm_head.weight",
    ] {
        assert!(!is_routed_expert_tensor(n), "{n}");
    }
}

/// 2026-10-09: Under the tp layout the hook defers every routed-expert tensor whatever its
/// dtype; under ep it defers none of the packed ones (the existing rules only).
#[test]
fn the_hook_defers_routed_experts_only_under_tp() {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    let packed = [
        ("mlp.experts.7.down_proj.weight", WeightDtype::UInt8),
        ("mlp.experts.7.down_proj.weight_scale", WeightDtype::FP8E4M3),
        ("mlp.experts.7.down_proj.weight_scale_2", WeightDtype::FP32),
    ];
    let hook = |c: &ModelConfig| {
        super::super::Glm5NextWeightLoader
            .defer_predicate(c)
            .unwrap()
    };
    let ep = hook(&c);
    c.moe_expert_layout = MoeExpertLayout::Tp;
    let tp = hook(&c);
    for (leaf, dtype) in packed {
        let n = format!("model.language_model.layers.5.{leaf}");
        assert!(!ep(&n, dtype), "ep defers {leaf}");
        assert!(tp(&n, dtype), "tp keeps {leaf} resident");
    }
    let shared = "model.language_model.layers.5.mlp.shared_experts.gate_proj.weight";
    assert!(!tp(shared, WeightDtype::BF16));
}
