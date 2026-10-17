// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Tests of csf-encoded routed-expert scales on the mock device: a csf expert binds
//! the bytes its native twin binds, sliced on every rank and whole; a corrupted plane changes
//! exactly the corrupted scale; incomplete, doubled or mistyped planes are refused; the defer
//! hook keeps the components off the device and nothing else new.
//!
//! The planes are built by a test encoder written from the format text (block order, packed
//! fields, exception records, one index page), independent of the decoder's tile walk.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{DeferredTensor, WeightDtype, WeightStore, WeightTensor};

use super::super::expert_slab::ExpertSlab;
use super::super::expert_tp::bind_routed_expert;
use super::is_csf_component;
use crate::glm5next_mlp::expert_tp::{ExpertShard, expert_slice};
use crate::weight_loader::ModelWeightLoader;

const LAYER: usize = 4;
const HIDDEN: usize = 32;
const ID: usize = 3;

type Tensor = (String, WeightDtype, Vec<usize>, Vec<u8>);

fn qualified(leaf: &str) -> String {
    format!("model.language_model.layers.{LAYER}.{leaf}")
}

/// 2026-10-17: Position-tagged bytes, never zero.
fn tagged(len: usize, salt: usize) -> Vec<u8> {
    (0..len)
        .map(|i| ((i * 29 + salt) % 255 + 1) as u8)
        .collect()
}

/// 2026-10-17: Scale-like bytes: most of a row inside a 16-wide window, every seventh far off.
fn scales(rows: usize, cols: usize, salt: usize) -> Vec<u8> {
    (0..rows * cols)
        .map(|i| {
            let anchor = ((i / cols) * 13 + salt) % 90;
            if i % 7 == 3 {
                ((i * 31 + salt) % 0x7F) as u8
            } else {
                (anchor + (i * 5 + salt) % 16) as u8
            }
        })
        .collect()
}

/// 2026-10-17: The five components (bases, codes, page bases, offsets, exceptions) of an NVFP4
/// `[rows, cols]` plane that fits one index page.
fn encode(logical: &[u8], rows: usize, cols: usize) -> [Vec<u8>; 5] {
    assert!(rows.is_multiple_of(32));
    let (row_blocks, col_blocks) = (rows / 32, cols.div_ceil(2));
    assert!(
        row_blocks.div_ceil(4) * col_blocks.div_ceil(2) <= 32,
        "one page only"
    );
    let bases: Vec<u8> = (0..rows)
        .map(|r| (*logical[r * cols..(r + 1) * cols].iter().min().unwrap()).min(240))
        .collect();
    let (mut fields, mut records, mut prefix) = (Vec::new(), Vec::new(), vec![0u16]);
    for tile_r in (0..row_blocks).step_by(4) {
        for tile_c in (0..col_blocks).step_by(2) {
            for cb in tile_c..(tile_c + 2).min(col_blocks) {
                for rb in tile_r..(tile_r + 4).min(row_blocks) {
                    let width = (cols - 2 * cb).min(2);
                    for lr in 0..32 {
                        let row = rb * 32 + lr;
                        for lc in 0..width {
                            let v = logical[row * cols + 2 * cb + lc];
                            let delta = v.wrapping_sub(bases[row]);
                            if v >= bases[row] && delta < 16 {
                                fields.push(delta);
                            } else {
                                fields.push(0);
                                records.push(((v as u16) << 8) | (2 * lr + lc) as u16);
                            }
                        }
                    }
                    prefix.push(records.len() as u16);
                }
            }
        }
    }
    let codes = fields.chunks(2).map(|f| f[0] | (f[1] << 4)).collect();
    let le16 = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    [
        bases,
        codes,
        0u32.to_le_bytes().to_vec(),
        le16(&prefix),
        le16(&records),
    ]
}

const SUFFIXES: [(&str, WeightDtype); 5] = [
    ("bases", WeightDtype::UInt8),
    ("codes", WeightDtype::UInt8),
    ("page_bases", WeightDtype::UInt32),
    ("offsets", WeightDtype::UInt16),
    ("exceptions", WeightDtype::UInt16),
];

/// 2026-10-17: Expert `ID` of width `full`, its block scales native or as a csf plane.
fn expert(full: usize, csf: bool) -> Vec<Tensor> {
    let mut t = Vec::new();
    for (i, (p, n, k)) in [
        ("gate_proj", full, HIDDEN),
        ("up_proj", full, HIDDEN),
        ("down_proj", HIDDEN, full),
    ]
    .into_iter()
    .enumerate()
    {
        let w = |leaf: &str| qualified(&format!("mlp.experts.{ID}.{p}.{leaf}"));
        t.push((
            w("weight"),
            WeightDtype::UInt8,
            vec![n, k / 2],
            tagged(n * k / 2, i),
        ));
        let s = scales(n, k / 16, 40 * i + 1);
        if csf {
            for ((suffix, dtype), bytes) in SUFFIXES.iter().zip(encode(&s, n, k / 16)) {
                let elems = bytes.len() / dtype.byte_size();
                t.push((
                    w(&format!("weight_scale.csf32_{suffix}")),
                    *dtype,
                    vec![elems],
                    bytes,
                ));
            }
        } else {
            t.push((w("weight_scale"), WeightDtype::FP8E4M3, vec![n, k / 16], s));
        }
        let s2 = [0.5f32, 0.25, 0.125][i].to_le_bytes().to_vec();
        t.push((w("weight_scale_2"), WeightDtype::FP32, vec![], s2));
    }
    t
}

/// 2026-10-17: A store with every tensor deferred to one scratch shard, plus the tensors
/// `resident(name)` picks uploaded to `gpu`.
struct Shard {
    store: WeightStore,
    path: std::path::PathBuf,
}

impl Drop for Shard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn shard(tag: &str, t: &[Tensor], gpu: &MockGpuBackend, resident: impl Fn(&str) -> bool) -> Shard {
    let dir = std::env::temp_dir().join(format!("metrale-glm5next-csf-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{tag}.safetensors"));
    let (mut blob, mut map, mut deferred) = (
        vec![0x5Au8; 13],
        std::collections::HashMap::new(),
        Vec::new(),
    );
    for (name, dtype, shape, bytes) in t {
        if resident(name) {
            let ptr = gpu.alloc(bytes.len().max(1)).unwrap();
            gpu.copy_h2d(bytes, ptr).unwrap();
            let (shape, dtype) = (shape.clone(), *dtype);
            map.insert(name.clone(), WeightTensor { ptr, shape, dtype });
        } else {
            let (path, offset) = (path.clone(), blob.len() as u64);
            let (shape, dtype) = (shape.clone(), *dtype);
            deferred.push((
                name.clone(),
                DeferredTensor {
                    path,
                    offset,
                    shape,
                    dtype,
                },
            ));
            blob.extend_from_slice(bytes);
        }
    }
    std::fs::write(&path, &blob).unwrap();
    let mut store = WeightStore::from_map(map);
    for (name, d) in deferred {
        store.defer(name, d);
    }
    Shard { store, path }
}

fn read(gpu: &MockGpuBackend, p: DevicePtr, len: usize) -> Vec<u8> {
    let mut v = vec![0u8; len];
    gpu.copy_d2h(p, &mut v).unwrap();
    v
}

/// 2026-10-17: (codes, scales, scale_2) of each projection as bound under `shard`.
fn bind(
    store: &WeightStore,
    gpu: &MockGpuBackend,
    shard: ExpertShard,
) -> Vec<(Vec<u8>, Vec<u8>, f32)> {
    let slab = ExpertSlab::new(ID..ID + 1);
    let e = bind_routed_expert(gpu, store, LAYER, ID, shard, &slab).unwrap();
    let (rows, cols) = match shard {
        ExpertShard::Sliced(s) => (s.len, s.len),
        ExpertShard::Whole => (store_width(store), store_width(store)),
    };
    [
        (&e.gate_proj, rows, HIDDEN),
        (&e.up_proj, rows, HIDDEN),
        (&e.down_proj, HIDDEN, cols),
    ]
    .into_iter()
    .map(|(p, n, k)| {
        (
            read(gpu, p.packed, n * k / 2),
            read(gpu, p.scale, n * k / 16),
            p.scale_2,
        )
    })
    .collect()
}

fn store_width(store: &WeightStore) -> usize {
    let name = qualified(&format!("mlp.experts.{ID}.gate_proj.weight"));
    match store.deferred(&name) {
        Some(d) => d.shape[0],
        None => store.get(&name).unwrap().shape[0],
    }
}

/// 2026-10-17: Sliced on every rank (an even and an uneven split), a csf expert binds the codes,
/// scales and global scales its native twin binds.
#[test]
fn a_csf_expert_binds_its_native_twins_slices() {
    for (full, tp) in [(256usize, 2usize), (640, 3)] {
        let gpu = MockGpuBackend::new();
        let native = shard(
            &format!("native-{full}"),
            &expert(full, false),
            &gpu,
            |_| false,
        );
        let csf = shard(&format!("csf-{full}"), &expert(full, true), &gpu, |_| false);
        for r in 0..tp {
            let s = expert_slice(full, tp, r).unwrap();
            let want = bind(&native.store, &gpu, ExpertShard::Sliced(s));
            let got = bind(&csf.store, &gpu, ExpertShard::Sliced(s));
            assert_eq!(got, want, "full {full} rank {r}/{tp}");
        }
    }
}

/// 2026-10-17: Whole (the EP layout): resident codes stay zero-copy, the deferred plane is
/// decoded and uploaded as the native scales.
#[test]
fn a_whole_csf_expert_uploads_the_native_scales() {
    let gpu = MockGpuBackend::new();
    let full = 128;
    let resident = |name: &str| !is_csf_component(name);
    let native = shard("whole-native", &expert(full, false), &gpu, resident);
    let csf = shard("whole-csf", &expert(full, true), &gpu, resident);
    assert_eq!(
        bind(&csf.store, &gpu, ExpertShard::Whole),
        bind(&native.store, &gpu, ExpertShard::Whole)
    );
    let slab = ExpertSlab::new(ID..ID + 1);
    let e = bind_routed_expert(&gpu, &csf.store, LAYER, ID, ExpertShard::Whole, &slab).unwrap();
    let codes = qualified(&format!("mlp.experts.{ID}.down_proj.weight"));
    assert_eq!(e.down_proj.packed, csf.store.get(&codes).unwrap().ptr);
}

/// 2026-10-17: Negative control for the two tests above: one flipped code nibble moves exactly
/// one bound scale, so the comparisons see the decoded plane, not a fallback.
#[test]
fn a_flipped_code_moves_exactly_one_scale() {
    let gpu = MockGpuBackend::new();
    let full = 128;
    let mut t = expert(full, true);
    let codes = qualified(&format!(
        "mlp.experts.{ID}.gate_proj.weight_scale.csf32_codes"
    ));
    let c = t.iter_mut().find(|x| x.0 == codes).unwrap();
    // 2026-10-17: Field 12 (row 6, column 0) is in range, not an exception record.
    c.3[6] ^= 0x01;
    let resident = |name: &str| !is_csf_component(name);
    let native = shard("flip-native", &expert(full, false), &gpu, resident);
    let csf = shard("flip-csf", &t, &gpu, resident);
    let want = bind(&native.store, &gpu, ExpertShard::Whole);
    let got = bind(&csf.store, &gpu, ExpertShard::Whole);
    let moved = (0..want[0].1.len())
        .filter(|&i| want[0].1[i] != got[0].1[i])
        .count();
    assert_eq!(moved, 1, "gate scales");
    assert_eq!(
        (&got[1], &got[2]),
        (&want[1], &want[2]),
        "up and down untouched"
    );
}

/// 2026-10-17: An incomplete plane, a plane beside a native scale, and a mistyped component are
/// each refused with the reason, under both layouts.
#[test]
fn malformed_planes_are_refused() {
    let full = 128;
    let leaf = |l: &str| qualified(&format!("mlp.experts.{ID}.up_proj.{l}"));
    let without = |name: String| -> Vec<Tensor> {
        expert(full, true)
            .into_iter()
            .filter(|x| x.0 != name)
            .collect()
    };
    let mut doubled = expert(full, true);
    let native_scale = expert(full, false)
        .into_iter()
        .find(|x| x.0 == leaf("weight_scale"));
    doubled.push(native_scale.unwrap());
    let mut mistyped = expert(full, true);
    let off = mistyped
        .iter_mut()
        .find(|x| x.0 == leaf("weight_scale.csf32_offsets"))
        .unwrap();
    (off.1, off.2) = (WeightDtype::UInt8, vec![off.3.len()]);
    for (tag, t, why) in [
        (
            "incomplete",
            without(leaf("weight_scale.csf32_exceptions")),
            "incomplete",
        ),
        ("doubled", doubled, "present beside"),
        ("mistyped", mistyped, "expected 1-D UInt16"),
    ] {
        let gpu = MockGpuBackend::new();
        let resident = |name: &str| !is_csf_component(name);
        for (layout, store_resident, shard_kind) in [
            (
                "sliced",
                false,
                ExpertShard::Sliced(expert_slice(full, 2, 0).unwrap()),
            ),
            ("whole", true, ExpertShard::Whole),
        ] {
            let sh = shard(&format!("{tag}-{layout}"), &t, &gpu, |n| {
                store_resident && resident(n)
            });
            let slab = ExpertSlab::new(ID..ID + 1);
            let err = bind_routed_expert(&gpu, &sh.store, LAYER, ID, shard_kind, &slab)
                .err()
                .unwrap_or_else(|| panic!("{tag} {layout} bound"));
            assert!(format!("{err:#}").contains(why), "{tag} {layout}: {err:#}");
        }
    }
}

/// 2026-10-17: The defer hook keeps all five components off the device under the default
/// layout, and not the native scale, its global scale or the codes.
#[test]
fn the_defer_hook_keeps_csf_components_off_the_device() {
    let hook = super::super::Glm5NextWeightLoader
        .defer_predicate(&ModelConfig::qwen3_next_80b_nvfp4())
        .expect("glm5_next declares a defer predicate");
    let leaf = |l: &str| qualified(&format!("mlp.experts.{ID}.down_proj.{l}"));
    for (suffix, dtype) in SUFFIXES {
        let name = leaf(&format!("weight_scale.csf32_{suffix}"));
        assert!(hook(&name, dtype), "{name}");
    }
    for (name, dtype) in [
        (leaf("weight_scale"), WeightDtype::FP8E4M3),
        (leaf("weight_scale_2"), WeightDtype::FP32),
        (leaf("weight"), WeightDtype::UInt8),
    ] {
        assert!(!hook(&name, dtype), "{name}");
    }
}
