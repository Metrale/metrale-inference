// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Tests of the 8-bit head on the mock device: the defer hook withholds exactly the
//! F8_E4M3 head and its scales; a deferred head binds its BF16 widening (MX and block-128
//! scales, checked against a reference written from the format text) and nothing else reaches
//! the device; a resident pair widens to the same bytes; a 16-bit head binds the store tensor
//! itself; a head without scales, or deferred at another width, is refused.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use std::collections::HashMap;

use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_model_weights::weights::{DeferredTensor, WeightDtype, WeightStore, WeightTensor};

use super::{HEAD, HEAD_SCALE, head_weight, is_fp8_head_tensor};
use crate::weight_loader::ModelWeightLoader;

const VOCAB: usize = 3;
const HIDDEN: usize = 64;

type Tensor = (&'static str, WeightDtype, Vec<usize>, Vec<u8>);

/// 2026-10-17: E4M3 from the format text: bias 7, 3 mantissa bits, `S.1111.111` is NaN.
fn e4m3_ref(b: u8) -> f64 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let (e, m) = (i32::from((b >> 3) & 0xF), f64::from(b & 7));
    assert_ne!(b & 0x7F, 0x7F, "the fixtures hold no NaN byte");
    match e {
        0 => sign * m / 8.0 * 2f64.powi(-6),
        _ => sign * (1.0 + m / 8.0) * 2f64.powi(e - 7),
    }
}

/// 2026-10-17: Distinct finite E4M3 bytes of both signs, never `0x7F`/`0xFF`.
fn weight_bytes() -> Vec<u8> {
    (0..VOCAB * HIDDEN)
        .map(|i| {
            let b = (i * 37 % 0x7F) as u8;
            if i % 3 == 0 { b | 0x80 } else { b }
        })
        .collect()
}

/// 2026-10-17: The expected BF16 bit patterns of `w` under `scale(r, c)`.
fn reference(w: &[u8], scale: impl Fn(usize, usize) -> f64) -> Vec<u8> {
    w.iter()
        .enumerate()
        .flat_map(|(i, &b)| {
            let v = e4m3_ref(b) * scale(i / HIDDEN, i % HIDDEN);
            half::bf16::from_f64(v).to_bits().to_le_bytes()
        })
        .collect()
}

/// 2026-10-17: One UE8M0 exponent per 32 weights of a row: `[VOCAB, HIDDEN / 32]`.
fn mx_scale() -> (Vec<u8>, impl Fn(usize, usize) -> f64) {
    let e = [126u8, 127, 128, 120, 133, 127];
    let at = move |r: usize, c: usize| 2f64.powi(i32::from(e[r * 2 + c / 32]) - 127);
    (e.to_vec(), at)
}

struct Shard {
    store: WeightStore,
    path: std::path::PathBuf,
}

impl Drop for Shard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// 2026-10-17: A store holding `t`: the names `deferred(name)` picks deferred to one scratch
/// shard, the rest uploaded to `gpu`.
fn shard(tag: &str, t: &[Tensor], gpu: &MockGpuBackend, deferred: impl Fn(&str) -> bool) -> Shard {
    let dir = std::env::temp_dir().join(format!("metrale-glm5next-head-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{tag}.safetensors"));
    let (mut blob, mut map, mut later) = (vec![0xA5u8; 11], HashMap::new(), Vec::new());
    for (name, dtype, shape, bytes) in t {
        let (shape, dtype) = (shape.clone(), *dtype);
        if deferred(name) {
            let offset = blob.len() as u64;
            let path = path.clone();
            later.push((
                name.to_string(),
                DeferredTensor {
                    path,
                    offset,
                    shape,
                    dtype,
                },
            ));
            blob.extend_from_slice(bytes);
        } else {
            let ptr = gpu.alloc(bytes.len().max(1)).unwrap();
            gpu.copy_h2d(bytes, ptr).unwrap();
            map.insert(name.to_string(), WeightTensor { ptr, shape, dtype });
        }
    }
    std::fs::write(&path, &blob).unwrap();
    let mut store = WeightStore::from_map(map);
    for (name, d) in later {
        store.defer(name, d);
    }
    Shard { store, path }
}

fn fp8_pair(scale_dtype: WeightDtype, scale_shape: Vec<usize>, scale: Vec<u8>) -> Vec<Tensor> {
    vec![
        (
            HEAD,
            WeightDtype::FP8E4M3,
            vec![VOCAB, HIDDEN],
            weight_bytes(),
        ),
        (HEAD_SCALE, scale_dtype, scale_shape, scale),
    ]
}

fn hook() -> metrale_model_weights::weights::DeferHook {
    super::super::Glm5NextWeightLoader
        .defer_predicate(&ModelConfig::qwen3_next_80b_nvfp4())
        .expect("glm5_next declares a defer predicate")
}

/// 2026-10-17: The loader's hook withholds the F8_E4M3 head and the head scales, and still
/// uploads a 16-bit head and every other F8_E4M3 tensor; the negative controls fail if the
/// predicate keyed on the dtype or the suffix alone.
#[test]
fn the_hook_defers_the_8bit_head_and_its_scales_only() {
    let hook = hook();
    assert!(hook(HEAD, WeightDtype::FP8E4M3));
    assert!(hook(HEAD_SCALE, WeightDtype::UInt8));
    assert!(hook(HEAD_SCALE, WeightDtype::FP32));
    assert!(is_fp8_head_tensor(HEAD, WeightDtype::FP8E4M3));
    for dtype in [WeightDtype::BF16, WeightDtype::FP32, WeightDtype::UInt8] {
        assert!(!hook(HEAD, dtype), "{dtype:?} head");
    }
    for name in [
        "model.lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.3.self_attn.o_proj.weight",
        "model.language_model.layers.3.self_attn.o_proj.weight_scale",
    ] {
        assert!(!hook(name, WeightDtype::FP8E4M3), "{name}");
    }
}

/// 2026-10-17: A deferred MX head binds the reference widening; the only device allocation is
/// the BF16 head, owned by the store's derived buffers.
#[test]
fn a_deferred_mx_head_binds_its_bf16_widening() {
    let gpu = MockGpuBackend::new();
    let (e, at) = mx_scale();
    let t = fp8_pair(WeightDtype::UInt8, vec![VOCAB, HIDDEN / 32], e);
    let s = shard("mx", &t, &gpu, |n| {
        hook()(n, t.iter().find(|x| x.0 == n).unwrap().1)
    });
    assert!(s.store.get(HEAD).is_err() && s.store.get(HEAD_SCALE).is_err());

    let head = head_weight(&gpu, &s.store).unwrap();
    let want = reference(&weight_bytes(), at);
    assert_eq!(gpu.read_alloc(head.weight).unwrap(), want);
    assert_eq!(s.store.derived().len(), 1);
    assert_eq!(s.store.derived().bytes(), VOCAB * HIDDEN * 2);
    // 2026-10-17: Positive control: the scales moved the result (unit scales would not match).
    assert_ne!(want, reference(&weight_bytes(), |_, _| 1.0));
}

/// 2026-10-17: A block-128 head (one F32 per tile; the tile covers the whole small head) binds
/// the reference widening in both stored scale shapes.
#[test]
fn a_deferred_block_128_head_binds_its_bf16_widening() {
    for shape in [vec![1, 1, 1, 1], vec![1, 1]] {
        let gpu = MockGpuBackend::new();
        let t = fp8_pair(
            WeightDtype::FP32,
            shape.clone(),
            0.375f32.to_le_bytes().to_vec(),
        );
        let s = shard("tile", &t, &gpu, |_| true);
        let head = head_weight(&gpu, &s.store).unwrap();
        let want = reference(&weight_bytes(), |_, _| 0.375);
        assert_eq!(gpu.read_alloc(head.weight).unwrap(), want, "{shape:?}");
    }
}

/// 2026-10-17: A loader that ignored the hook leaves the pair resident; it widens to the same
/// bytes as the deferred pair.
#[test]
fn a_resident_8bit_pair_widens_to_the_deferred_bytes() {
    let (e, _) = mx_scale();
    let t = fp8_pair(WeightDtype::UInt8, vec![VOCAB, HIDDEN / 32], e);
    let (ga, gb) = (MockGpuBackend::new(), MockGpuBackend::new());
    let (a, b) = (
        shard("res-a", &t, &ga, |_| false),
        shard("res-b", &t, &gb, |_| true),
    );
    let ha = head_weight(&ga, &a.store).unwrap();
    let hb = head_weight(&gb, &b.store).unwrap();
    assert_eq!(
        ga.read_alloc(ha.weight).unwrap(),
        gb.read_alloc(hb.weight).unwrap()
    );
}

/// 2026-10-17: A 16-bit head binds the store's own tensor: same pointer, nothing adopted, so
/// the current recipe's head is untouched.
#[test]
fn a_16bit_head_binds_the_store_tensor_itself() {
    let gpu = MockGpuBackend::new();
    let bf16: Vec<u8> = (0..VOCAB * HIDDEN * 2)
        .map(|i| (i * 7 % 251) as u8)
        .collect();
    let t = vec![(HEAD, WeightDtype::BF16, vec![VOCAB, HIDDEN], bf16.clone())];
    let s = shard("bf16", &t, &gpu, |n| hook()(n, WeightDtype::BF16));
    let head = head_weight(&gpu, &s.store).unwrap();
    assert_eq!(head.weight, s.store.get(HEAD).unwrap().ptr);
    assert_eq!(gpu.read_alloc(head.weight).unwrap(), bf16);
    assert!(s.store.derived().is_empty());
}

/// 2026-10-17: An 8-bit head without scales, a deferred 16-bit head, and a mis-shaped scale
/// are refused rather than bound as raw bytes.
#[test]
fn an_unscaled_or_misdeclared_head_is_refused() {
    let gpu = MockGpuBackend::new();
    let lone = vec![(
        HEAD,
        WeightDtype::FP8E4M3,
        vec![VOCAB, HIDDEN],
        weight_bytes(),
    )];
    let s = shard("lone", &lone, &gpu, |_| true);
    let err = format!("{:#}", head_weight(&gpu, &s.store).unwrap_err());
    assert!(err.contains(HEAD_SCALE), "{err}");

    let wide = vec![(
        HEAD,
        WeightDtype::BF16,
        vec![VOCAB, HIDDEN],
        vec![0x11; VOCAB * HIDDEN * 2],
    )];
    let s = shard("wide", &wide, &gpu, |_| true);
    let err = format!("{:#}", head_weight(&gpu, &s.store).unwrap_err());
    assert!(err.contains("deferred"), "{err}");

    let t = fp8_pair(
        WeightDtype::UInt8,
        vec![VOCAB, HIDDEN / 16],
        vec![127; VOCAB * 4],
    );
    let s = shard("per16", &t, &gpu, |_| true);
    assert!(head_weight(&gpu, &s.store).is_err());
    assert!(
        s.store.derived().is_empty(),
        "a refused head adopts nothing"
    );
}
