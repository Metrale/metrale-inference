// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Tests for `load_checkpoint_kv_scales` / `load_kv_scales` against a
//! `WeightStore` whose tensors live in `MockGpuBackend` memory. The mock is the
//! `GpuBackend` seam these loaders already take; its allocations hold real
//! bytes, so the host reads here are the production `copy_d2h` path.

use std::collections::HashMap;

use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_model_weights::weights::{WeightDtype, WeightStore, WeightTensor};

use super::{load_checkpoint_kv_scales, load_kv_scales};

fn bf16_bits(v: f32) -> [u8; 2] {
    ((v.to_bits() >> 16) as u16).to_le_bytes()
}

fn upload(
    gpu: &MockGpuBackend,
    bytes: &[u8],
    shape: Vec<usize>,
    dtype: WeightDtype,
) -> WeightTensor {
    let ptr = gpu.alloc(bytes.len()).expect("alloc");
    gpu.copy_h2d(bytes, ptr).expect("h2d");
    WeightTensor { ptr, shape, dtype }
}

fn bf16_scalar(gpu: &MockGpuBackend, v: f32) -> WeightTensor {
    upload(gpu, &bf16_bits(v), vec![1], WeightDtype::BF16)
}

fn f32_scalar(gpu: &MockGpuBackend, v: f32) -> WeightTensor {
    upload(gpu, &v.to_le_bytes(), vec![], WeightDtype::FP32)
}

/// 2026-09-28: Path B: the count serve logs and what the layers load cannot
/// disagree. Four spellings-per-layer plus one layer without scales: the census
/// counts exactly the layers whose loader returns checkpoint values, and every
/// value is the one in the store (the unsloth BF16 `self_attn.k_scale` layers
/// used to come back as 1.0 while being counted).
#[test]
fn the_census_count_equals_the_layers_that_load_checkpoint_scales() {
    let gpu = MockGpuBackend::new();
    let p = |i: usize| format!("model.language_model.layers.{i}.self_attn");
    let mut map = HashMap::new();
    let mut expected = HashMap::new();
    for (i, (k_suffix, v_suffix), k, v, bf16) in [
        (3, ("k_scale", "v_scale"), 0.02746582, 0.02453613, true),
        (7, ("k_scale", "v_scale"), 0.0324707, 0.02478027, true),
        (
            11,
            ("k_proj.k_scale", "v_proj.v_scale"),
            0.0260881,
            0.2742745,
            false,
        ),
        (15, ("attn.k_scale", "attn.v_scale"), 0.05, 0.25, false),
    ] {
        let mk = |x: f32| {
            if bf16 {
                bf16_scalar(&gpu, x)
            } else {
                f32_scalar(&gpu, x)
            }
        };
        map.insert(format!("{}.{k_suffix}", p(i)), mk(k));
        map.insert(format!("{}.{v_suffix}", p(i)), mk(v));
        let round = |x: f32| {
            if bf16 {
                f32::from_bits(x.to_bits() & 0xFFFF_0000)
            } else {
                x
            }
        };
        expected.insert(p(i), (round(k), round(v)));
    }
    map.insert(format!("{}.q_proj.weight", p(19)), bf16_scalar(&gpu, 0.0));
    let store = WeightStore::from_map(map);

    let census = store.kv_scale_census().expect("census");
    let mut loaded_from_checkpoint = 0;
    for i in [3, 7, 11, 15, 19] {
        let got = load_checkpoint_kv_scales(&store, &p(i), &gpu).expect("loads");
        assert_eq!(
            got.is_some(),
            census.layers.contains_key(&p(i)),
            "layer {i}"
        );
        assert_eq!(got, expected.get(&p(i)).copied(), "layer {i}");
        loaded_from_checkpoint += usize::from(got.is_some());
    }
    assert_eq!(census.len(), loaded_from_checkpoint);
    assert_eq!(census.len(), 4);
    assert_eq!(
        load_kv_scales(&store, &p(19), &gpu).expect("absent is not an error"),
        (1.0, 1.0)
    );
}

/// 2026-09-28: Path C: a scale that is not one finite positive BF16/FP32 number is
/// an error from both entry points. The old loader logged a warning and used
/// 1.0, which is the clipping case serve's own warning describes.
#[test]
fn a_malformed_scale_is_an_error_not_one() {
    let gpu = MockGpuBackend::new();
    let p = "model.layers.3.self_attn";
    let two = |g: &MockGpuBackend| {
        let mut b = bf16_bits(0.03).to_vec();
        b.extend(bf16_bits(0.04));
        upload(g, &b, vec![2], WeightDtype::BF16)
    };
    let cases: [(&str, fn(&MockGpuBackend) -> WeightTensor); 5] = [
        ("shape [2]", two),
        ("FP8 dtype", |g| {
            upload(g, &[0x38], vec![1], WeightDtype::FP8E4M3)
        }),
        ("zero", |g| f32_scalar(g, 0.0)),
        ("negative", |g| f32_scalar(g, -0.03)),
        ("NaN", |g| bf16_scalar(g, f32::NAN)),
    ];
    for (what, bad_k) in cases {
        let mut map = HashMap::new();
        map.insert(format!("{p}.k_scale"), bad_k(&gpu));
        map.insert(format!("{p}.v_scale"), bf16_scalar(&gpu, 0.02));
        let store = WeightStore::from_map(map);
        let err = load_checkpoint_kv_scales(&store, p, &gpu).unwrap_err();
        assert!(
            format!("{err:#}").contains(&format!("{p}.k_scale")),
            "{what}: {err:#}"
        );
        assert!(
            load_kv_scales(&store, p, &gpu).is_err(),
            "{what}: no 1.0 fallback"
        );
    }
}

/// 2026-09-28: Path C: a half pair or two spellings stop the load; the layer
/// does not fall back to 1.0.
#[test]
fn an_incomplete_or_ambiguous_pair_stops_the_load() {
    let gpu = MockGpuBackend::new();
    let p = "model.layers.3.self_attn";
    let half = WeightStore::from_map(HashMap::from([(
        format!("{p}.k_scale"),
        bf16_scalar(&gpu, 0.03),
    )]));
    assert!(load_kv_scales(&half, p, &gpu).is_err());
    let both = WeightStore::from_map(HashMap::from([
        (format!("{p}.k_scale"), bf16_scalar(&gpu, 0.03)),
        (format!("{p}.v_scale"), bf16_scalar(&gpu, 0.03)),
        (format!("{p}.k_proj.k_scale"), f32_scalar(&gpu, 0.03)),
        (format!("{p}.v_proj.v_scale"), f32_scalar(&gpu, 0.03)),
    ]));
    assert!(load_kv_scales(&both, p, &gpu).is_err());
}
