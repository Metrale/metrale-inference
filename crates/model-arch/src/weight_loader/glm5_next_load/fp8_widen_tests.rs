// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The E4M3 widening against a reference written from the formats' definitions
//! (OCP E4M3: bias 7, subnormals at exponent 0, `S.1111.111` NaN; UE8M0: `2^(e - 127)`, `0xFF`
//! NaN), not from the cache crate's table the widening reads, and the layer pass through
//! `LayerSource::collect` over a real store.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_kda::binding::{KdaDtype, KdaTensorSource};
use crate::weight_loader::glm5_next_load::LayerSource;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_model_weights::weights::{WeightStore, WeightTensor};

/// 2026-10-17: One E4M3 byte by its definition; `None` for the two NaN encodings.
fn e4m3_ref(b: u8) -> Option<f64> {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let (e, m) = (i32::from((b >> 3) & 0xF), f64::from(b & 7));
    match (e, b & 0x7F) {
        (_, 0x7F) => None,
        (0, _) => Some(sign * m / 8.0 * 2f64.powi(-6)),
        _ => Some(sign * (1.0 + m / 8.0) * 2f64.powi(e - 7)),
    }
}

fn bf16_bits(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn want_bits(v: f64) -> u16 {
    half::bf16::from_f64(v).to_bits()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// 2026-10-17: Every finite E4M3 byte (zeros of both signs, the subnormal `0x01`, the maxima
/// `0x7E`/`0xFE`) under UE8M0 exponents from the subnormal `e = 0` to the largest one that keeps
/// `448 * 2^(e - 127)` finite, one exponent per 32-wide block: each output is the reference
/// rounded once to BF16, and each block reads its own exponent.
#[test]
fn every_finite_e4m3_byte_widens_exactly_under_its_block_exponent() {
    let exps: [u8; 8] = [0, 1, 64, 126, 127, 128, 200, 246];
    let finite: Vec<u8> = (0..=255u8).filter(|b| e4m3_ref(*b).is_some()).collect();
    assert_eq!(finite.len(), 254);
    let k = 256;
    let n = exps.len();
    let mut weight = vec![0u8; n * k];
    let mut scale = vec![0u8; n * (k / MX_BLOCK)];
    for r in 0..n {
        for c in 0..k {
            weight[r * k + c] = finite[(r * 37 + c) % finite.len()];
        }
        for blk in 0..k / MX_BLOCK {
            scale[r * (k / MX_BLOCK) + blk] = exps[(r + blk) % exps.len()];
        }
    }
    let out = widen_fp8_to_bf16(
        "t",
        &weight,
        &[n, k],
        WeightDtype::UInt8,
        &[n, k / MX_BLOCK],
        &scale,
    )
    .unwrap();
    let got = bf16_bits(&out);
    for r in 0..n {
        for c in 0..k {
            let e = scale[r * (k / MX_BLOCK) + c / MX_BLOCK];
            let v = e4m3_ref(weight[r * k + c]).unwrap() * 2f64.powi(i32::from(e) - 127);
            assert_eq!(got[r * k + c], want_bits(v), "[{r}, {c}] e={e}");
        }
    }
    // 2026-10-17: The spec's landmarks, spelled out once.
    let one = |w: u8, e: u8| {
        let mut row = vec![0u8; 32];
        row[0] = w;
        let b = widen_fp8_to_bf16("t", &row, &[1, 32], WeightDtype::FP8E8M0, &[1, 1], &[e]);
        half::bf16::from_bits(bf16_bits(&b.unwrap())[0]).to_f32()
    };
    assert_eq!(one(0x00, 127).to_bits(), 0.0f32.to_bits());
    assert_eq!(one(0x80, 127).to_bits(), (-0.0f32).to_bits());
    assert_eq!(one(0x7E, 127), 448.0);
    assert_eq!(one(0xFE, 127), -448.0);
    assert_eq!(one(0x01, 127), 2f32.powi(-9));
    assert_eq!(one(0x38, 0), 2f32.powi(-127));
    assert_eq!(one(0x38, 254), 2f32.powi(127));
}

/// 2026-10-17: A 130 x 260 weight under one F32 per 128 x 128 tile, in both stored scale shapes:
/// the ragged last tile row and column read their own scale on either side of each edge.
#[test]
fn block_128_tiles_cover_ragged_edges_in_both_scale_shapes() {
    let (n, k) = (130, 260);
    let tiles: Vec<f32> = (0..6).map(|i| 2f32.powi(i * 3 - 7) * 1.5).collect();
    let weight = vec![0x40u8; n * k]; // 2.0
    for shape in [vec![2, 1, 3, 1], vec![2, 3]] {
        let out = widen_fp8_to_bf16(
            "t",
            &weight,
            &[n, k],
            WeightDtype::FP32,
            &shape,
            &f32_bytes(&tiles),
        )
        .unwrap();
        let got = bf16_bits(&out);
        for r in [0, 127, 128, 129] {
            for c in [0, 127, 128, 255, 256, 259] {
                let want = want_bits(2.0 * f64::from(tiles[(r / 128) * 3 + c / 128]));
                assert_eq!(got[r * k + c], want, "{shape:?} [{r}, {c}]");
            }
        }
    }
}

/// 2026-10-17: A tile scale that is not a power of two rounds once in `f32` and once to BF16.
#[test]
fn a_block_128_product_rounds_to_nearest_even() {
    let s = 1.0f32 + 2f32.powi(-9); // 1.001953125, below BF16 precision
    let out = widen_fp8_to_bf16(
        "t",
        &[0x38],
        &[1, 1],
        WeightDtype::FP32,
        &[1, 1],
        &f32_bytes(&[s]),
    )
    .unwrap();
    assert_eq!(bf16_bits(&out)[0], half::bf16::from_f32(s).to_bits());
    assert_eq!(half::bf16::from_bits(bf16_bits(&out)[0]).to_f32(), 1.0);
}

/// 2026-10-17: NaN bytes, the UE8M0 NaN, an overflowing product, a NaN tile scale and every
/// scale dtype or shape outside the two layouts are refused.
#[test]
fn malformed_or_non_finite_inputs_are_refused() {
    let w = vec![0x38u8; 2 * 64];
    let mx =
        |w: &[u8], s: &[u8]| widen_fp8_to_bf16("t", w, &[2, 64], WeightDtype::UInt8, &[2, 2], s);
    assert!(mx(&w, &[127; 4]).is_ok(), "positive control");
    let mut nan_w = w.clone();
    nan_w[70] = 0x7F;
    assert!(mx(&nan_w, &[127; 4]).is_err(), "0x7F weight");
    nan_w[70] = 0xFF;
    assert!(mx(&nan_w, &[127; 4]).is_err(), "0xFF weight");
    assert!(mx(&w, &[127, 127, 255, 127]).is_err(), "UE8M0 NaN");
    let mut big = w.clone();
    big[0] = 0x7E;
    assert!(
        mx(&big, &[254, 127, 127, 127]).is_err(),
        "448 * 2^127 overflows"
    );
    assert!(
        mx(&big, &[246, 127, 127, 127]).is_ok(),
        "448 * 2^119 does not"
    );
    assert!(mx(&w, &[127; 3]).is_err(), "short scale");
    assert!(mx(&w[..100], &[127; 4]).is_err(), "short weight");
    let bad = |dtype, shape: &[usize], s: &[u8]| {
        widen_fp8_to_bf16("t", &w, &[2, 64], dtype, shape, s).is_err()
    };
    assert!(bad(WeightDtype::UInt8, &[2, 4], &[127; 8]), "per-16 scale");
    assert!(bad(WeightDtype::UInt8, &[4, 2], &[127; 8]), "transposed");
    assert!(bad(WeightDtype::BF16, &[2, 2], &[0; 8]), "BF16 scale");
    assert!(
        bad(WeightDtype::FP8E4M3, &[2, 4], &[0x38; 8]),
        "NVFP4-style scale"
    );
    assert!(
        bad(WeightDtype::FP32, &[2, 1], &f32_bytes(&[1.0; 2])),
        "too many tiles"
    );
    assert!(
        bad(WeightDtype::FP32, &[1, 1], &f32_bytes(&[f32::NAN])),
        "NaN tile"
    );
    assert!(
        !bad(WeightDtype::FP32, &[1, 1], &f32_bytes(&[1.0])),
        "positive control"
    );
    let odd_k = widen_fp8_to_bf16(
        "t",
        &[0x38; 40],
        &[1, 40],
        WeightDtype::UInt8,
        &[1, 1],
        &[127],
    );
    assert!(odd_k.is_err(), "K not a multiple of 32");
    let three_d = widen_fp8_to_bf16("t", &w, &[2, 1, 64], WeightDtype::UInt8, &[2, 2], &[127; 4]);
    assert!(three_d.is_err(), "3-D weight");
}

const LAYER: usize = 3;

fn put(
    gpu: &MockGpuBackend,
    map: &mut HostStore,
    rel: &str,
    dtype: WeightDtype,
    shape: &[usize],
    bytes: &[u8],
) {
    let p = gpu.alloc(bytes.len().max(1)).unwrap();
    gpu.copy_h2d(bytes, p).unwrap();
    let shape = shape.to_vec();
    let name = format!("model.language_model.layers.{LAYER}.{rel}");
    map.insert(
        name,
        WeightTensor {
            ptr: p,
            shape,
            dtype,
        },
    );
}

type HostStore = std::collections::HashMap<String, WeightTensor>;

/// 2026-10-17: Through `LayerSource::collect` over a real store: a block-128 attention
/// projection and an MX dense projection reach the binders as BF16 with their scales gone from
/// the names and from `get`, while a packed-NVFP4 pair, a BF16 norm and an E4M3 tensor without a
/// scale keep their stored bytes.
#[test]
fn collect_widens_scaled_e4m3_pairs_and_leaves_everything_else() {
    let gpu = MockGpuBackend::new();
    let mut map = HostStore::new();
    let attn_w: Vec<u8> = (0..128 * 256)
        .map(|i| [0x38u8, 0xC0, 0x01, 0x7E][i % 4])
        .collect();
    let attn_s = f32_bytes(&[0.5, 4.0]);
    put(
        &gpu,
        &mut map,
        "self_attn.o_proj.weight",
        WeightDtype::FP8E4M3,
        &[128, 256],
        &attn_w,
    );
    put(
        &gpu,
        &mut map,
        "self_attn.o_proj.weight_scale",
        WeightDtype::FP32,
        &[1, 1, 2, 1],
        &attn_s,
    );
    let mlp_w = vec![0x30u8; 4 * 64];
    put(
        &gpu,
        &mut map,
        "mlp.gate_proj.weight",
        WeightDtype::FP8E4M3,
        &[4, 64],
        &mlp_w,
    );
    put(
        &gpu,
        &mut map,
        "mlp.gate_proj.weight_scale",
        WeightDtype::UInt8,
        &[4, 2],
        &[128; 8],
    );
    let packed = vec![0x21u8; 4 * 8];
    put(
        &gpu,
        &mut map,
        "mlp.up_proj.weight",
        WeightDtype::UInt8,
        &[4, 8],
        &packed,
    );
    put(
        &gpu,
        &mut map,
        "mlp.up_proj.weight_scale",
        WeightDtype::FP8E4M3,
        &[4, 1],
        &[0x38; 4],
    );
    let norm = half::bf16::from_f32(1.25).to_le_bytes().repeat(8);
    put(
        &gpu,
        &mut map,
        "input_layernorm.weight",
        WeightDtype::BF16,
        &[8],
        &norm,
    );
    put(
        &gpu,
        &mut map,
        "mlp.lone.weight",
        WeightDtype::FP8E4M3,
        &[2, 2],
        &[0x38; 4],
    );
    let store = WeightStore::from_map(map);
    let src = LayerSource::collect(&gpu, &store, LAYER).unwrap();

    let names = src.names();
    for gone in [
        "self_attn.o_proj.weight_scale",
        "mlp.gate_proj.weight_scale",
    ] {
        assert!(!names.iter().any(|n| n == gone), "{gone} still listed");
        assert!(src.raw(gone).is_err(), "{gone} still held");
    }
    assert!(
        names.iter().any(|n| n == "mlp.up_proj.weight_scale"),
        "NVFP4 scale kept"
    );

    let o = src.get("self_attn.o_proj.weight").expect("o_proj");
    assert_eq!(o.dtype, KdaDtype::Bf16);
    assert_eq!(o.shape, vec![128, 256]);
    let want = widen_fp8_to_bf16(
        "o",
        &attn_w,
        &[128, 256],
        WeightDtype::FP32,
        &[1, 1, 2, 1],
        &attn_s,
    );
    assert_eq!(o.bytes, want.unwrap().as_slice());
    assert!(src.get("self_attn.o_proj.weight_scale").is_none());
    // 2026-10-17: `0x30` is 0.5; the block exponent 128 doubles it.
    assert_eq!(src.f32("mlp.gate_proj.weight").unwrap(), vec![1.0; 4 * 64]);

    assert_eq!(
        src.raw("mlp.up_proj.weight").unwrap(),
        (WeightDtype::UInt8, vec![4, 8], packed)
    );
    assert_eq!(src.raw("input_layernorm.weight").unwrap().2, norm);
    assert_eq!(src.raw("mlp.lone.weight").unwrap().0, WeightDtype::FP8E4M3);
}

/// 2026-10-17: A layer with no scaled E4M3 pair (the 16-bit and NVFP4 exports) passes through
/// `widen_layer_fp8` byte for byte and consumes nothing; one malformed pair fails the layer.
#[test]
fn the_layer_pass_is_an_identity_without_pairs_and_fails_on_a_bad_one() {
    let mut t = HostTensors::new();
    t.insert(
        "a.weight".into(),
        (WeightDtype::BF16, vec![2], vec![1, 2, 3, 4]),
    );
    t.insert(
        "b.weight".into(),
        (WeightDtype::UInt8, vec![2, 8], vec![7; 16]),
    );
    t.insert(
        "b.weight_scale".into(),
        (WeightDtype::FP8E4M3, vec![2, 1], vec![0x38; 2]),
    );
    t.insert("c".into(), (WeightDtype::FP32, vec![1], f32_bytes(&[3.0])));
    let before = t.clone();
    assert!(widen_layer_fp8(&mut t).unwrap().is_empty());
    assert_eq!(t, before);

    t.insert(
        "d.weight".into(),
        (WeightDtype::FP8E4M3, vec![1, 32], vec![0x38; 32]),
    );
    t.insert(
        "d.weight_scale".into(),
        (WeightDtype::UInt8, vec![1, 2], vec![127; 2]),
    );
    assert!(
        widen_layer_fp8(&mut t.clone()).is_err(),
        "a per-16 UE8M0 scale is refused"
    );
    t.insert(
        "d.weight_scale".into(),
        (WeightDtype::UInt8, vec![1, 1], vec![127]),
    );
    assert_eq!(
        widen_layer_fp8(&mut t).unwrap(),
        vec!["d.weight_scale".to_string()]
    );
    assert_eq!(t["d.weight"].0, WeightDtype::BF16);
    t.remove("d.weight");
    assert_eq!(t, before, "the other tensors are untouched");
}
