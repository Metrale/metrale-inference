// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Every scheme's bytes decode back to the values within the format's bound, scales
//! are finite and non-negative, no NaN code is written, and a zero row encodes zeros.
//!
//! Owner: metrale-ml-utils.
//! Invariants: decoding here is written from the format definitions (`FP8_E4M3_LUT`,
//! `dequant_nvfp4_to_f32`), independent of the encoder under test.

use metrale_core::numeric::{FP8_E4M3_LUT, dequant_nvfp4_to_f32};

use super::*;
use crate::rng::Stream;

fn group(scheme: Scheme, rows: u64, cols: u64) -> QuantGroup {
    QuantGroup {
        scheme,
        module: "m".into(),
        weight: "m.weight".into(),
        scale: "m.weight_scale".into(),
        global: None,
        input: None,
        rows,
        cols,
    }
}

fn values(rows: usize, cols: usize) -> Vec<f32> {
    let s = Stream::for_tensor(5, "m.weight", "F8_E4M3", &[rows as u64, cols as u64]);
    let mut v: Vec<f32> = (0..rows * cols)
        .map(|i| s.normal4(i as u64) * 0.05)
        .collect();
    v[cols..2 * cols].fill(0.0);
    v
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn bf16s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn check_fp8(v: &[f32], codes: &[u8], scale_of: impl Fn(usize) -> f32) {
    for (i, (&x, &c)) in v.iter().zip(codes).enumerate() {
        assert_ne!(c & 0x7F, 0x7F, "NaN code at {i}");
        let s = scale_of(i);
        assert!(s.is_finite() && s >= 0.0);
        let back = FP8_E4M3_LUT[c as usize] * s;
        // 2026-10-03: E4M3 has 3 mantissa bits: half an ulp is 1/16 relative, plus the
        // subnormal step for values far below the block max.
        assert!(
            (back - x).abs() <= x.abs() / 16.0 + s * 2f32.powi(-9) * 0.5 + 1e-12,
            "{x} -> {back}"
        );
    }
}

#[test]
fn fp8_block_channel_and_tensor_round_trip() {
    let (rows, cols) = (200usize, 300usize);
    let v = values(rows, cols);
    let block = quantize_group(
        &group(
            Scheme::Fp8Block { bn: 128, bk: 128 },
            rows as u64,
            cols as u64,
        ),
        &v,
        GroupDtypes {
            weight: Dtype::F8E4m3,
            scale: Dtype::Bf16,
            global: None,
            input: None,
        },
    )
    .unwrap();
    let bs = bf16s(&block[1]);
    assert_eq!(bs.len(), 2 * 3);
    check_fp8(&v, &block[0], |i| {
        bs[(i / cols / 128) * 3 + (i % cols) / 128]
    });
    let chan = quantize_group(
        &group(Scheme::Fp8Channel, rows as u64, cols as u64),
        &v,
        GroupDtypes {
            weight: Dtype::F8E4m3,
            scale: Dtype::Bf16,
            global: None,
            input: Some(Dtype::F32),
        },
    )
    .unwrap();
    let cs = bf16s(&chan[1]);
    check_fp8(&v, &chan[0], |i| cs[i / cols]);
    assert_eq!(cs[1], 0.0, "the zero row has a zero scale");
    assert!(chan[0][cols..2 * cols].iter().all(|&c| c == 0));
    assert_eq!(f32s(&chan[2]), vec![ACT_AMAX / 448.0]);
    let ten = quantize_group(
        &group(Scheme::Fp8Tensor, rows as u64, cols as u64),
        &v,
        GroupDtypes {
            weight: Dtype::F8E4m3,
            scale: Dtype::F32,
            global: None,
            input: None,
        },
    )
    .unwrap();
    let ts = f32s(&ten[1])[0];
    check_fp8(&v, &ten[0], |_| ts);
}

#[test]
fn nvfp4_globals_follow_their_dialect() {
    let (rows, cols) = (32usize, 64usize);
    let v = values(rows, cols);
    let dt = GroupDtypes {
        weight: Dtype::U8,
        scale: Dtype::F8E4m3,
        global: Some(Dtype::F32),
        input: Some(Dtype::F32),
    };
    let mo = quantize_group(
        &group(
            Scheme::Nvfp4(Nvfp4Global::ModelOpt),
            rows as u64,
            cols as u64,
        ),
        &v,
        dt,
    )
    .unwrap();
    let ct = quantize_group(
        &group(
            Scheme::Nvfp4(Nvfp4Global::CompressedTensors),
            rows as u64,
            cols as u64,
        ),
        &v,
        dt,
    )
    .unwrap();
    assert_eq!(mo[0], ct[0], "same codes");
    assert_eq!(mo[1], ct[1], "same block scales");
    let s2 = f32s(&mo[2])[0];
    let g = f32s(&ct[2])[0];
    assert!(
        (s2 * g - 1.0).abs() < 1e-6,
        "ModelOpt stores amax/2688, compressed-tensors its inverse"
    );
    assert_eq!(f32s(&mo[3])[0], ACT_AMAX / NVFP4_RANGE);
    assert_eq!(f32s(&ct[3])[0], NVFP4_RANGE / ACT_AMAX);
    let back = dequant_nvfp4_to_f32("m", &mo[0], &[rows, cols / 2], &mo[1], s2).unwrap();
    let amax = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    for (x, y) in v.iter().zip(&back) {
        assert!((x - y).abs() <= amax / 6.0, "{x} -> {y}");
    }
}

#[test]
fn plain_values_encode_by_dtype_and_others_are_refused() {
    assert_eq!(
        encode("t", &[1.0, -2.0], Dtype::Bf16).unwrap(),
        vec![0x80, 0x3F, 0x00, 0xC0]
    );
    assert_eq!(
        encode("t", &[1.0], Dtype::F32).unwrap(),
        1.0f32.to_le_bytes().to_vec()
    );
    assert!(encode("t", &[1.0], Dtype::F16).is_err());
}
