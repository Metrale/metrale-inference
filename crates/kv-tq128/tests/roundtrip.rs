// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: Whole-group encode/decode: error bounds per width, the rate-distortion floor,
//! norm preservation, determinism, and the zero group.

mod common;

use common::Rng;
use metrale_kv_tq128::numeric::f16_bits_to_f32;
use metrale_kv_tq128::{
    KvSide, Tq128Codec, decode_group_rotated, dequant_group, dequant_group_bf16, encode_group,
};

/// Mean relative squared error over `n` Gaussian groups (K side: decode scale 1.0).
fn rel_mse(codec: Tq128Codec, side: KvSide, n: usize, seed: u64) -> f64 {
    let mut rng = Rng::new(seed);
    let mut block = vec![0_u8; codec.block_bytes()];
    let (mut err, mut sig) = (0.0_f64, 0.0_f64);
    for _ in 0..n {
        let x = rng.normals(128, 1.0);
        encode_group(codec, side, &x, &mut block).unwrap();
        let y = dequant_group(codec, side, &block).unwrap();
        for (a, b) in x.iter().zip(&y) {
            err += (f64::from(*a) - f64::from(*b)).powi(2);
            sig += f64::from(*a).powi(2);
        }
    }
    err / sig
}

#[test]
fn error_falls_with_rate_and_stays_inside_measured_bounds() {
    // Bounds: the K-side figure each codec measures on this seed (0.290, 0.0781, 0.0205,
    // 0.00923) plus ~12% margin; they fail if a codebook, the rotation or the search degrades.
    // Each TCQ bound is also below the Lloyd-Max scalar quantizer at the same symbol width
    // (0.363, 0.117, 0.0345 for 1, 2, 3 bits): the trellis must buy its gain.
    let bound = |c| match c {
        Tq128Codec::Tcq1 => 0.325,
        Tq128Codec::Tcq2 => 0.0875,
        Tq128Codec::Tcq3 => 0.023,
        Tq128Codec::Turbo4N128 => 0.0104,
    };
    let mut prev = f64::INFINITY;
    for codec in [
        Tq128Codec::Tcq1,
        Tq128Codec::Tcq2,
        Tq128Codec::Tcq3,
        Tq128Codec::Turbo4N128,
    ] {
        let d = rel_mse(codec, KvSide::Key, 400, 30);
        let bpv = (codec.block_bytes() * 8) as f64 / 128.0;
        println!(
            "{codec:?} bpv {bpv} rel-mse {d:.5} floor {:.5}",
            2f64.powf(-2.0 * bpv)
        );
        assert!(d < bound(codec), "{codec:?}: {d}");
        // No code beats the Gaussian rate-distortion function D(R) = 2^(-2R).
        assert!(
            d > 2f64.powf(-2.0 * bpv),
            "{codec:?} below the Shannon floor: {d}"
        );
        assert!(d < prev, "{codec:?} not better than the lower rate");
        prev = d;
    }
}

#[test]
fn decode_restores_the_input_norm() {
    let mut rng = Rng::new(31);
    for codec in Tq128Codec::ALL {
        let mut block = vec![0_u8; codec.block_bytes()];
        for _ in 0..50 {
            let x = rng.normals(128, 3.0);
            encode_group(codec, KvSide::Key, &x, &mut block).unwrap();
            let y = decode_group_rotated(codec, KvSide::Key, &block).unwrap();
            let n = |v: &[f32]| v.iter().map(|a| f64::from(*a).powi(2)).sum::<f64>().sqrt();
            let rel = (n(&y) - n(&x)).abs() / n(&x);
            // fp16 keeps 11 significant bits: half an ulp is 2^-11 relative.
            assert!(rel < 2f64.powi(-11) + 1e-6, "{codec:?}: {rel}");
        }
    }
}

#[test]
fn value_side_decodes_at_the_codec_scale() {
    let mut rng = Rng::new(32);
    let x = rng.normals(128, 1.0);
    for codec in Tq128Codec::ALL {
        let mut block = vec![0_u8; codec.block_bytes()];
        encode_group(codec, KvSide::Value, &x, &mut block).unwrap();
        let idx = metrale_kv_tq128::group_indices(codec, &block).unwrap();
        let y = decode_group_rotated(codec, KvSide::Value, &block).unwrap();
        let scale = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]))
            * codec.decode_scale(KvSide::Value);
        for (o, i) in y.iter().zip(idx) {
            assert_eq!(
                o.to_bits(),
                (codec.codebook(KvSide::Value)[usize::from(i)] * scale).to_bits()
            );
        }
    }
}

#[test]
fn encode_is_deterministic_and_the_zero_group_decodes_to_zeros() {
    let mut rng = Rng::new(33);
    let x = rng.normals(128, 0.5);
    for codec in Tq128Codec::ALL {
        let (mut a, mut b) = (
            vec![0_u8; codec.block_bytes()],
            vec![1_u8; codec.block_bytes()],
        );
        encode_group(codec, KvSide::Key, &x, &mut a).unwrap();
        encode_group(codec, KvSide::Key, &x, &mut b).unwrap();
        assert_eq!(a, b);
        encode_group(codec, KvSide::Key, &[0.0; 128], &mut a).unwrap();
        assert_eq!(&a[..2], &[0, 0], "{codec:?} zero norm");
        let y = dequant_group_bf16(codec, KvSide::Value, &a).unwrap();
        assert!(y.iter().all(|&h| h & 0x7FFF == 0), "{codec:?}: {y:?}");
    }
}
