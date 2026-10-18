// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The block format: layout constants, hand-derived known answers for encode and
//! decode, whole-head behaviour, and every refusal.

mod common;

use common::Rng;
use metrale_kv_tq128::codebooks::{TCQ3_K, TCQ3_V, TURBO4_LEVELS, TURBO4_MID};
use metrale_kv_tq128::numeric::f16_bits_to_f32;
use metrale_kv_tq128::rotation::{SIGNS1, SIGNS2};
use metrale_kv_tq128::{
    KvSide, Tq128Codec, Tq128Error, decode_group_rotated, dequant_group_bf16, dequant_head_bf16,
    encode_group, encode_head, group_indices, nibble_at,
};

#[test]
fn layouts_match_the_documented_formats() {
    let rows = [
        (Tq128Codec::Tcq3, "turbo3_tcq", 49, 52, 512),
        (Tq128Codec::Tcq2, "turbo2_tcq", 33, 36, 256),
        (Tq128Codec::Tcq1, "turbo1_tcq", 17, 20, 256),
        (Tq128Codec::Turbo4N128, "turbo4_n128", 64, 66, 16),
    ];
    for (c, name, payload, block, book) in rows {
        assert_eq!(
            (c.name(), c.payload_bytes(), c.block_bytes()),
            (name, payload, block)
        );
        for side in [KvSide::Key, KvSide::Value] {
            assert_eq!(c.codebook(side).len(), book);
            assert_eq!(c.codebook_len(), book);
            assert!(c.codebook(side).iter().all(|v| v.is_finite()));
        }
    }
    let v = |c: Tq128Codec| c.decode_scale(KvSide::Value);
    assert_eq!(
        [
            v(Tq128Codec::Tcq1),
            v(Tq128Codec::Tcq2),
            v(Tq128Codec::Tcq3)
        ],
        [1.26, 1.06, 1.02]
    );
    assert_eq!(v(Tq128Codec::Turbo4N128), 1.0);
    assert!(
        Tq128Codec::ALL
            .iter()
            .all(|c| c.decode_scale(KvSide::Key) == 1.0)
    );
    assert_ne!(
        Tq128Codec::Tcq3.codebook(KvSide::Key),
        Tq128Codec::Tcq3.codebook(KvSide::Value)
    );
    for i in 0..15 {
        assert!(
            TURBO4_LEVELS[i] < TURBO4_MID[i] && TURBO4_MID[i] < TURBO4_LEVELS[i + 1],
            "{i}"
        );
    }
}

#[test]
fn turbo4_n128_encodes_a_scaled_basis_vector_as_derived_by_hand() {
    // x = 3 e_0: norm 3, x/norm = e_0, rotate(e_0)[i] = s1[0] s2[i] / sqrt(128) = +-0.0884.
    // +0.0884 lies in [mid[10], mid[11]) = [0.0707, 0.0972): index 11 (level 0.083317);
    // -0.0884 lies in [mid[3], mid[4]): index 4 (level -0.083317).
    let mut x = [0.0_f32; 128];
    x[0] = 3.0;
    let mut block = [0_u8; 66];
    encode_group(Tq128Codec::Turbo4N128, KvSide::Key, &x, &mut block).unwrap();
    for i in 0..128 {
        let want = if SIGNS1[0] * SIGNS2[i] > 0 { 11 } else { 4 };
        assert_eq!(nibble_at(&block[2..], i), want, "element {i}");
    }
    // Stored norm = 3 / |recon| = 3 / (0.083317 * sqrt(128)), to one fp16 ulp.
    let norm = f64::from(f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]])));
    let want = 3.0 / (f64::from(0.083317_f32) * 128f64.sqrt());
    assert!((norm - want).abs() <= 2f64.powi(-9), "{norm} vs {want}");
}

#[test]
fn tcq3_decodes_a_hand_built_block() {
    // norm 1.0 (0x3C00); stream: prefix 0b000101, then 128 symbols 0b111, then two zero pad
    // bits. States: 5 | 7 << 6 = 453, (453 >> 3) | 448 = 504, (504 >> 3) | 448 = 511, 511, ..
    let mut block = [0_u8; 52];
    block[..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
    block[2] = 0b1100_0101;
    block[3..50].fill(0xFF);
    block[50] = 0b0011_1111;
    let idx = group_indices(Tq128Codec::Tcq3, &block).unwrap();
    assert_eq!(&idx[..4], &[453, 504, 511, 511]);
    assert!(idx[3..].iter().all(|&s| s == 511));
    let k = decode_group_rotated(Tq128Codec::Tcq3, KvSide::Key, &block).unwrap();
    assert_eq!(
        (k[0], k[1], k[127]),
        (TCQ3_K[453], TCQ3_K[504], TCQ3_K[511])
    );
    let v = decode_group_rotated(Tq128Codec::Tcq3, KvSide::Value, &block).unwrap();
    assert_eq!(v[0], TCQ3_V[453] * 1.02);
}

#[test]
fn wide_heads_are_independent_128_groups() {
    let mut rng = Rng::new(40);
    for codec in Tq128Codec::ALL {
        for head_dim in [128, 256, 512] {
            let row = rng.normals(head_dim, 1.0);
            let bb = codec.block_bytes();
            let mut whole = vec![0_u8; head_dim / 128 * bb];
            encode_head(codec, KvSide::Key, &row, &mut whole).unwrap();
            let mut out = vec![0_u16; head_dim];
            dequant_head_bf16(codec, KvSide::Key, &whole, &mut out).unwrap();
            for g in 0..head_dim / 128 {
                let mut one = vec![0_u8; bb];
                encode_group(codec, KvSide::Key, &row[g * 128..(g + 1) * 128], &mut one).unwrap();
                assert_eq!(
                    &whole[g * bb..(g + 1) * bb],
                    &one[..],
                    "{codec:?} group {g}"
                );
                let y = dequant_group_bf16(codec, KvSide::Key, &one).unwrap();
                assert_eq!(&out[g * 128..(g + 1) * 128], &y[..]);
            }
        }
    }
}

#[test]
fn malformed_inputs_are_refused() {
    let c = Tq128Codec::Tcq2;
    let mut out = vec![0_u8; 36];
    for head_dim in [0, 96, 200] {
        let row = vec![0.0_f32; head_dim];
        assert_eq!(
            encode_head(c, KvSide::Key, &row, &mut out).unwrap_err(),
            Tq128Error::HeadDim { head_dim }
        );
        let mut deq = vec![0_u16; head_dim];
        assert_eq!(
            dequant_head_bf16(c, KvSide::Key, &out, &mut deq).unwrap_err(),
            Tq128Error::HeadDim { head_dim }
        );
    }
    assert_eq!(
        encode_head(c, KvSide::Key, &[0.0; 256], &mut out).unwrap_err(),
        Tq128Error::BlockLength {
            expected: 72,
            found: 36
        }
    );
    assert_eq!(
        encode_group(c, KvSide::Key, &[0.0; 127], &mut out).unwrap_err(),
        Tq128Error::GroupLength { found: 127 }
    );
    assert_eq!(
        encode_group(c, KvSide::Key, &[0.0; 128], &mut out[..35]).unwrap_err(),
        Tq128Error::BlockLength {
            expected: 36,
            found: 35
        }
    );
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut x = [0.5_f32; 128];
        x[5] = bad;
        assert_eq!(
            encode_group(c, KvSide::Value, &x, &mut out).unwrap_err(),
            Tq128Error::NonFinite { index: 5 }
        );
    }
    assert_eq!(
        group_indices(Tq128Codec::Turbo4N128, &[0; 36]).unwrap_err(),
        Tq128Error::BlockLength {
            expected: 66,
            found: 36
        }
    );
    let mut deq = [0_u16; 128];
    assert_eq!(
        dequant_head_bf16(c, KvSide::Key, &[0; 35], &mut deq).unwrap_err(),
        Tq128Error::BlockLength {
            expected: 36,
            found: 35
        }
    );
}
