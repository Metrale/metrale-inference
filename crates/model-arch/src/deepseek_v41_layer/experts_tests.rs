// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The FP8 engram row decode against the reference's dequantize, value by value,
//! over every E4M3 code and a spread of scales.

use super::decode_fp8_rows;
use crate::deepseek_v41_ref::mat::{e4m3, e8m0};

#[test]
fn decode_is_the_reference_dequantize_bit_for_bit_and_picks_the_scale_by_block() {
    let (hd, sc) = (256usize, 8usize);
    let codes: Vec<u8> = (0..2 * hd)
        .map(|i| (i % 256) as u8)
        .filter(|&c| c & 0x7F != 0x7F)
        .chain([0u8, 1, 2, 3, 0x80, 0x81])
        .take(2 * hd)
        .collect();
    assert_eq!(codes.len(), 2 * hd);
    let scales: Vec<u8> = [100u8, 110, 120, 127, 130, 140, 90, 160]
        .iter()
        .chain(&[127u8, 126, 128, 125, 129, 124, 131, 123])
        .copied()
        .collect();
    let bits = decode_fp8_rows(&codes, &scales, hd, sc);
    for (i, &b) in bits.iter().enumerate() {
        let (r, d) = (i / hd, i % hd);
        let want = e4m3(codes[i]) * e8m0(scales[r * sc + d / 32]);
        let got = f32::from_bits(u32::from(b) << 16);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "row {r} col {d} code {}",
            codes[i]
        );
    }
}

#[test]
#[should_panic(expected = "E8M0 NaN scale")]
fn decode_refuses_the_nan_scale() {
    decode_fp8_rows(&[1u8; 32], &[0xFF], 32, 1);
}
