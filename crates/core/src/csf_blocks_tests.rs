// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Tests for [`super::decode_csf_32x2_v1`].
//!
//! Owner: metrale-core.
//!
//! Three kinds of evidence:
//! - literal vectors whose code bytes and expected scales are written out by
//!   hand from the spec text (no encoder involved);
//! - round trips through the independent reference encoder in
//!   `csf_blocks_test_ref.rs` over edge-value and odd-geometry matrices;
//! - negative controls: one mutation per invariant, each asserting the error
//!   names the invariant that fired, next to an unmutated positive control.

use super::{
    CsfGeometry, CsfOffsetWidth, CsfPlaneRef, decode_csf_32x2_v1, decode_csf_32x2_v1_to_vec,
};
use crate::mxfp4_e8m0::fp8_e8m0_to_f32;
use crate::numeric::fp8_e4m3_to_f32;

#[path = "csf_blocks_test_ref.rs"]
mod reference;

use reference::{Encoded, Rng, encode, min_bases, scale_like};

const NV: CsfOffsetWidth = CsfOffsetWidth::Nvfp4;
const MX: CsfOffsetWidth = CsfOffsetWidth::Mxfp4;

fn le16(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn le32(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn decode_err(enc: &Encoded) -> String {
    format!("{:#}", decode_csf_32x2_v1_to_vec(&enc.plane()).unwrap_err())
}

fn round_trip(logical: &[u8], rows: usize, columns: usize, width: CsfOffsetWidth) -> Encoded {
    let bases = min_bases(logical, rows, columns, width);
    let enc = encode(logical, rows, columns, width, &bases);
    assert_eq!(decode_csf_32x2_v1_to_vec(&enc.plane()).unwrap(), logical);
    enc
}

/// 2026-10-17: The counts derived from geometry equal the element counts a
/// published NVFP4 expert checkpoint stores per projection (down_proj
/// `[4096, K=2048]` and gate/up `[2048, K=4096]`).
#[test]
fn geometry_matches_published_expert_component_counts() {
    for (rows, columns) in [(4096, 128), (2048, 256)] {
        let g = CsfGeometry::new(rows, columns, NV).unwrap();
        assert_eq!(g.blocks(), 8192);
        assert_eq!(g.groups, 1024);
        assert_eq!(g.pages, 32);
        assert_eq!(g.offsets_entries(), 8224);
        assert_eq!(g.codes_len(), 262_144);
    }
    let g = CsfGeometry::new(32, 129, MX).unwrap();
    assert_eq!((g.column_blocks, g.groups, g.pages), (65, 33, 2));
    assert_eq!(g.codes_len(), 32 * 129 / 8);
}

#[test]
fn geometry_rejects_bad_extents() {
    for (rows, columns) in [(0, 4), (16, 4), (33, 4), (32, 0)] {
        assert!(
            CsfGeometry::new(rows, columns, NV).is_err(),
            "{rows}x{columns}"
        );
    }
}

/// 2026-10-17: One 32x2 block, codes written by hand: byte `lr` holds
/// field `(lr, 0)` in its low nibble and `(lr, 1)` in its high nibble.
#[test]
fn literal_single_block_with_one_exception() {
    let geometry = CsfGeometry::new(32, 2, NV).unwrap();
    let bases: Vec<u8> = (0..32).collect();
    let codes: Vec<u8> = (0..32u8)
        .map(|lr| (lr % 16) | (((lr + 1) % 16) << 4))
        .collect();
    let exceptions = le16(&[(0x7E << 8) | (2 * 5 + 1)]);
    let (page_bases, offsets) = (le32(&[0]), le16(&[0, 1]));
    let plane = CsfPlaneRef {
        geometry,
        bases: &bases,
        codes: &codes,
        page_bases: &page_bases,
        offsets: &offsets,
        exceptions: &exceptions,
    };
    let out = decode_csf_32x2_v1_to_vec(&plane).unwrap();
    for r in 0..32usize {
        for c in 0..2usize {
            let want = if (r, c) == (5, 1) {
                0x7E
            } else {
                (r + (r + c) % 16) as u8
            };
            assert_eq!(out[r * 2 + c], want, "row {r} col {c}");
        }
    }
    assert_eq!(fp8_e4m3_to_f32(out[5 * 2 + 1]), 448.0);
}

/// 2026-10-17: A full 128x4 tile with every field of the k-th visited block
/// set to k. The spec order is column pair 0's four row blocks, then pair
/// 1's, so scale `(r, c)` must read `4 * (c / 2) + r / 32`.
#[test]
fn literal_tile_visits_column_pairs_then_row_blocks() {
    let geometry = CsfGeometry::new(128, 4, NV).unwrap();
    let bases = vec![0u8; 128];
    let codes: Vec<u8> = (0..8u8).flat_map(|k| [k | (k << 4); 32]).collect();
    let (page_bases, offsets) = (le32(&[0]), le16(&[0; 9]));
    let plane = CsfPlaneRef {
        geometry,
        bases: &bases,
        codes: &codes,
        page_bases: &page_bases,
        offsets: &offsets,
        exceptions: &[],
    };
    let out = decode_csf_32x2_v1_to_vec(&plane).unwrap();
    for r in 0..128 {
        for c in 0..4 {
            assert_eq!(out[r * 4 + c] as usize, 4 * (c / 2) + r / 32, "({r},{c})");
        }
    }
}

/// 2026-10-17: 32x3 gives a 64-field block then a 32-field width-1 block;
/// the second block's byte `k` packs rows `2k` and `2k+1`.
#[test]
fn literal_final_single_column_block() {
    let geometry = CsfGeometry::new(32, 3, NV).unwrap();
    let bases = vec![10u8; 32];
    let mut codes = vec![0x11u8; 32];
    codes.extend((0..16u8).map(|k| ((2 * k) % 16) | (((2 * k + 1) % 16) << 4)));
    let exceptions = le16(&[2 * 7]);
    let (page_bases, offsets) = (le32(&[0]), le16(&[0, 0, 1]));
    let mut plane = CsfPlaneRef {
        geometry,
        bases: &bases,
        codes: &codes,
        page_bases: &page_bases,
        offsets: &offsets,
        exceptions: &exceptions,
    };
    let out = decode_csf_32x2_v1_to_vec(&plane).unwrap();
    for r in 0..32usize {
        assert_eq!(&out[r * 3..r * 3 + 2], &[11, 11]);
        let want = if r == 7 { 0 } else { 10 + (r % 16) as u8 };
        assert_eq!(out[r * 3 + 2], want, "row {r}");
    }
    // Negative control: column 1 does not exist in a width-1 block.
    let bad = le16(&[2 * 7 + 1]);
    plane.exceptions = &bad;
    let err = format!("{:#}", decode_csf_32x2_v1_to_vec(&plane).unwrap_err());
    assert!(err.contains("out of bounds"), "{err}");
}

/// 2026-10-17: MXFP4 1-bit fields over E8M0 scale bytes, bases at the 254
/// ceiling and at 127 (1.0), with zero and max-exponent exceptions.
#[test]
fn literal_mxfp4_one_bit_fields() {
    let geometry = CsfGeometry::new(32, 2, MX).unwrap();
    let mut bases = vec![127u8; 32];
    bases[31] = 254;
    let codes = vec![0b1010_1010u8; 8];
    let exceptions = le16(&[2 * 3, (254 << 8) | (2 * 4 + 1)]);
    let (page_bases, offsets) = (le32(&[0]), le16(&[0, 2]));
    let mut plane = CsfPlaneRef {
        geometry,
        bases: &bases,
        codes: &codes,
        page_bases: &page_bases,
        offsets: &offsets,
        exceptions: &exceptions,
    };
    let out = decode_csf_32x2_v1_to_vec(&plane).unwrap();
    for r in 0..32usize {
        let base = bases[r];
        let mut want = [base, base + 1];
        if r == 3 {
            want[0] = 0;
        }
        if r == 4 {
            want[1] = 254;
        }
        assert_eq!(&out[r * 2..r * 2 + 2], &want, "row {r}");
    }
    assert_eq!(fp8_e8m0_to_f32(out[0]), 1.0);
    assert_eq!(fp8_e8m0_to_f32(out[1]), 2.0);
    assert_eq!(fp8_e8m0_to_f32(out[6]), 0.0);
    assert_eq!(out[31 * 2 + 1], 255);
    // Negative control: 255 is above the 1-bit palette ceiling.
    let mut too_high = bases.clone();
    too_high[0] = 255;
    plane.bases = &too_high;
    let err = format!("{:#}", decode_csf_32x2_v1_to_vec(&plane).unwrap_err());
    assert!(err.contains("exceeds 254"), "{err}");
}

/// 2026-10-17: Zero, every E4M3 subnormal, the smallest normal and the
/// largest finite byte round-trip, whether reached by palette or exception.
#[test]
fn edge_scale_bytes_round_trip() {
    let edges = [0x00u8, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x7E];
    let (rows, columns) = (64, 10);
    let mut logical = Vec::new();
    for r in 0..rows {
        for c in 0..columns {
            logical.push(edges[(r + c) % edges.len()]);
        }
    }
    let enc = round_trip(&logical, rows, columns, NV);
    assert!(
        enc.records() > 0,
        "0x7E must be an exception above a 0x00 base"
    );
    let all_max = vec![0x7Eu8; rows * columns];
    assert_eq!(round_trip(&all_max, rows, columns, NV).records(), 0);
    let all_zero = vec![0u8; rows * columns];
    assert_eq!(round_trip(&all_zero, rows, columns, NV).records(), 0);
    let top = vec![0xFFu8; rows * columns];
    let enc = round_trip(&top, rows, columns, NV);
    assert_eq!((enc.bases[0], enc.records()), (240, 0));
    assert_eq!(fp8_e4m3_to_f32(0x01), 2f32.powi(-9));
    assert_eq!(fp8_e4m3_to_f32(0x07), 7.0 * 2f32.powi(-9));
    assert_eq!(fp8_e4m3_to_f32(0x08), 2f32.powi(-6));
}

#[test]
fn odd_and_clipped_geometries_round_trip() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for (rows, columns) in [
        (32, 1),
        (32, 129),
        (64, 3),
        (96, 7),
        (160, 5),
        (128, 4),
        (256, 17),
        (4096, 128),
        (2048, 256),
    ] {
        for width in [NV, MX] {
            let logical = scale_like(rows, columns, &mut rng);
            let enc = round_trip(&logical, rows, columns, width);
            // 2026-10-17: A 32x1 tensor is one clipped block of 32 draws, which may all sit in
            // the base window; it only has to round-trip. Every larger draw must exercise the
            // exception path.
            assert!(
                enc.records() > 0 || rows * columns <= 32,
                "{rows}x{columns} {width:?} needs exceptions"
            );
        }
    }
}

/// 2026-10-17: 32x129 spans two pages; the second page's prefixes restart
/// at zero and its origin is twice the first page's record count.
#[test]
fn second_page_restarts_its_prefix() {
    let mut rng = Rng(7);
    let logical = scale_like(32, 129, &mut rng);
    let enc = round_trip(&logical, 32, 129, NV);
    let page0_blocks = 64; // 32 groups x 2 blocks (one row block per tile)
    let first_of_page1 = page0_blocks + 1;
    let at = |i: usize| u16::from_le_bytes([enc.offsets[2 * i], enc.offsets[2 * i + 1]]);
    assert_eq!(at(first_of_page1), 0);
    let page0_records = at(page0_blocks) as u32;
    let origin1 = u32::from_le_bytes(enc.page_bases[4..8].try_into().unwrap());
    assert_eq!(origin1, 2 * page0_records);
    assert!(page0_records > 0);
}

fn valid_two_page() -> Encoded {
    let mut rng = Rng(42);
    let logical = scale_like(32, 129, &mut rng);
    round_trip(&logical, 32, 129, NV)
}

fn set_u16(bytes: &mut [u8], index: usize, value: u16) {
    bytes[2 * index..2 * index + 2].copy_from_slice(&value.to_le_bytes());
}

#[test]
fn rejects_each_broken_invariant() {
    let good = valid_two_page();
    assert!(decode_csf_32x2_v1_to_vec(&good.plane()).is_ok());

    let mut e = good.clone();
    e.bases[3] = 241;
    assert!(decode_err(&e).contains("exceeds 240"));

    let mut e = good.clone();
    e.codes.pop();
    assert!(decode_err(&e).contains("codes holds"));

    let mut e = good.clone();
    e.page_bases.push(0);
    assert!(decode_err(&e).contains("page_bases holds"));

    let mut e = good.clone();
    e.offsets.truncate(e.offsets.len() - 2);
    assert!(decode_err(&e).contains("offsets holds"));

    let mut e = good.clone();
    e.exceptions.push(0);
    assert!(decode_err(&e).contains("is odd"));

    let mut e = good.clone();
    set_u16(&mut e.offsets, 65, 1);
    assert!(decode_err(&e).contains("first prefix"));

    let mut e = good.clone();
    e.page_bases[4..8].copy_from_slice(&0u32.to_le_bytes());
    assert!(decode_err(&e).contains("origin"));

    let mut e = good.clone();
    e.exceptions.extend_from_slice(&[0, 0]);
    assert!(decode_err(&e).contains("consumed"));
}

/// 2026-10-17: Single-block planes isolate the per-record checks.
fn single_block(offsets_end: u16, records: &[u16]) -> Encoded {
    Encoded {
        geometry: CsfGeometry::new(32, 2, NV).unwrap(),
        bases: vec![0; 32],
        codes: vec![0; 32],
        page_bases: le32(&[0]),
        offsets: le16(&[0, offsets_end]),
        exceptions: le16(records),
    }
}

#[test]
fn rejects_bad_records() {
    assert!(decode_csf_32x2_v1_to_vec(&single_block(2, &[1, 63]).plane()).is_ok());
    let many: Vec<u16> = (0..65).collect();
    assert!(decode_err(&single_block(65, &many)).contains("exceeds its 64 fields"));
    assert!(decode_err(&single_block(1, &[64])).contains("out of bounds"));
    assert!(decode_err(&single_block(2, &[5, 5])).contains("strictly increasing"));
    assert!(decode_err(&single_block(2, &[9, 4])).contains("strictly increasing"));
    assert!(decode_err(&single_block(3, &[1, 2])).contains("past the 2-record arena"));
    let mut e = single_block(0, &[]);
    e.offsets = le16(&[1, 0]);
    assert!(decode_err(&e).contains("first prefix"));
}

#[test]
fn rejects_wrong_output_length() {
    let enc = single_block(0, &[]);
    let mut out = vec![0u8; 63];
    let err = format!(
        "{:#}",
        decode_csf_32x2_v1(&enc.plane(), &mut out).unwrap_err()
    );
    assert!(err.contains("output holds 63"), "{err}");
}
