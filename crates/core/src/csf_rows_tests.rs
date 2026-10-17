// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Tests for the csf device form ([`super::encode_csf_rows`],
//! [`super::decode_csf_rows`], [`super::CsfRowsLayout`]).
//!
//! Owner: metrale-core.
//!
//! Evidence: a literal blob written out by hand from the region table; round trips over edge
//! values (base 240 with field 15, scales below the base, column 255); the per-component charge
//! summing to the blob size; and one negative control per invariant beside a positive control.

use super::{
    CSF_ROWS_ALIGN, CsfRowsLayout, csf_rows_records, csf_rows_supported, decode_csf_rows,
    encode_csf_rows, encode_csf_rows_for_plane,
};
use crate::csf_blocks::{SUFFIX_BASES, SUFFIX_CODES, SUFFIX_EXCEPTIONS, SUFFIX_OFFSETS};

/// 2026-10-17: Deterministic xorshift64* bytes.
struct Rng(u64);

impl Rng {
    fn next_u8(&mut self) -> u8 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 56) as u8
    }
}

/// 2026-10-17: Rows mostly inside a 16-wide window above a per-row base, one in eight anywhere.
fn scale_like(rows: usize, cols: usize, rng: &mut Rng) -> (Vec<u8>, Vec<u8>) {
    let mut scales = Vec::with_capacity(rows * cols);
    let mut bases = Vec::with_capacity(rows);
    for _ in 0..rows {
        let base = rng.next_u8() % 241;
        bases.push(base);
        for _ in 0..cols {
            let s = if rng.next_u8() < 32 {
                rng.next_u8()
            } else {
                base + rng.next_u8() % 16
            };
            scales.push(s);
        }
    }
    (scales, bases)
}

fn ptr(blob: &[u8], l: &CsfRowsLayout, r: usize) -> u32 {
    let at = l.row_ptr_offset() + 4 * r;
    u32::from_le_bytes(blob[at..at + 4].try_into().unwrap())
}

/// 2026-10-17: A 32x4 plane, every byte of the blob derived by hand from the region table.
#[test]
fn a_literal_blob_matches_the_region_table() {
    let (rows, cols) = (32usize, 4usize);
    let mut scales = vec![0u8; rows * cols];
    let mut bases = vec![0u8; rows];
    for r in 0..rows {
        bases[r] = 10;
        for c in 0..cols {
            scales[r * cols + c] = 10 + c as u8;
        }
    }
    // 2026-10-17: Row 0: column 1 below the base, column 3 at base + 16; row 31: base 240,
    // column 2 at 255 (field 15, not a record).
    scales[1] = 9;
    scales[3] = 26;
    bases[31] = 240;
    scales[31 * 4..32 * 4].copy_from_slice(&[240, 241, 255, 243]);
    let l = CsfRowsLayout::new(rows, cols, 3).unwrap();
    // 2026-10-17: codes 64 | bases 32 | row_ptr 132 -> head round16(164) = 176 | records 16.
    assert_eq!(
        (l.bases_offset(), l.row_ptr_offset(), l.exceptions_offset()),
        (64, 96, 240)
    );
    assert_eq!(l.bytes(), 256);
    let blob = encode_csf_rows(&scales, &bases, &l).unwrap();
    // 2026-10-17: Row 0 fields (0, rec->0, 2, rec->0): byte 0 = 0x00, byte 1 = 0x02.
    assert_eq!(&blob[0..2], &[0x00, 0x02]);
    // 2026-10-17: Row 1 fields (0, 1, 2, 3): 0x10, 0x32.
    assert_eq!(&blob[2..4], &[0x10, 0x32]);
    // 2026-10-17: Row 31 fields (0, 1, 15, 3): 0x10, 0x3F.
    assert_eq!(&blob[62..64], &[0x10, 0x3F]);
    assert_eq!(blob[64], 10);
    assert_eq!(blob[64 + 31], 240);
    assert_eq!((ptr(&blob, &l, 0), ptr(&blob, &l, 1)), (0, 2));
    assert_eq!((ptr(&blob, &l, 31), ptr(&blob, &l, 32)), (2, 2));
    // 2026-10-17: Records (9 << 8) | 1 and (26 << 8) | 3, then zero padding.
    assert_eq!(&blob[240..246], &[0x01, 0x09, 0x03, 0x1A, 0x00, 0x00]);
    assert!(blob[228..240].iter().all(|&b| b == 0), "head padding");
    assert_eq!(decode_csf_rows(&blob, &l).unwrap(), scales);
}

/// 2026-10-17: Encode then decode returns the scales over random planes of every column count
/// class (4, a non-power of two, the widest 256), with record columns reaching 255.
#[test]
fn round_trips_are_exact() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for (rows, cols) in [
        (32usize, 4usize),
        (64, 12),
        (96, 128),
        (32, 256),
        (128, 256),
    ] {
        let (mut scales, bases) = scale_like(rows, cols, &mut rng);
        scales[cols - 1] = bases[0].wrapping_add(200);
        let records = csf_rows_records(&scales, &bases, cols);
        assert!(records > 0, "[{rows}, {cols}] exercises records");
        let (l, blob) = encode_csf_rows_for_plane(&scales, &bases, cols, records + 5).unwrap();
        assert_eq!(blob.len(), l.bytes());
        assert_eq!(ptr(&blob, &l, rows) as usize, records);
        assert_eq!(
            decode_csf_rows(&blob, &l).unwrap(),
            scales,
            "[{rows}, {cols}]"
        );
    }
}

/// 2026-10-17: Negative control for the round trips: one flipped field nibble moves exactly one
/// decoded scale, so the decoder reads the codes and not a copy.
#[test]
fn a_flipped_nibble_moves_exactly_one_scale() {
    let mut rng = Rng(7);
    let (rows, cols) = (64usize, 16usize);
    let (scales, bases) = scale_like(rows, cols, &mut rng);
    let (l, mut blob) = encode_csf_rows_for_plane(&scales, &bases, cols, rows * cols).unwrap();
    let pos = (0..rows * cols)
        .find(|&i| scales[i] >= bases[i / cols] && scales[i] - bases[i / cols] < 8)
        .unwrap();
    blob[pos / 2] ^= 0x8 << (4 * ((pos % cols) & 1));
    let got = decode_csf_rows(&blob, &l).unwrap();
    let moved: Vec<usize> = (0..rows * cols).filter(|&i| got[i] != scales[i]).collect();
    assert_eq!(moved, vec![pos]);
}

/// 2026-10-17: The bases, codes and exceptions charges of a plane sum to its blob size, the other
/// components charge nothing, and every region is aligned for its loads.
#[test]
fn component_charges_sum_to_the_blob() {
    for (rows, cols, e) in [
        (32usize, 4usize, 0usize),
        (2048, 256, 1171),
        (4096, 128, 2335),
    ] {
        let l = CsfRowsLayout::new(rows, cols, e).unwrap();
        let charge = |s: &str, n: usize| CsfRowsLayout::component_charge(s, n).unwrap();
        let sum = charge(SUFFIX_CODES, rows * cols / 2)
            + charge(SUFFIX_BASES, rows)
            + charge(SUFFIX_EXCEPTIONS, e);
        assert_eq!(sum, l.bytes(), "[{rows}, {cols}] E {e}");
        assert_eq!(charge(SUFFIX_OFFSETS, 999), 0);
        assert!(l.row_ptr_offset().is_multiple_of(4));
        assert!(l.exceptions_offset().is_multiple_of(CSF_ROWS_ALIGN));
        assert!(l.bytes().is_multiple_of(CSF_ROWS_ALIGN));
    }
    assert_eq!(
        CsfRowsLayout::component_charge(SUFFIX_EXCEPTIONS, usize::MAX),
        None
    );
}

/// 2026-10-17: Unsupported geometry is refused (positive controls on each boundary).
#[test]
fn unsupported_geometry_is_refused() {
    for (rows, cols) in [(32usize, 4usize), (32, 256), (64, 8)] {
        assert!(csf_rows_supported(rows, cols), "[{rows}, {cols}]");
    }
    for (rows, cols) in [
        (0usize, 4usize),
        (48, 4),
        (32, 0),
        (32, 6),
        (32, 260),
        (32, 2),
    ] {
        assert!(!csf_rows_supported(rows, cols), "[{rows}, {cols}]");
        let err = CsfRowsLayout::new(rows, cols, 0).unwrap_err();
        assert!(format!("{err}").contains("no device form"), "{err}");
    }
}

/// 2026-10-17: Encoding refuses a base above 240, a capacity below the records, and mismatched
/// lengths; each beside the passing case.
#[test]
fn encoding_refuses_what_it_cannot_represent() {
    let (rows, cols) = (32usize, 4usize);
    let scales = vec![255u8; rows * cols];
    let mut bases = vec![240u8; rows];
    let l = CsfRowsLayout::new(rows, cols, 0).unwrap();
    assert!(encode_csf_rows(&scales, &bases, &l).is_ok());
    bases[5] = 241;
    let err = encode_csf_rows(&scales, &bases, &l).unwrap_err();
    assert!(format!("{err}").contains("exceeds 240"), "{err}");
    bases[5] = 0;
    let err = encode_csf_rows(&scales, &bases, &l).unwrap_err();
    assert!(format!("{err}").contains("exceed the capacity"), "{err}");
    let err = encode_csf_rows(&scales[1..], &bases, &l).unwrap_err();
    assert!(format!("{err}").contains("scales and"), "{err}");
}

/// 2026-10-17: Decoding refuses a corrupted prefix, column or order; each mutation of a valid
/// blob that decodes.
#[test]
fn decoding_refuses_corrupt_blobs() {
    let (rows, cols) = (32usize, 8usize);
    let mut scales: Vec<u8> = (0..rows * cols).map(|i| 50 + (i % 16) as u8).collect();
    let bases = vec![50u8; rows];
    // 2026-10-17: Two records in row 0 (columns 2 and 5), one in row 1 (column 0).
    scales[2] = 1;
    scales[5] = 200;
    scales[cols] = 3;
    let (l, blob) = encode_csf_rows_for_plane(&scales, &bases, cols, 4).unwrap();
    assert_eq!(decode_csf_rows(&blob, &l).unwrap(), scales);
    let exc = l.exceptions_offset();
    let set_ptr = |b: &mut Vec<u8>, r: usize, v: u32| {
        let at = l.row_ptr_offset() + 4 * r;
        b[at..at + 4].copy_from_slice(&v.to_le_bytes());
    };
    type Mutation<'a> = (&'a str, Box<dyn Fn(&mut Vec<u8>) + 'a>);
    let mutations: Vec<Mutation> = vec![
        ("first row prefix", Box::new(|b| set_ptr(b, 0, 1))),
        ("outside 0..4", Box::new(|b| set_ptr(b, rows, 5))),
        ("outside 0..4", Box::new(|b| set_ptr(b, 3, 2))),
        ("outside the row", Box::new(move |b| b[exc] = cols as u8)),
        ("out of order", Box::new(move |b| b[exc + 2] = 2)),
        ("exceeds 240", Box::new(|b| b[l.bases_offset() + 7] = 241)),
        ("blob of", Box::new(|b| b.truncate(b.len() - 16))),
    ];
    for (why, mutate) in mutations {
        let mut b = blob.clone();
        mutate(&mut b);
        let err = decode_csf_rows(&b, &l).unwrap_err();
        assert!(format!("{err}").contains(why), "{why}: {err}");
    }
}
