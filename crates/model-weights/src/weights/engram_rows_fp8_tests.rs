// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: `Fp8EngramRows` against a shard file whose every byte is known: rows come back
//! from the right offsets in `ids` order, and every refusal fires before a read.

use std::io::Write;

use super::*;

const ROWS: usize = 5;
const HD: usize = 8;
const SC: usize = 2;
const CODES_AT: u64 = 3;

fn code(r: usize, d: usize) -> u8 {
    (r * 31 + d * 7 + 1) as u8
}

fn scale(r: usize, c: usize) -> u8 {
    (100 + r * 3 + c) as u8
}

/// A shard with 3 junk bytes, the codes, then the scales.
fn shard() -> (tempfile::NamedTempFile, DeferredTensor, DeferredTensor) {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    let mut b = vec![0xEEu8; CODES_AT as usize];
    b.extend((0..ROWS).flat_map(|r| (0..HD).map(move |d| code(r, d))));
    b.extend((0..ROWS).flat_map(|r| (0..SC).map(move |c| scale(r, c))));
    f.write_all(&b).unwrap();
    let p = f.path().to_path_buf();
    let codes =
        DeferredTensor::from_shape(p.clone(), CODES_AT, vec![ROWS, HD], WeightDtype::FP8E4M3)
            .unwrap();
    let scales_at = CODES_AT + (ROWS * HD) as u64;
    let scales =
        DeferredTensor::from_shape(p, scales_at, vec![ROWS, SC], WeightDtype::FP8E8M0).unwrap();
    (f, codes, scales)
}

#[test]
fn rows_come_back_from_their_offsets_in_ids_order() {
    let (_f, c, s) = shard();
    let rd = Fp8EngramRows::new(vec![(7, c, s)]).unwrap();
    let ids = [4u64, 0, 2, 4];
    let mut codes = vec![0u8; ids.len() * HD];
    let mut scales = vec![0u8; ids.len() * SC];
    rd.read_rows(7, &ids, &mut codes, &mut scales).unwrap();
    for (i, &id) in ids.iter().enumerate() {
        let r = id as usize;
        assert_eq!(
            &codes[i * HD..(i + 1) * HD],
            &(0..HD).map(|d| code(r, d)).collect::<Vec<_>>()[..]
        );
        assert_eq!(
            &scales[i * SC..(i + 1) * SC],
            &(0..SC).map(|c| scale(r, c)).collect::<Vec<_>>()[..]
        );
    }
}

#[test]
fn read_refuses_a_row_past_the_table_an_unknown_layer_and_a_wrong_buffer() {
    let (_f, c, s) = shard();
    let rd = Fp8EngramRows::new(vec![(7, c, s)]).unwrap();
    let (mut codes, mut scales) = (vec![0u8; HD], vec![0u8; SC]);
    let e = rd
        .read_rows(7, &[ROWS as u64], &mut codes, &mut scales)
        .unwrap_err();
    assert!(e.to_string().contains("engram row 5 >= 5"), "{e}");
    assert!(rd.read_rows(6, &[0], &mut codes, &mut scales).is_err());
    let mut short = vec![0u8; HD - 1];
    assert!(rd.read_rows(7, &[0], &mut short, &mut scales).is_err());
}

#[test]
fn new_refuses_wrong_dtypes_untiled_scales_repeats_and_nothing() {
    let (_f, c, s) = shard();
    assert!(Fp8EngramRows::new(vec![]).is_err());
    let mut bad = c.clone();
    bad.dtype = WeightDtype::BF16;
    assert!(Fp8EngramRows::new(vec![(1, bad, s.clone())]).is_err());
    let mut bad = s.clone();
    bad.dtype = WeightDtype::FP8E4M3;
    assert!(Fp8EngramRows::new(vec![(1, c.clone(), bad)]).is_err());
    let mut bad = s.clone();
    bad.shape = vec![ROWS, 3];
    assert!(Fp8EngramRows::new(vec![(1, c.clone(), bad)]).is_err());
    let mut bad = s.clone();
    bad.shape = vec![ROWS - 1, SC];
    assert!(Fp8EngramRows::new(vec![(1, c.clone(), bad)]).is_err());
    let e = Fp8EngramRows::new(vec![(1, c.clone(), s.clone()), (1, c, s)]);
    assert!(e.is_err());
}
