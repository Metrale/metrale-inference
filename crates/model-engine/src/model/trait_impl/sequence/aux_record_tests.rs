// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: Tests for the aux-blob section of a swap record (`aux_record.rs`).
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use std::io::Cursor;

use super::aux_record::{read_aux_blobs, write_aux_blobs};

const NUM_LAYERS: usize = 48;

fn encode(blobs: &[(u32, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    write_aux_blobs(&mut out, blobs).expect("encode");
    out
}

fn decode(bytes: &[u8], num_layers: usize) -> anyhow::Result<Vec<(u32, Vec<u8>)>> {
    read_aux_blobs(&mut Cursor::new(bytes), num_layers)
}

#[test]
fn a_record_round_trips_in_the_written_order() {
    let blobs = vec![
        (2u32, vec![1u8, 2, 3]),
        (0, Vec::new()),
        (47, (0..=255u8).cycle().take(5000).collect()),
    ];
    let bytes = encode(&blobs);
    assert_eq!(decode(&bytes, NUM_LAYERS).expect("decode"), blobs);
}

#[test]
fn an_empty_section_reads_back_empty() {
    let bytes = encode(&[]);
    assert_eq!(bytes.len(), 4, "the count alone");
    assert!(decode(&bytes, NUM_LAYERS).expect("decode").is_empty());
}

/// 2026-10-13: Every proper prefix of a record is refused, so a short spill file
/// never restores a partial carry as if it were complete.
#[test]
fn every_truncation_is_refused() {
    let blobs = vec![(3u32, vec![9u8; 17]), (5, vec![4u8; 2])];
    let bytes = encode(&blobs);
    for cut in 0..bytes.len() {
        assert!(
            decode(&bytes[..cut], NUM_LAYERS).is_err(),
            "a {cut}-byte prefix of a {}-byte record must not decode",
            bytes.len()
        );
    }
    assert!(
        decode(&bytes, NUM_LAYERS).is_ok(),
        "control: the whole record decodes"
    );
}

#[test]
fn a_layer_index_outside_the_model_is_refused() {
    let bytes = encode(&[(48u32, vec![1u8])]);
    assert!(decode(&bytes, 48).is_err());
    assert!(
        decode(&bytes, 49).is_ok(),
        "control: one more layer makes it valid"
    );
}

#[test]
fn a_layer_listed_twice_is_refused() {
    let bytes = encode(&[(7u32, vec![1u8]), (7, vec![2u8])]);
    assert!(decode(&bytes, NUM_LAYERS).is_err());
    let bytes = encode(&[(7u32, vec![1u8]), (8, vec![2u8])]);
    assert!(
        decode(&bytes, NUM_LAYERS).is_ok(),
        "control: distinct layers decode"
    );
}

/// 2026-10-13: A count above the layer count is refused before any blob is read,
/// which is what a record from a different model or a corrupt file looks like.
#[test]
fn a_count_above_the_layer_count_is_refused() {
    let mut bytes = 5u32.to_le_bytes().to_vec();
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    assert!(decode(&bytes, 2).is_err());
}
