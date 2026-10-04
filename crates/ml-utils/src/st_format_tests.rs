// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The shard header round-trips through the index parser, is 8-aligned and lays the
//! data out contiguously; shard splitting keeps order and never splits a tensor.
//!
//! Owner: metrale-ml-utils.
//! Invariants: none beyond the types.

use super::*;
use crate::index::parse_header;

#[test]
fn a_header_round_trips_through_the_index_parser() {
    let tensors = [
        HeaderTensor {
            name: "a.weight",
            dtype: Dtype::Bf16,
            shape: &[4, 8],
        },
        HeaderTensor {
            name: "a.weight_scale_2",
            dtype: Dtype::F32,
            shape: &[],
        },
        HeaderTensor {
            name: "b.weight",
            dtype: Dtype::U8,
            shape: &[3, 5],
        },
    ];
    let meta = BTreeMap::from([("format".to_string(), "pt".to_string())]);
    let bytes = shard_header(&tensors, &meta);
    let len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    assert_eq!(len, bytes.len() - 8);
    assert!(len.is_multiple_of(8), "header padded to 8");
    let parsed = parse_header("s.safetensors", &bytes[8..]).expect("parse");
    let names: Vec<&str> = parsed.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, ["a.weight", "a.weight_scale_2", "b.weight"]);
    assert_eq!(parsed[2].shape, vec![3, 5]);
    let v: serde_json::Value = serde_json::from_slice(&bytes[8..]).unwrap();
    assert_eq!(
        v["a.weight_scale_2"]["data_offsets"],
        serde_json::json!([64, 68])
    );
    assert_eq!(v["b.weight"]["data_offsets"], serde_json::json!([68, 83]));
    assert_eq!(v["__metadata__"]["format"], "pt");
}

#[test]
fn shards_keep_order_and_give_an_oversized_tensor_its_own_file() {
    assert_eq!(
        split_shards(&[3, 3, 3, 10, 1], 6),
        vec![vec![0, 1], vec![2], vec![3], vec![4]]
    );
    assert_eq!(split_shards(&[1, 1], 100), vec![vec![0, 1]]);
    assert!(split_shards(&[], 5).is_empty());
    assert_eq!(shard_file(0, 3), "model-00001-of-00003.safetensors");
}
