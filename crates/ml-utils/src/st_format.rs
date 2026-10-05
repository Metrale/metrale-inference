// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The safetensors container as pure data: a shard's header bytes, the shard
//! layout of a whole checkpoint and its `model.safetensors.index.json`. A sink then only
//! appends bytes.
//!
//! Owner: metrale-ml-utils.
//! Invariants:
//! - A shard file is `u64 LE header length`, the header JSON padded with spaces to a multiple
//!   of 8, then each tensor's bytes in header order with no gaps.
//! - `__metadata__` values are strings (the format allows nothing else).

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use crate::index::Dtype;

/// 2026-10-03: One tensor as a shard header states it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeaderTensor<'a> {
    /// 2026-10-03: Name.
    pub name: &'a str,
    /// 2026-10-03: Element type.
    pub dtype: Dtype,
    /// 2026-10-03: Shape.
    pub shape: &'a [u64],
}

/// 2026-10-03: The bytes that open a shard: the 8-byte length and the padded header JSON. The
/// tensors' data follows in the order given.
pub fn shard_header(tensors: &[HeaderTensor<'_>], metadata: &BTreeMap<String, String>) -> Vec<u8> {
    let mut obj = Map::new();
    if !metadata.is_empty() {
        let m: Map<String, Value> = metadata
            .iter()
            .map(|(k, v)| (k.clone(), Value::String(v.clone())))
            .collect();
        obj.insert("__metadata__".into(), Value::Object(m));
    }
    let mut offset = 0u64;
    for t in tensors {
        let len = t.shape.iter().product::<u64>() * t.dtype.bytes();
        obj.insert(
            t.name.to_string(),
            json!({ "dtype": t.dtype.name(), "shape": t.shape, "data_offsets": [offset, offset + len] }),
        );
        offset += len;
    }
    let mut text = serde_json::to_vec(&Value::Object(obj)).expect("a JSON value serializes");
    while !text.len().is_multiple_of(8) {
        text.push(b' ');
    }
    let mut out = (text.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&text);
    out
}

/// 2026-10-03: The file name of shard `i` (0-based) of `n`.
pub fn shard_file(i: usize, n: usize) -> String {
    format!("model-{:05}-of-{:05}.safetensors", i + 1, n)
}

/// 2026-10-03: Split tensors (given by their byte sizes, in order) into shards of at most
/// `max_bytes` each; a tensor larger than `max_bytes` gets a shard of its own. Returns, per
/// shard, the tensor positions it holds.
pub fn split_shards(sizes: &[u64], max_bytes: u64) -> Vec<Vec<usize>> {
    let mut shards: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut cur_bytes = 0u64;
    for (i, &s) in sizes.iter().enumerate() {
        if !cur.is_empty() && cur_bytes + s > max_bytes {
            shards.push(std::mem::take(&mut cur));
            cur_bytes = 0;
        }
        cur.push(i);
        cur_bytes += s;
    }
    if !cur.is_empty() {
        shards.push(cur);
    }
    shards
}

/// 2026-10-03: `model.safetensors.index.json` for `weight_map` (tensor -> shard file).
pub fn index_json(weight_map: &BTreeMap<String, String>, total_size: u64) -> String {
    let v = json!({ "metadata": { "total_size": total_size }, "weight_map": weight_map });
    serde_json::to_string_pretty(&v).expect("a JSON value serializes") + "\n"
}

#[cfg(test)]
#[path = "st_format_tests.rs"]
mod st_format_tests;
