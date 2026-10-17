// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: [`super::resolve_shards`] routes every indexed tensor to the
//! file the shard walk opens, including shards in a subdirectory.

use super::resolve_shards;
use std::path::PathBuf;

fn write_index(dir: &std::path::Path, entries: &[(&str, &str)]) {
    let map: serde_json::Map<String, serde_json::Value> = entries
        .iter()
        .map(|(t, s)| (t.to_string(), serde_json::Value::from(*s)))
        .collect();
    let body = serde_json::json!({ "metadata": {}, "weight_map": map });
    std::fs::write(dir.join("model.safetensors.index.json"), body.to_string()).unwrap();
}

#[test]
fn subdirectory_shard_receives_its_tensors() {
    let dir = tempfile::tempdir().unwrap();
    write_index(
        dir.path(),
        &[
            ("a.weight", "model-00001-of-00002.safetensors"),
            ("b.weight", "model-00001-of-00002.safetensors"),
            ("c.weight", "model-00002-of-00002.safetensors"),
            ("q.weight", "tensors/native-QB.safetensors"),
            ("r.weight", "tensors/native-QB.safetensors"),
        ],
    );
    let (files, map) = resolve_shards(dir.path()).unwrap();
    let map = map.expect("an index yields a routing map");
    let names = |p: &PathBuf| {
        let mut v = map.get(p).cloned().unwrap_or_default();
        v.sort();
        v
    };
    assert_eq!(files.len(), 3);
    // 2026-10-17: Every file the walk will open has a map entry under that exact path.
    for f in &files {
        assert!(map.contains_key(f), "{} has no routing entry", f.display());
    }
    let sub = dir.path().join("tensors/native-QB.safetensors");
    assert!(files.contains(&sub));
    assert_eq!(names(&sub), ["q.weight", "r.weight"]);
    // 2026-10-17: Positive control: a flat shard routes as before.
    let flat = dir.path().join("model-00001-of-00002.safetensors");
    assert_eq!(names(&flat), ["a.weight", "b.weight"]);
    // 2026-10-17: Every indexed tensor is routed exactly once.
    let total: usize = map.values().map(Vec::len).sum();
    assert_eq!(total, 5);
}

#[test]
fn unindexed_directory_yields_no_routing_map() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("model.safetensors"), b"").unwrap();
    let (files, map) = resolve_shards(dir.path()).unwrap();
    assert_eq!(files, [dir.path().join("model.safetensors")]);
    assert!(map.is_none());
}
