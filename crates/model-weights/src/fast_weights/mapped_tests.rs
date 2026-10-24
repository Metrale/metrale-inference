// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-19: [`super::MappedSafetensors`] returns each tensor's own bytes, shape and stored
//! dtype across shards, and refuses a name held by two shards.

use super::MappedSafetensors;

/// 2026-10-19: Write a safetensors file holding `tensors` (name, dtype, shape, bytes) in order.
fn write_shard(path: &std::path::Path, tensors: &[(&str, &str, &[u64], &[u8])]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let start = data.len();
        data.extend_from_slice(bytes);
        header.insert(
            name.to_string(),
            serde_json::json!({ "dtype": dtype, "shape": shape, "data_offsets": [start, data.len()] }),
        );
    }
    let h = serde_json::Value::Object(header).to_string().into_bytes();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    out.extend_from_slice(&data);
    std::fs::write(path, out).unwrap();
}

#[test]
fn tensors_come_back_with_their_own_bytes_shape_and_stored_dtype() {
    let dir = tempfile::tempdir().unwrap();
    write_shard(
        &dir.path()
            .join("model.safetensors-00001-of-00002.safetensors"),
        &[
            ("a", "I8", &[2, 2], &[1, 2, 3, 4]),
            ("b", "BF16", &[1, 3], &[5, 6, 7, 8, 9, 10]),
        ],
    );
    write_shard(
        &dir.path()
            .join("model.safetensors-00002-of-00002.safetensors"),
        &[("c", "F8_E8M0", &[3], &[127, 128, 129])],
    );
    let m = MappedSafetensors::open(dir.path()).unwrap();
    assert_eq!(m.len(), 3);
    let a = m.get("a").unwrap();
    assert_eq!(
        (a.stored_dtype, a.shape, a.bytes),
        ("I8", &[2usize, 2][..], &[1u8, 2, 3, 4][..])
    );
    let b = m.get("b").unwrap();
    assert_eq!(
        (b.stored_dtype, b.shape, b.bytes),
        ("BF16", &[1usize, 3][..], &[5u8, 6, 7, 8, 9, 10][..])
    );
    let c = m.get("c").unwrap();
    assert_eq!(
        (c.stored_dtype, c.bytes),
        ("F8_E8M0", &[127u8, 128, 129][..])
    );
    assert!(m.get("d").is_none());
}

#[test]
fn a_name_in_two_shards_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    for i in 1..=2 {
        write_shard(
            &dir.path()
                .join(format!("model.safetensors-0000{i}-of-00002.safetensors")),
            &[("a", "F32", &[1], &[0, 0, 128, 63])],
        );
    }
    let e = MappedSafetensors::open(dir.path())
        .err()
        .expect("a duplicate must be refused");
    assert!(e.to_string().contains("more than one shard"), "{e}");
}
