// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: Stored dtypes map to their [`RawDtype`]; any other, and an absent name, is
//! refused with the InvalidArgument code.

use metrale_error::ErrorCode;
use metrale_model_weights::fast_weights::MappedSafetensors;

use super::super::checkpoint::{RawDtype, TensorSource};

fn shard(dir: &std::path::Path, tensors: &[(&str, &str, u64)]) {
    let mut header = serde_json::Map::new();
    let mut at = 0u64;
    for (name, dtype, n) in tensors {
        header.insert(
            name.to_string(),
            serde_json::json!({ "dtype": dtype, "shape": [n], "data_offsets": [at, at + n] }),
        );
        at += n;
    }
    let h = serde_json::Value::Object(header).to_string().into_bytes();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    out.extend((0..at).map(|i| i as u8));
    std::fs::write(dir.join("model.safetensors"), out).unwrap();
}

#[test]
fn stored_dtypes_map_and_others_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let kinds = [
        ("b", "BF16", 2),
        ("f", "F32", 4),
        ("e", "F8_E4M3", 1),
        ("s", "F8_E8M0", 1),
        ("p", "I8", 1),
        ("u", "U8", 1),
    ];
    shard(dir.path(), &kinds);
    let m = MappedSafetensors::open(dir.path()).unwrap();
    let want = [
        RawDtype::Bf16,
        RawDtype::F32,
        RawDtype::F8E4m3,
        RawDtype::F8E8m0,
        RawDtype::I8,
    ];
    for ((name, _, n), want) in kinds.iter().zip(want) {
        let t = m.tensor(name).unwrap();
        assert_eq!(
            (t.dtype, t.shape.as_slice(), t.bytes.len() as u64),
            (want, &[*n][..], *n),
            "{name}"
        );
    }
    assert_eq!(
        m.tensor("p").unwrap().bytes,
        &[8u8][..],
        "the tensor's own span"
    );
    for (name, needle) in [
        ("u", "stored as U8"),
        ("absent", "absent from the checkpoint"),
    ] {
        let e = m.tensor(name).err().expect("refused");
        assert_eq!(e.code(), ErrorCode::InvalidArgument);
        assert!(e.to_string().contains(needle), "{e}");
    }
}
