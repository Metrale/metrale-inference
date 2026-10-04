// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Header parsing accepts a well-formed header and refuses each malformed one by
//! name (Path B: every refusal is a distinct input the parser must not accept).
//!
//! Owner: metrale-ml-utils.
//! Invariants: none beyond the types.

use super::*;

fn header(body: &str) -> Vec<u8> {
    body.as_bytes().to_vec()
}

#[test]
fn a_well_formed_header_parses_and_skips_metadata() {
    let h = header(
        r#"{"__metadata__":{"format":"pt"},
            "x.weight":{"dtype":"F8_E4M3","shape":[2,16],"data_offsets":[0,32]},
            "x.weight_scale_inv":{"dtype":"BF16","shape":[1,1],"data_offsets":[32,34]}}"#,
    );
    let idx = TensorIndex::from_headers(&[("s0".into(), h)]).expect("index");
    assert_eq!(idx.len(), 2);
    assert_eq!(idx.get("x.weight").unwrap().dtype, Dtype::F8E4m3);
    assert_eq!(idx.bytes(), 34);
}

#[test]
fn malformed_headers_are_refused_by_name() {
    let cases = [
        (
            r#"{"t":{"dtype":"Q4","shape":[1],"data_offsets":[0,1]}}"#,
            "dtype Q4",
        ),
        (
            r#"{"t":{"dtype":"U8","shape":[-1],"data_offsets":[0,1]}}"#,
            "non-integer",
        ),
        (
            r#"{"t":{"dtype":"U8","shape":[2],"data_offsets":[2,0]}}"#,
            "reversed",
        ),
        (
            r#"{"t":{"dtype":"BF16","shape":[2],"data_offsets":[0,3]}}"#,
            "3 bytes",
        ),
        (
            r#"{"t":{"dtype":"U8","shape":[4294967296,4294967296,4294967296],"data_offsets":[0,1]}}"#,
            "overflows",
        ),
        (r#"[1]"#, "not an object"),
    ];
    for (body, want) in cases {
        let err = TensorIndex::from_headers(&[("s".into(), header(body))]).unwrap_err();
        assert!(err.to_string().contains(want), "{body}: {err}");
    }
}

#[test]
fn a_tensor_in_two_shards_is_refused() {
    let t = r#"{"t":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
    let err =
        TensorIndex::from_headers(&[("a".into(), header(t)), ("b".into(), header(t))]).unwrap_err();
    assert!(err.to_string().contains("both a and b"), "{err}");
}

#[test]
fn the_digest_sees_shape_and_dtype_but_not_the_shard() {
    let e = |shard: &str, shape: Vec<u64>, dtype| TensorEntry {
        name: "t".into(),
        dtype,
        shape,
        shard: shard.into(),
        offset: 0,
    };
    let d = |x| TensorIndex::from_entries(vec![x]).unwrap().digest();
    assert_eq!(d(e("a", vec![2], Dtype::U8)), d(e("b", vec![2], Dtype::U8)));
    assert_ne!(d(e("a", vec![2], Dtype::U8)), d(e("a", vec![3], Dtype::U8)));
    assert_ne!(d(e("a", vec![2], Dtype::U8)), d(e("a", vec![2], Dtype::I8)));
}
