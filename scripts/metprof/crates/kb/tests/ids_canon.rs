// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: natural-key ids and canonical JSON (PLAN W1.1).

use metprof_kb::canon::{canonical, content_sha256};
use metprof_kb::ids::{natural_key, point_id};
use metprof_kb::schema::FusionRule;
use metprof_kb::schema::{Body, Kind, Record};
use serde_json::{Value, json};

#[test]
fn same_key_gives_same_id_and_one_changed_byte_does_not() {
    let key = natural_key(&["attn", "flash_fwd"]);
    assert_eq!(point_id(Kind::Kernel, &key), point_id(Kind::Kernel, &key));
    let mut bytes = key.clone().into_bytes();
    bytes[0] ^= 0x01;
    let changed = String::from_utf8(bytes).expect("still ascii");
    assert_ne!(
        point_id(Kind::Kernel, &key),
        point_id(Kind::Kernel, &changed)
    );
    // The kind is part of the id: one key, two kinds, two points.
    assert_ne!(
        point_id(Kind::Kernel, &key),
        point_id(Kind::KernelSymbol, &key)
    );
    // The separator keeps ("ab","c") and ("a","bc") apart.
    assert_ne!(
        point_id(Kind::Kernel, &natural_key(&["ab", "c"])),
        point_id(Kind::Kernel, &natural_key(&["a", "bc"]))
    );
}

#[test]
fn canonical_json_ignores_key_order() {
    let a: Value =
        serde_json::from_str(r#"{"b":1,"a":{"y":[1,{"q":2,"p":3}],"x":null}}"#).expect("valid");
    let b: Value =
        serde_json::from_str(r#"{"a":{"x":null,"y":[1,{"p":3,"q":2}]},"b":1}"#).expect("valid");
    assert_eq!(canonical(&a).expect("a"), canonical(&b).expect("b"));
    assert_eq!(
        String::from_utf8(canonical(&a).expect("a")).expect("utf8"),
        r#"{"a":{"x":null,"y":[1,{"p":3,"q":2}]},"b":1}"#
    );
    // Array order is content, not layout.
    let c = json!({"a": {"x": null, "y": [{"p": 3, "q": 2}, 1]}, "b": 1});
    assert_ne!(canonical(&a).expect("a"), canonical(&c).expect("c"));
}

#[test]
fn canonical_json_round_trips_f64_exactly() {
    for x in [
        0.1_f64,
        1e-300,
        123_456.789,
        2.0_f64.sqrt(),
        f64::MAX,
        5e-324,
    ] {
        let bytes = canonical(&json!({ "v": x })).expect("finite");
        let back: Value = serde_json::from_slice(&bytes).expect("parses");
        assert_eq!(
            back["v"].as_f64().expect("number").to_bits(),
            x.to_bits(),
            "{x:e}"
        );
    }
}

#[test]
fn content_sha256_follows_content_not_recorded_at() {
    let body = |rule: &str| {
        Body::FusionRule(Box::new(FusionRule {
            rule: json!({ "name": rule })
                .as_object()
                .cloned()
                .expect("object"),
            inherited_from: None,
        }))
    };
    let r1 = Record::complete(
        "norm_matmul",
        "a rule",
        "2026-01-01T00:00:00Z",
        false,
        body("r"),
    );
    let mut r2 = r1.clone();
    r2.recorded_at = "2026-02-02T00:00:00Z".into();
    assert_eq!(
        content_sha256(&r1).expect("r1"),
        content_sha256(&r2).expect("r2")
    );
    let r3 = Record::complete(
        "norm_matmul",
        "a rule",
        "2026-01-01T00:00:00Z",
        false,
        body("s"),
    );
    assert_ne!(
        content_sha256(&r1).expect("r1"),
        content_sha256(&r3).expect("r3")
    );
}
