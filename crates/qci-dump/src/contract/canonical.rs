// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Canonical JSON (RFC 8785 for documents without floating point) and the
//! SHA-256 content digest built on it. Objects are written with their members sorted by
//! the UTF-16 code units of the key, no whitespace, strings in serde_json's escaping
//! (which matches RFC 8785: `\b \f \n \r \t \" \\`, other controls as lower-case `\u00xx`,
//! everything else verbatim).
//!
//! Owner: qci-dump (contract binding).
//! Invariants: a document holding a non-integer number has no canonical form here.

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::ids::Sha256Digest;

/// 2026-10-11: Why a document has no canonical form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonicalError {
    /// 2026-10-11: A floating-point number, at this JSON pointer.
    Float(String),
    /// 2026-10-11: The value did not serialize to JSON.
    Serialize(String),
}

/// 2026-10-11: The canonical JSON text of `doc`.
pub fn to_canonical_json<T: Serialize>(doc: &T) -> Result<String, CanonicalError> {
    let value = serde_json::to_value(doc).map_err(|e| CanonicalError::Serialize(e.to_string()))?;
    let mut out = String::new();
    write(&value, "", &mut out)?;
    Ok(out)
}

/// 2026-10-11: SHA-256 over the canonical JSON of `doc`.
pub fn digest<T: Serialize>(doc: &T) -> Result<Sha256Digest, CanonicalError> {
    let text = to_canonical_json(doc)?;
    Ok(sha256(text.as_bytes()))
}

/// 2026-10-11: SHA-256 of raw bytes.
pub fn sha256(bytes: &[u8]) -> Sha256Digest {
    let mut out = [0u8; 32];
    out.copy_from_slice(&Sha256::digest(bytes));
    Sha256Digest(out)
}

fn string(s: &str) -> String {
    Value::String(s.to_string()).to_string()
}

fn write(value: &Value, at: &str, out: &mut String) -> Result<(), CanonicalError> {
    match value {
        Value::Null | Value::Bool(_) | Value::String(_) => out.push_str(&value.to_string()),
        Value::Number(n) => {
            if n.is_f64() {
                return Err(CanonicalError::Float(at.to_string()));
            }
            out.push_str(&n.to_string());
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write(item, &format!("{at}/{i}"), out)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
            out.push('{');
            for (i, key) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&string(key));
                out.push(':');
                write(&map[key], &format!("{at}/{key}"), out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn members_are_sorted_and_whitespace_is_dropped() {
        let doc = json!({"b": [3, {"z": true, "a": null}], "a": "x\ny", "\u{e9}": 1, "B": -2});
        assert_eq!(
            to_canonical_json(&doc).unwrap(),
            r#"{"B":-2,"a":"x\ny","b":[3,{"a":null,"z":true}],"é":1}"#
        );
    }

    #[test]
    fn key_order_is_utf16_not_utf8() {
        let doc = json!({"\u{ff61}": 1, "\u{1f600}": 2});
        assert_eq!(
            to_canonical_json(&doc).unwrap(),
            "{\"\u{1f600}\":2,\"\u{ff61}\":1}"
        );
    }

    #[test]
    fn controls_escape_lower_case_and_floats_have_no_canonical_form() {
        assert_eq!(
            to_canonical_json(&json!("\u{1f}\u{7f}")).unwrap(),
            "\"\\u001f\u{7f}\""
        );
        assert_eq!(
            to_canonical_json(&json!({"a": [1, 2.5]})),
            Err(CanonicalError::Float("/a/1".to_string()))
        );
    }

    #[test]
    fn the_digest_ignores_input_member_order() {
        let a = digest(&json!({"x": 1, "y": {"p": 2, "q": 3}})).unwrap();
        let b = digest(&json!({"y": {"q": 3, "p": 2}, "x": 1})).unwrap();
        assert_eq!(a, b);
        assert_eq!(
            sha256(b"abc").to_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_ne!(a, digest(&json!({"x": 2, "y": {"p": 2, "q": 3}})).unwrap());
    }
}
