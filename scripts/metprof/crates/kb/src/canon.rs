// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: canonical JSON (object keys in byte order at every level, no
//! whitespace, f64 in the shortest form that round-trips) and the SHA-256
//! digests taken over it. Whatever order a map was built in, its canonical
//! bytes are the same.

use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::schema::Record;
use crate::{Error, Result};

fn sorted(v: &Value) -> Value {
    match v {
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            let mut out = Map::new();
            for k in keys {
                out.insert(k.clone(), sorted(&m[k]));
            }
            Value::Object(out)
        }
        Value::Array(a) => Value::Array(a.iter().map(sorted).collect()),
        other => other.clone(),
    }
}

/// The canonical bytes of any serialisable value.
pub fn canonical<T: Serialize>(x: &T) -> Result<Vec<u8>> {
    let v = serde_json::to_value(x).map_err(|e| Error(format!("canonical JSON: {e}")))?;
    serde_json::to_vec(&sorted(&v)).map_err(|e| Error(format!("canonical JSON: {e}")))
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// SHA-256 of the canonical JSON of `x`.
pub fn digest<T: Serialize>(x: &T) -> Result<String> {
    Ok(sha256_hex(&canonical(x)?))
}

/// A record's `content_sha256` (over `Record::content`).
pub fn content_sha256(r: &Record) -> Result<String> {
    digest(&r.content()?)
}
