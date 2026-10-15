// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: what a result was measured on (pure): the provenance file a
//! kernel tree requires, checked, and its digests.
//!
//! ```toml
//! binary = "path/to/serve-binary"   # or binary_sha256 = "<64 hex>"
//! git_sha = "<hex>"
//! git_dirty = false
//! env = "path/to/env-dump"          # or env_sha256 = "<64 hex>"
//! recipe = "org/recipe-id"
//! isl = 1024
//! osl = 128
//! concurrency = 1
//! rank = 0
//! ```
//!
//! Every key is required, and any value may be the explicit string
//! `"unknown"`: an unrecorded fact is stated as unknown, never left out or
//! guessed. A path is hashed by the caller (the core never reads a file).

use serde_json::{Map, Value, json};
use sha2::{Digest as _, Sha256};

use crate::core::error::Result;
use crate::core::families::check_keys;
use crate::core::repr::{quote, toml_value};
use crate::fail;

pub const UNKNOWN: &str = "unknown";
const REQUIRED: [&str; 7] = [
    "git_sha",
    "git_dirty",
    "recipe",
    "isl",
    "osl",
    "concurrency",
    "rank",
];
const OPTIONAL: [&str; 4] = ["binary", "binary_sha256", "env", "env_sha256"];

/// A digest stated in the file, or a file the caller must hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Digest {
    Path(String),
    Sha256(String),
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProvenanceSpec {
    pub binary: Digest,
    pub env: Digest,
    /// git_sha, git_dirty, recipe, isl, osl, concurrency, rank, in that order.
    pub fields: Map<String, Value>,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn is_unknown(v: &toml::Value) -> bool {
    v.as_str() == Some(UNKNOWN)
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn digest(doc: &toml::Table, path_key: &str, sha_key: &str) -> Result<Digest> {
    match (doc.get(path_key), doc.get(sha_key)) {
        (Some(p), None) if is_unknown(p) => Ok(Digest::Unknown),
        (Some(p), None) => match p.as_str().filter(|s| !s.is_empty()) {
            Some(s) => Ok(Digest::Path(s.to_string())),
            None => fail!(
                "provenance: {path_key} must be a path or \"unknown\", got {}",
                toml_value(p)
            ),
        },
        (None, Some(h)) if is_unknown(h) => Ok(Digest::Unknown),
        (None, Some(h)) => match h.as_str().filter(|s| is_sha256(s)) {
            Some(s) => Ok(Digest::Sha256(s.to_string())),
            None => fail!(
                "provenance: {sha_key} must be 64 lower-case hex digits or \"unknown\", got {}",
                toml_value(h)
            ),
        },
        _ => fail!("provenance: exactly one of {path_key} and {sha_key} is required"),
    }
}

pub fn parse_provenance(doc: &toml::Table) -> Result<ProvenanceSpec> {
    check_keys("provenance", doc, &REQUIRED, &OPTIONAL)?;
    let binary = digest(doc, "binary", "binary_sha256")?;
    let env = digest(doc, "env", "env_sha256")?;
    let mut fields = Map::new();
    for key in REQUIRED {
        let v = &doc[key];
        let value = if is_unknown(v) {
            json!(UNKNOWN)
        } else {
            let ok = match key {
                "git_sha" => v
                    .as_str()
                    .filter(|s| {
                        s.len() >= 7 && s.len() <= 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
                    })
                    .map(|s| json!(s.to_ascii_lowercase())),
                "git_dirty" => v.as_bool().map(|b| json!(b)),
                "recipe" => v.as_str().filter(|s| !s.is_empty()).map(|s| json!(s)),
                _ => v.as_integer().filter(|i| *i >= 0).map(|i| json!(i)),
            };
            match ok {
                Some(x) => x,
                None => fail!(
                    "provenance: {} has an invalid value {} (state \"unknown\" if it was not recorded)",
                    quote(key),
                    toml_value(v)
                ),
            }
        };
        fields.insert(key.to_string(), value);
    }
    Ok(ProvenanceSpec {
        binary,
        env,
        fields,
    })
}

/// The provenance object of one tree. `binary` and `env` are the resolved
/// digests ("unknown" when not recorded).
pub fn render(
    spec: &ProvenanceSpec,
    capture_sha256: &str,
    binary: &str,
    env: &str,
    device: i64,
    segment: &str,
    circuit: Option<Value>,
) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("capture_sha256".into(), json!(capture_sha256));
    m.insert("binary_sha256".into(), json!(binary));
    for (k, v) in &spec.fields {
        if k == "recipe" {
            m.insert("env_sha256".into(), json!(env));
        }
        m.insert(k.clone(), v.clone());
    }
    m.insert("device".into(), json!(device));
    m.insert("segment".into(), json!(segment));
    if let Some(c) = circuit {
        m.insert("circuit".into(), c);
    }
    m.insert("metprof_version".into(), json!(env!("CARGO_PKG_VERSION")));
    m
}
