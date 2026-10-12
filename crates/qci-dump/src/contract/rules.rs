// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The rules of `schemas/platform/v1/common.v1.schema.json` that serde cannot
//! express, one function each, named after the schema `$defs` entry they implement. A
//! failed rule is a [`Violation`] naming the JSON pointer of the offending value.
//!
//! Owner: qci-dump (contract binding).
//! Invariants: lengths count Unicode scalar values, as JSON Schema `maxLength` does.

use serde::{Deserialize, Deserializer};

use super::PLATFORM_VERSION;

/// 2026-10-11: The JSON pointer of the first value that broke a rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation(pub String);

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid field {}", self.0)
    }
}

/// 2026-10-11: `Ok` when `ok`, else a violation at `at`.
pub fn require(ok: bool, at: &str) -> Result<(), Violation> {
    if ok {
        Ok(())
    } else {
        Err(Violation(at.to_string()))
    }
}

/// 2026-10-11: `version`: exactly [`PLATFORM_VERSION`].
pub fn version(v: u32) -> Result<(), Violation> {
    require(v == PLATFORM_VERSION, "/version")
}

/// 2026-10-11: `text`: 1 to 256 characters.
pub fn text(s: &str) -> bool {
    (1..=256).contains(&s.chars().count())
}

/// 2026-10-11: `model_name`: `^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$`, the route a client
/// names (mfs DECISIONS PC-R).
pub fn model_name(s: &str) -> bool {
    let b = s.as_bytes();
    (1..=128).contains(&b.len())
        && b[0].is_ascii_alphanumeric()
        && b.iter()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b':' | b'/' | b'-'))
}

/// 2026-10-11: `recipe_id`: `^[a-z0-9][a-z0-9._-]{0,63}(/[a-z0-9][a-z0-9._-]{0,63})?$`.
pub fn recipe_id(s: &str) -> bool {
    let part = |p: &str| {
        let b = p.as_bytes();
        (1..=64).contains(&b.len())
            && (b[0].is_ascii_lowercase() || b[0].is_ascii_digit())
            && b.iter().all(|c| {
                c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, b'.' | b'_' | b'-')
            })
    };
    match s.split_once('/') {
        Some((family, stem)) => part(family) && part(stem),
        None => part(s),
    }
}

/// 2026-10-11: `value`: at most 1024 characters.
pub fn value(s: &str) -> bool {
    s.chars().count() <= 1024
}

/// 2026-10-11: `timestamp_ms`: never before the epoch.
pub fn timestamp(ms: i64) -> bool {
    ms >= 0
}

/// 2026-10-11: `dns_label`: `^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$`.
pub fn dns_label(s: &str) -> bool {
    let b = s.as_bytes();
    let inner = |c: &u8| c.is_ascii_lowercase() || c.is_ascii_digit();
    match b {
        [] => false,
        [one] => inner(one),
        [first, middle @ .., last] => {
            b.len() <= 63
                && inner(first)
                && inner(last)
                && middle.iter().all(|c| inner(c) || *c == b'-')
        }
    }
}

/// 2026-10-11: `flag_key`: `^[a-z][a-z0-9_]{0,63}$`.
pub fn flag_key(s: &str) -> bool {
    let b = s.as_bytes();
    (1..=64).contains(&b.len())
        && b[0].is_ascii_lowercase()
        && b.iter()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == b'_')
}

/// 2026-10-11: `repository`: `owner/name`, each part
/// `^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$`.
pub fn repository(s: &str) -> bool {
    let part = |p: &str| {
        let b = p.as_bytes();
        (1..=96).contains(&b.len())
            && b[0].is_ascii_alphanumeric()
            && b.iter()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-'))
    };
    matches!(s.split_once('/'), Some((owner, name)) if part(owner) && part(name))
}

/// 2026-10-11: Deserialize a field that must be written, as a value or `null`; serde would
/// otherwise read a missing `Option` field as `None`.
pub(crate) fn present<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    d: D,
) -> Result<Option<T>, D::Error> {
    Option::deserialize(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_rules_follow_the_schema_patterns() {
        assert!(text("x") && text(&"é".repeat(256)) && !text("") && !text(&"a".repeat(257)));
        assert!(value("") && value(&"a".repeat(1024)) && !value(&"a".repeat(1025)));
        for ok in ["a", "gb10", "strix-halo", &"a".repeat(63)] {
            assert!(dns_label(ok), "{ok}");
        }
        for bad in ["", "-a", "a-", "Gb10", "a_b", &"a".repeat(64)] {
            assert!(!dns_label(bad), "{bad}");
        }
        assert!(flag_key("concurrency") && flag_key("prompt_set") && flag_key("x1"));
        for bad in ["", "Seed", "1x", "max-tokens", &"a".repeat(65)] {
            assert!(!flag_key(bad), "{bad}");
        }
        assert!(repository("openai/gpt-oss-20b") && repository("unsloth/gemma-4-26B-A4B-it-GGUF"));
        for bad in ["gemma", "a/b/c", "/b", "a/", "-a/b", "a b/c"] {
            assert!(!repository(bad), "{bad}");
        }
        for ok in [
            "gpt-oss-20b",
            "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4",
            "m:v1.2_x",
            &"a".repeat(128),
        ] {
            assert!(model_name(ok), "{ok}");
        }
        for bad in ["", "gpt oss", "-m", "/m", "m\n", "é", &"a".repeat(129)] {
            assert!(!model_name(bad), "{bad:?}");
        }
        for ok in ["gpt-oss/gpt-oss-20b-mxfp4", "qwen3.6-27b", "a/b"] {
            assert!(recipe_id(ok), "{ok}");
        }
        for bad in [
            "",
            "GPT/OSS",
            "a/b/c",
            "/b",
            "a/",
            "-a",
            "a b",
            &"a".repeat(65),
        ] {
            assert!(!recipe_id(bad), "{bad}");
        }
        assert!(timestamp(0) && !timestamp(-1));
        assert_eq!(version(1), Ok(()));
        assert_eq!(version(2), Err(Violation("/version".into())));
    }
}
