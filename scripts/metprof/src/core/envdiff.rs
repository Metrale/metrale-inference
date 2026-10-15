// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: environment diff (pure): an A/B must differ in exactly one
//! lever.
//!
//! An environment is either `/proc/PID/environ` content (NUL-separated) or a
//! text file of KEY=VALUE lines (blank lines and lines starting with # are
//! skipped). Keys matching an --ignore pattern (per-process noise such as
//! PIDs, ports, log paths) are dropped before the comparison. Anything other
//! than exactly one differing key is an error that lists every difference.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

use regex::Regex;

use crate::core::error::Result;
use crate::core::repr::quote;
use crate::fail;

static KEY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*$").expect("the env key pattern is a valid regex")
});

pub type Env = BTreeMap<String, String>;

pub fn parse_env(raw: &[u8], place: &str) -> Result<Env> {
    let Ok(text) = std::str::from_utf8(raw) else {
        fail!("{place}: the environment is not UTF-8");
    };
    let entries: Vec<&str> = if text.contains('\0') {
        text.split('\0').filter(|e| !e.is_empty()).collect()
    } else {
        text.lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect()
    };
    let mut env = Env::new();
    for (n, e) in entries.iter().enumerate() {
        let parsed = e.split_once('=').filter(|(k, _)| KEY.is_match(k));
        let Some((key, value)) = parsed else {
            let head: String = e.chars().take(60).collect();
            fail!(
                "{place}: entry {} is not KEY=VALUE: {}",
                n + 1,
                quote(&head)
            );
        };
        if env.get(key).is_some_and(|v| v != value) {
            fail!("{place}: {key} is set twice with different values");
        }
        env.insert(key.to_string(), value.to_string());
    }
    Ok(env)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvChange {
    pub key: String,
    pub a: Option<String>,
    pub b: Option<String>,
}

pub fn env_diff(a: &Env, b: &Env, ignore: &[Regex]) -> Vec<EnvChange> {
    let keys: BTreeSet<&String> = a
        .keys()
        .chain(b.keys())
        .filter(|k| !ignore.iter().any(|p| p.is_match(k)))
        .collect();
    keys.into_iter()
        .filter(|k| a.get(*k) != b.get(*k))
        .map(|k| EnvChange {
            key: k.clone(),
            a: a.get(k).cloned(),
            b: b.get(k).cloned(),
        })
        .collect()
}

fn shown(v: &Option<String>) -> String {
    v.as_deref()
        .map(quote)
        .unwrap_or_else(|| "None".to_string())
}

pub fn one_lever(diffs: &[EnvChange]) -> Result<&EnvChange> {
    if let [only] = diffs {
        return Ok(only);
    }
    let listing: Vec<String> = diffs
        .iter()
        .map(|d| format!("{}: {} -> {}", d.key, shown(&d.a), shown(&d.b)))
        .collect();
    let listing = if listing.is_empty() {
        "none".to_string()
    } else {
        listing.join("; ")
    };
    fail!(
        "an A/B must differ in exactly one lever, found {} differing keys ({listing})",
        diffs.len()
    );
}
