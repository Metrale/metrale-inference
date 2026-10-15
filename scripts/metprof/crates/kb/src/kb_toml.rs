// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `kb.toml`, where metprof's KB sink writes (DESIGN §3.1), as
//! pure types and a parser. Nothing has a default: every key is required,
//! paths are absolute, and sealing the local export is either a key file or
//! the explicit `seal = "off"`. The embedder is named by the path of its
//! `embedding.toml` (ADDENDUM-EMBEDDINGS E2), the one place it is configured.
//! The caller reads the files; this module reads nothing.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::port::Identity;
use crate::{Error, Result, bail};

pub const KB_TOML_SCHEMA: i64 = 1;

/// Where the sink writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KbPaths {
    pub spool_dir: PathBuf,
    pub durable_dir: PathBuf,
    pub export_dir: PathBuf,
}

/// How the local export is sealed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SealConfig {
    /// `seal = "off"`: only `kkb.jsonl` is written.
    Off,
    /// `seal_key_file`: the 32-byte hex local export key.
    KeyFile(PathBuf),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KbConfig {
    pub paths: KbPaths,
    pub seal: SealConfig,
    /// The `embedding.toml` naming the embedder.
    pub embedding: PathBuf,
    pub identity: Identity,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Raw {
    schema: i64,
    spool_dir: String,
    durable_dir: String,
    export_dir: String,
    seal_key_file: Option<String>,
    seal: Option<String>,
    embedding: String,
    identity: Identity,
}

fn absolute(key: &str, v: String) -> Result<PathBuf> {
    if !Path::new(&v).is_absolute() {
        bail!("kb.toml: {key} {v:?} is not an absolute path");
    }
    Ok(PathBuf::from(v))
}

/// Parse `kb.toml`.
pub fn parse(text: &str) -> Result<KbConfig> {
    let raw: Raw = toml::from_str(text).map_err(|e| Error(format!("kb.toml: {}", e.message())))?;
    if raw.schema != KB_TOML_SCHEMA {
        bail!("kb.toml: schema {} is not {KB_TOML_SCHEMA}", raw.schema);
    }
    let seal = match (raw.seal_key_file, raw.seal.as_deref()) {
        (Some(f), None) => SealConfig::KeyFile(absolute("seal_key_file", f)?),
        (None, Some("off")) => SealConfig::Off,
        (None, Some(other)) => bail!("kb.toml: seal {other:?} is not \"off\""),
        (Some(_), Some(_)) => bail!("kb.toml: seal_key_file and seal = \"off\" are exclusive"),
        (None, None) => bail!("kb.toml: seal_key_file, or seal = \"off\", is required"),
    };
    let id = &raw.identity;
    for (k, v) in [
        ("box_alias", &id.box_alias),
        ("campaign", &id.campaign),
        ("hardware_class", &id.hardware_class),
    ] {
        if v.trim().is_empty() {
            bail!("kb.toml: [identity] {k} is empty");
        }
    }
    Ok(KbConfig {
        paths: KbPaths {
            spool_dir: absolute("spool_dir", raw.spool_dir)?,
            durable_dir: absolute("durable_dir", raw.durable_dir)?,
            export_dir: absolute("export_dir", raw.export_dir)?,
        },
        seal,
        embedding: absolute("embedding", raw.embedding)?,
        identity: raw.identity,
    })
}
