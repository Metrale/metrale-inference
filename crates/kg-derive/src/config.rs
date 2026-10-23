// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The knowledge-graph config (`kg-config/v1`; design §2.2). Every field is required
//! and unknown fields are refused (PCND): a missing field fails at start with `KgConfig` and the
//! field's name. The source list is explicit; nothing is discovered except the files of the one
//! named memory directory.
//!
//! Owner: metrale-kg-derive.
//! Invariants:
//! - Source paths are relative to `sources.root`; none names a credential file
//!   ([`crate::secrets::name_is_secret`]).
//! - The embedder is either `openrouter` with every field, or the declared `unavailable` mode with
//!   its reason. There is no third state.

use crate::prelude::*;
use crate::secrets::name_is_secret;
use crate::settings::EmbedLayer;
use metrale_error::{ErrorCode, Result, error};
use serde::Deserialize;

/// 2026-10-23: The config schema id.
pub const CONFIG_SCHEMA: &str = "kg-config/v1";

/// 2026-10-23: The whole config.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub schema: String,
    pub sources: Sources,
    /// 2026-10-23: The project layer of the embedder settings ([`crate::settings`]); it
    /// supersedes the global layer per key. An absent table sets no key (layering, not a default).
    #[serde(default)]
    pub embedder: EmbedLayer,
    pub limits: Limits,
    pub store: Store,
    pub search: Search,
    pub validator: Validator,
    /// 2026-10-23: The root-relative path this config was read from, set by the host that read
    /// it (never by the file), so the config's own node has a `src`.
    #[serde(skip)]
    pub origin: String,
}

/// 2026-10-23: How a file's own validator runs (the interpreter is named, never looked up).
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Validator {
    /// 2026-10-23: The absolute path of the Python interpreter `ledger.py` runs under.
    pub python: String,
}

/// 2026-10-23: The SSOT files the graph is derived from.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Sources {
    /// 2026-10-23: The absolute directory every other path is relative to.
    pub root: String,
    pub campaigns: Vec<Campaign>,
    pub principles: Vec<String>,
    pub designs: Vec<String>,
    pub graph_theory_map: String,
    pub mathbomb: String,
    pub memory_dir: String,
    /// 2026-10-23: The LKB checkout (root-relative), read through metrale-circuit (P1).
    pub lkb_repo: String,
}

/// 2026-10-23: One campaign directory's record files.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Campaign {
    /// 2026-10-23: The campaign id tools name (`glm/megakernel`).
    pub id: String,
    pub ledger: String,
    pub tooling: String,
    pub charter: String,
    /// 2026-10-23: The ledger's own validator (`tools/ledger.py`), the validator of record.
    pub ledger_check: String,
}

/// 2026-10-23: Tool result limits.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub max_rows: u32,
    pub cypher_timeout_ms: u64,
    pub token_budget: u32,
}

/// 2026-10-23: Where the derived database lives (rebuildable, not committed).
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Store {
    pub data_dir: String,
    /// 2026-10-23: An absolute directory for private copies (validator runs, atomic writes).
    pub scratch_dir: String,
}

/// 2026-10-23: The hybrid ranking weights (ontology §5), declared and printed in results.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Search {
    pub w_cosine: f64,
    pub w_hops: f64,
    pub w_kind: f64,
}

impl Config {
    /// 2026-10-23: Parse and validate a config text.
    pub fn parse(text: &str) -> Result<Config> {
        let cfg: Config = toml::from_str(text)
            .map_err(|e| error!(ErrorCode::KgConfig, "(file)", e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if self.schema != CONFIG_SCHEMA {
            return Err(error!(
                ErrorCode::KgConfig,
                "schema",
                format!("must be {CONFIG_SCHEMA}")
            ));
        }
        if !self.sources.root.starts_with('/') {
            return Err(error!(
                ErrorCode::KgConfig,
                "sources.root", "must be absolute"
            ));
        }
        for p in self.source_paths() {
            if p.starts_with('/') || p.split('/').any(|c| c == "..") {
                return Err(error!(
                    ErrorCode::KgConfig,
                    p, "must be root-relative, without .."
                ));
            }
            if name_is_secret(&p) {
                return Err(error!(ErrorCode::KgConfig, p, "names a credential file"));
            }
        }
        let mut ids = BTreeSet::new();
        for c in &self.sources.campaigns {
            if !ids.insert(c.id.as_str()) {
                return Err(error!(
                    ErrorCode::KgConfig,
                    "sources.campaigns", "duplicate id"
                ));
            }
        }
        let w = self.search;
        if !(w.w_cosine >= 0.0 && w.w_hops >= 0.0 && w.w_kind >= 0.0) {
            return Err(error!(
                ErrorCode::KgConfig,
                "search", "weights must be >= 0"
            ));
        }
        Ok(())
    }

    /// 2026-10-23: Every single-file source path (root-relative), sorted.
    pub fn source_paths(&self) -> Vec<String> {
        let s = &self.sources;
        let mut out: Vec<String> = s.principles.iter().chain(&s.designs).cloned().collect();
        out.push(s.graph_theory_map.clone());
        out.push(s.mathbomb.clone());
        if !self.origin.is_empty() {
            out.push(self.origin.clone());
        }
        for c in &s.campaigns {
            out.extend([c.ledger.clone(), c.tooling.clone(), c.charter.clone()]);
        }
        out.sort();
        out.dedup();
        out
    }

    /// 2026-10-23: The campaign named `id`.
    pub fn campaign(&self, id: &str) -> Result<&Campaign> {
        self.sources
            .campaigns
            .iter()
            .find(|c| c.id == id)
            .ok_or_else(|| {
                let known: Vec<&str> = self
                    .sources
                    .campaigns
                    .iter()
                    .map(|c| c.id.as_str())
                    .collect();
                error!(
                    ErrorCode::KgToolArgument,
                    "campaign",
                    format!("no campaign `{id}`; known: {}", known.join(", "))
                )
            })
    }
}
