// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The metrale pack's `[sources]` table (`kg-config/v1`, design §2.2): the campaign
//! record files, principles, designs, research maps, the memory directory and the LKB. Every
//! field is required and unknown fields are refused (PCND).
//!
//! Owner: metrale-latlas-pack (cut from metrale-kg-derive `config.rs`, 2026-10-23 K0/K1).
//! Invariants: campaign ids are unique; `[sources.lkb]` is validated by `LkbSources::parse`.

use crate::prelude::*;
use latlas_error::{ErrorCode, Result, error};
use serde::Deserialize;

/// 2026-10-23: The config schema id a metrale-kg config carries.
pub const CONFIG_SCHEMA: &str = "kg-config/v1";

/// 2026-10-23: The whole config: the core tables around this pack's sources.
pub type Config = latlas_core::Config<Sources>;

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
    /// 2026-10-23: `[sources.lkb]`, the LKB of one hardware directory (P1), kept as the raw table
    /// `lkb_derive::LkbSources` parses. Absent: the graph holds the campaign records only, and
    /// `la_status owed_lkb` says so.
    #[serde(default)]
    pub lkb: Option<toml::Table>,
}

/// 2026-10-23: One campaign directory's record files.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Campaign {
    /// 2026-10-23: The campaign id tools name as the scope (`glm/megakernel`).
    pub id: String,
    pub ledger: String,
    pub tooling: String,
    pub charter: String,
    /// 2026-10-23: The campaign's run fingerprints (`RUNS.tsv`, P3).
    pub runs: String,
    /// 2026-10-23: The ledger's own validator (`tools/ledger.py`), the validator of record.
    pub ledger_check: String,
    /// 2026-10-23: The instance `la_status owed_lkb` reports when the call names none (declared
    /// here, never guessed; absent = the call must name one).
    #[serde(default)]
    pub default_instance: Option<String>,
}

impl Sources {
    /// 2026-10-23: Every single-file source path (root-relative), sorted.
    pub fn paths(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .principles
            .iter()
            .chain(&self.designs)
            .cloned()
            .collect();
        out.push(self.graph_theory_map.clone());
        out.push(self.mathbomb.clone());
        for c in &self.campaigns {
            out.extend([
                c.ledger.clone(),
                c.tooling.clone(),
                c.charter.clone(),
                c.runs.clone(),
            ]);
        }
        out.sort();
        out.dedup();
        out
    }

    /// 2026-10-23: Refuse duplicate campaign ids and an invalid `[sources.lkb]`.
    pub fn validate(&self) -> Result<()> {
        let mut ids = BTreeSet::new();
        for c in &self.campaigns {
            if !ids.insert(c.id.as_str()) {
                return Err(error!(
                    ErrorCode::LatlasConfig,
                    "sources.campaigns", "duplicate id"
                ));
            }
        }
        if let Some(t) = &self.lkb {
            crate::lkb_derive::LkbSources::parse(t)?;
        }
        Ok(())
    }

    /// 2026-10-23: The campaign named `id`.
    pub fn campaign(&self, id: &str) -> Result<&Campaign> {
        self.campaigns.iter().find(|c| c.id == id).ok_or_else(|| {
            let known: Vec<&str> = self.campaigns.iter().map(|c| c.id.as_str()).collect();
            error!(
                ErrorCode::LatlasToolArgument,
                "scope",
                format!("no campaign `{id}`; known: {}", known.join(", "))
            )
        })
    }
}
