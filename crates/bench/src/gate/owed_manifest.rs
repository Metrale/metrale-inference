// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: The owed-certification manifest `.benchmarks/owed/<model>-<sha>.toml`: what a
//! merge left owed on one model (`owed_prs` writes it), and when it is settled. A settled
//! manifest is deleted by the change that settles it (owner ruling R8), so `.benchmarks/owed/`
//! holds only what is still owed.
//!
//! Owner: bench gate.
//! Invariants:
//! - Pure: [`parse`] reads text, [`render`] writes it, [`settled`] judges standings it is given.
//! - [`render`] and [`parse`] share [`Manifest`], so a manifest `owed_prs` writes parses back to
//!   the same value.
//! - A pair is settled when its model passes the gate at the commit, or when the gate no longer
//!   registers that (hw, checkpoint) for the model: nothing is left to certify. Every other
//!   state, a stale or failing record and an unrecorded model alike, keeps it owed.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::check_models::{ModelStanding, SubjectState};

/// 2026-10-16: The only schema `parse` accepts.
pub const SCHEMA: u32 = 1;

/// 2026-10-16: The directory the manifests live in, relative to the repository root.
pub const DIR: &str = ".benchmarks/owed";

/// 2026-10-16: The manifest of `model` after the merge `merge_sha`, relative to the root.
pub fn path(model: &str, merge_sha: &str) -> String {
    format!("{DIR}/{model}-{merge_sha}.toml")
}

/// 2026-10-16: One owed (gate, hw, checkpoint) of the manifest's model.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pair {
    pub gate: String,
    pub hw: String,
    pub checkpoint: String,
}

/// 2026-10-16: One manifest: the merge that left it owed, the model directory and its pairs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: u32,
    pub merge_sha: String,
    pub model: String,
    pub owed: Vec<Pair>,
}

/// 2026-10-16: The manifest's text: a comment saying what settles it, then the TOML.
pub fn render(m: &Manifest) -> Result<String, String> {
    let body = toml::to_string(m).map_err(|e| e.to_string())?;
    Ok(format!(
        "# Certification owed after {}: a passing record of each entry below,\n\
         # measured at or after that commit, settles it. The change that settles\n\
         # every entry deletes this file.\n{body}",
        m.merge_sha
    ))
}

/// 2026-10-16: A manifest's text, refused when it is not TOML of [`SCHEMA`].
pub fn parse(text: &str) -> Result<Manifest, String> {
    let m: Manifest = toml::from_str(text).map_err(|e| e.to_string())?;
    if m.schema != SCHEMA {
        return Err(format!("schema {} is not {SCHEMA}", m.schema));
    }
    Ok(m)
}

/// 2026-10-16: Whether nothing `m` lists is still owed, given every gate's standings at the
/// commit (`check_models::ModelCheck::standings`).
pub fn settled(m: &Manifest, standings: &BTreeMap<String, Vec<ModelStanding>>) -> bool {
    m.owed.iter().all(|p| {
        let standing = standings.get(&p.gate).and_then(|list| {
            list.iter().find(|s| {
                s.subject.model == m.model
                    && s.subject.hw == p.hw
                    && s.subject.checkpoint == p.checkpoint
            })
        });
        match standing {
            None => true,
            Some(s) => s.state == SubjectState::Pass,
        }
    })
}

#[cfg(test)]
#[path = "owed_manifest_tests.rs"]
mod owed_manifest_tests;
