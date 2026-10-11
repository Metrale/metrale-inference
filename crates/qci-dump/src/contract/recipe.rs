// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The `ModelRecipe` v1 document a runtime was launched from (mfs DECISIONS
//! PC-W: release engineering produces it and ships it as `recipe.json` beside the release
//! manifest). The runtime identifies it the way the manifest does: by the SHA-256 of its
//! canonical JSON, never of a YAML launch recipe. That digest is the `recipe` of every
//! `InferenceUsage` record and of `RuntimeHealth`.
//!
//! The runtime does not admit recipes; Control does, against the whole schema. It reads the
//! parts it uses and refuses a document that cannot be the recipe it claims to be: another
//! version, a floating-point number (no canonical form), a malformed recipe id or
//! repository, or a source revision that is not a full commit.
//!
//! Owner: qci-dump (contract binding).
//! Invariants: the digest covers the document exactly as given, every key included.

use serde::Deserialize;
use serde_json::Value;

use super::canonical::{self, CanonicalError};
use super::ids::{GitCommit, Sha256Digest};
use super::rules::{self, require};

/// 2026-10-11: The checkpoint a recipe pins: `model.source`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSource {
    pub repository: String,
    pub revision: GitCommit,
}

/// 2026-10-11: A recipe document the runtime was launched from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchRecipe {
    /// 2026-10-11: SHA-256 of the document's canonical JSON.
    pub digest: Sha256Digest,
    pub recipe_id: String,
    pub source: ModelSource,
}

#[derive(Deserialize)]
struct Head {
    version: u32,
    recipe_id: String,
    model: HeadModel,
}

#[derive(Deserialize)]
struct HeadModel {
    source: HeadSource,
}

#[derive(Deserialize)]
struct HeadSource {
    repository: String,
    revision: String,
}

impl LaunchRecipe {
    /// 2026-10-11: Read a `ModelRecipe` v1 document from its bytes. The error says why it
    /// is not one the runtime can be launched from.
    pub fn read(bytes: &[u8]) -> Result<Self, String> {
        let doc: Value =
            serde_json::from_slice(bytes).map_err(|e| format!("not a JSON document: {e}"))?;
        let head: Head = serde_json::from_value(doc.clone())
            .map_err(|e| format!("not a ModelRecipe document: {e}"))?;
        let rule = |r: Result<(), rules::Violation>| r.map_err(|v| v.to_string());
        rule(rules::version(head.version))?;
        rule(require(rules::recipe_id(&head.recipe_id), "/recipe_id"))?;
        let s = head.model.source;
        rule(require(
            rules::repository(&s.repository),
            "/model/source/repository",
        ))?;
        let revision = GitCommit::from_hex(&s.revision)
            .map_err(|_| "invalid field /model/source/revision (not a full commit)".to_string())?;
        let digest = canonical::digest(&doc).map_err(|e| match e {
            CanonicalError::Float(at) => format!("invalid field {at} (not an integer)"),
            CanonicalError::Serialize(e) => e,
        })?;
        Ok(Self {
            digest,
            recipe_id: head.recipe_id,
            source: ModelSource {
                repository: s.repository,
                revision,
            },
        })
    }
}
