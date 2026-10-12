// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: What identifies the served model beyond its name (investor-mvp #28, #41): the
//! loaded checkpoint's revision and the launch recipe's digest. The served name is a route
//! and never an identity (mfs DECISIONS PC-R).
//!
//! - Revision: the loader resolves a Hub id to `<cache>/models--<org>--<name>/snapshots/<commit>`
//!   (`model_resolver`), so the snapshot directory names the revision it loaded. Any other
//!   directory has no known revision, and none is reported.
//! - Recipe: `--model-recipe recipe.json`, the `ModelRecipe` document a release ships
//!   (PC-W), identified by its canonical-JSON SHA-256. Without the flag there is none, and
//!   usage records and `RuntimeHealth` write `null`.
//!
//! A recipe pins a source revision. When the loaded snapshot's revision is known and
//! differs, the server refuses to start: every record would otherwise name a recipe that
//! is not what is served.
//!
//! Owner: server (model load).
//! Invariants: [`reconcile`] and [`snapshot_revision`] do no I/O; [`resolve`] only reads the
//! recipe file.

use std::path::Path;

use anyhow::{Context, Result};
use metrale_qci_dump::contract::ids::{GitCommit, Sha256Digest};
use metrale_qci_dump::contract::recipe::LaunchRecipe;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ModelIdentity {
    /// 2026-10-11: The loaded checkpoint's revision, when the loader knows it.
    pub(crate) revision: Option<GitCommit>,
    /// 2026-10-11: The launch recipe's digest, when one was given.
    pub(crate) recipe: Option<Sha256Digest>,
}

/// 2026-10-11: The revision a Hub snapshot directory names: `.../snapshots/<40-hex>`.
pub(crate) fn snapshot_revision(model_dir: &Path) -> Option<GitCommit> {
    let parent = model_dir.parent()?.file_name()?;
    if parent != "snapshots" {
        return None;
    }
    GitCommit::from_hex(model_dir.file_name()?.to_str()?).ok()
}

/// 2026-10-11: The identity of a model loaded from a snapshot of `revision` (when known)
/// under `recipe` (when given). Refuses a recipe that pins another revision.
pub(crate) fn reconcile(
    revision: Option<GitCommit>,
    recipe: Option<&LaunchRecipe>,
) -> Result<ModelIdentity, String> {
    if let (Some(loaded), Some(r)) = (revision, recipe)
        && loaded != r.source.revision
    {
        return Err(format!(
            "the recipe {} pins {}@{}, but the loaded checkpoint is revision {}",
            r.recipe_id,
            r.source.repository,
            r.source.revision.to_hex(),
            loaded.to_hex()
        ));
    }
    Ok(ModelIdentity {
        revision,
        recipe: recipe.map(|r| r.digest),
    })
}

/// 2026-10-11: Read `--model-recipe` (when given) and reconcile it with `model_dir`.
pub(crate) fn resolve(recipe_path: Option<&Path>, model_dir: &Path) -> Result<ModelIdentity> {
    let recipe = recipe_path
        .map(|path| {
            let bytes = std::fs::read(path)
                .with_context(|| format!("--model-recipe: cannot read {}", path.display()))?;
            LaunchRecipe::read(&bytes)
                .map_err(|e| anyhow::anyhow!("--model-recipe {}: {e}", path.display()))
        })
        .transpose()?;
    let identity =
        reconcile(snapshot_revision(model_dir), recipe.as_ref()).map_err(anyhow::Error::msg)?;
    tracing::info!(
        revision = identity
            .revision
            .map(|r| r.to_hex())
            .as_deref()
            .unwrap_or("unknown"),
        recipe = identity
            .recipe
            .map(|r| r.to_hex())
            .as_deref()
            .unwrap_or("none"),
        "model identity"
    );
    Ok(identity)
}

#[cfg(test)]
#[path = "model_identity_tests.rs"]
mod tests;
