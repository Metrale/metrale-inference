// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The recipe a gate serves, as part of what its records attest.
//!
//! A gate serves the recipe its BENCH entry names, read from `recipes/<id>.yaml` in the tree
//! under test ([`read_in_tree`]). A record carries the canonical content hash of what it served
//! ([`GateRecord::served_recipe_sha256`]), and stands at a later commit only while that commit's
//! recipe hashes the same ([`recipe_standing`]).
//!
//! The canonical form ([`canonical`]) is the parsed recipe (`metrale_config::recipe_yaml`), so
//! comments, blank lines, quoting and key order do not count. It drops `metadata`, which no
//! serve reads, and applies the run's serve overrides to `defaults`, so a recipe key that an
//! override pins does not count either.
//!
//! Owner: bench gate.
//! Invariants:
//! - [`recipe_standing`] answers `Same` only when both hashes exist and agree. A recipe the
//!   tree does not have, one that does not parse, and a record without a served-recipe hash
//!   are `Changed`, never `Same`: such a record was served from a recipe outside the tree
//!   (a node's cached index), so nothing attests what it served.
//! - A record with no `served_by` (an operator's own endpoint) served no recipe, so the
//!   recipe does not bear on its standing.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use metrale_config::recipe_yaml::{self, Yaml};
use sha2::{Digest, Sha256};

use super::record::GateRecord;

/// 2026-10-02: The top-level recipe key no serve reads (description, maintainer, dates).
const DISPLAY_ONLY: &str = "metadata";

/// 2026-10-02: Where recipe `id` (`<family>/<stem>`) lives in a tree.
pub fn recipe_path(id: &str) -> String {
    format!("recipes/{id}.yaml")
}

/// 2026-10-02: The text of recipe `id` in the tree at `root`. A recipe the tree does not have
/// is refused: a gate serves only the recipe committed beside its BENCH.toml, never a node's
/// cached copy.
pub fn read_in_tree(root: &Path, id: &str) -> Result<String> {
    let path = root.join(recipe_path(id));
    std::fs::read_to_string(&path).with_context(|| {
        format!(
            "recipe {id:?} is not in this tree at {}. A gate serves only the recipe committed \
             in the tree under test; it never falls back to a cached recipe index",
            path.display()
        )
    })
}

/// 2026-10-02: The served form of a recipe: parsed, without `metadata`, with `overrides`
/// replacing or adding `defaults` keys.
pub fn canonical(text: &str, overrides: &BTreeMap<String, String>) -> Result<Yaml> {
    let Yaml::Map(mut top) = recipe_yaml::parse(text)? else {
        bail!("a recipe must be a mapping");
    };
    top.remove(DISPLAY_ONLY);
    let Some(Yaml::Map(defaults)) = top.get_mut("defaults") else {
        bail!("a recipe's `defaults:` must be a mapping");
    };
    for (key, value) in overrides {
        defaults.insert(key.clone(), Yaml::Scalar(value.clone()));
    }
    Ok(Yaml::Map(top))
}

/// 2026-10-02: Hex sha256 of [`canonical`], over a JSON rendering whose maps are in key order.
pub fn content_sha256(text: &str, overrides: &BTreeMap<String, String>) -> Result<String> {
    let rendered = serde_json::to_string(&json(&canonical(text, overrides)?))?;
    Ok(Sha256::digest(rendered.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

fn json(y: &Yaml) -> serde_json::Value {
    match y {
        Yaml::Scalar(s) => serde_json::Value::String(s.clone()),
        Yaml::List(items) => serde_json::Value::Array(items.iter().map(json).collect()),
        Yaml::Map(m) => serde_json::Value::Array(
            m.iter()
                .map(|(k, v)| serde_json::Value::Array(vec![k.clone().into(), json(v)]))
                .collect(),
        ),
    }
}

/// 2026-10-02: Whether the recipe a record served is still what its gate serves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecipeStanding {
    /// 2026-10-02: Same recipe id, same canonical content; or no recipe was served.
    Same,
    /// 2026-10-02: Another recipe id, other content, or nothing to compare. Carries the
    /// recipe path the verdict names.
    Changed(String),
}

/// 2026-10-02: Compare what `record` served with what its gate serves in the tree at `root`:
/// the recipe the BENCH entry for the record's (hardware, model) names, rendered with the
/// record's serve overrides (which `check_record` requires to equal the entry's pins).
///
/// The record side is its `served_recipe_sha256`; a record without one is `Changed`.
pub fn recipe_standing(root: &Path, record: &GateRecord) -> RecipeStanding {
    let Some(served) = record.served_by.as_deref() else {
        return RecipeStanding::Same;
    };
    let head_id = super::read_baseline(root, super::group::serve_baseline_id(&record.benchmark_id))
        .ok()
        .and_then(|b| {
            b.resolve(&record.hardware.gate_key(), Some(&record.target_model))
                .ok()
                .and_then(|(_, entry)| entry.recipe.clone())
        });
    let Some(head_id) = head_id else {
        return RecipeStanding::Changed(recipe_path(served));
    };
    let changed = || RecipeStanding::Changed(recipe_path(&head_id));
    if head_id != served {
        return changed();
    }
    let Ok(head) = read_in_tree(root, &head_id)
        .and_then(|text| content_sha256(&text, &record.serve_overrides))
    else {
        return changed();
    };
    let Some(recorded) = &record.served_recipe_sha256 else {
        return changed();
    };
    if *recorded == head {
        RecipeStanding::Same
    } else {
        changed()
    }
}

impl GateRecord {
    /// 2026-10-02: Attach the canonical content hash of the recipe the gate served
    /// ([`content_sha256`] over the recipe text and the run's serve overrides).
    #[must_use]
    pub fn with_served_recipe(mut self, sha256: String) -> Self {
        self.served_recipe_sha256 = Some(sha256);
        self
    }
}
