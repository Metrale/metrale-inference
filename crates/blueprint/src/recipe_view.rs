// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: `RecipeView`: a recipe as a projection of a blueprint, both directions.
//!
//! | Blueprint field | recipe v2 key |
//! |---|---|
//! | `checkpoint.repo` | `model` |
//! | `hardware.world_size` | `min_nodes` |
//! | `policy.flags` | `defaults:` (minus host keys) |
//! | `settings.levers`, `settings.passthrough` | `env:` |
//! | unsigned `Resources` | `port`, `host`, `chat_template` (the path), `container`, the id |
//!
//! Owner: blueprint.
//! Invariants:
//! - Lossless: [`recipe_parts`] of [`project`] gives back the same `model`, `min_nodes`,
//!   `defaults` and `env`, so the rendered `met serve` argv is identical (the server's
//!   `to_recipe_reproduces_argv` test, over every recipe in the tree).
//! - A `METRALE_*` lever the table does not declare, and a lever value that is not its type's
//!   canonical spelling, are refused (`BlueprintRecipe`), never typed as text.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;

use metrale_error::{Error, ErrorCode, error};
use metrale_settings::spec::Rule;

use crate::ids::Sha256;
use crate::types::{
    Blueprint, Fixed, FlagValue, LeverValue, Resources, ServePolicy, ServeSettings,
};

/// 2026-10-22: `defaults:` keys that are host resources: unsigned, supplied at launch.
pub const RESOURCE_KEYS: &[&str] = &["port", "host", "chat_template"];

/// 2026-10-22: The recipe fields the projection reads and writes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecipeParts {
    /// 2026-10-22: `family/stem`.
    pub id: String,
    /// 2026-10-22: `recipe_version`.
    pub version: String,
    /// 2026-10-22: `model`.
    pub model: String,
    /// 2026-10-22: `runtime`.
    pub runtime: Option<String>,
    /// 2026-10-22: `container`.
    pub container: String,
    /// 2026-10-22: `min_nodes`.
    pub min_nodes: u32,
    /// 2026-10-22: `defaults:`.
    pub defaults: BTreeMap<String, String>,
    /// 2026-10-22: `env:`.
    pub env: BTreeMap<String, String>,
}

/// 2026-10-22: A recipe split into blueprint parts. `policy.chat_template` is `None`: the
/// builder pins the template's content.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Projection {
    /// 2026-10-22: `checkpoint.repo`.
    pub model: String,
    /// 2026-10-22: `hardware.world_size`.
    pub world_size: u32,
    /// 2026-10-22: The signed flags.
    pub policy: ServePolicy,
    /// 2026-10-22: The levers.
    pub settings: ServeSettings,
    /// 2026-10-22: The unsigned host resources.
    pub resources: Resources,
}

/// 2026-10-22: SHA-256 over the published lever table, row by row.
pub fn lever_table_digest() -> Sha256 {
    let mut rows = alloc::vec::Vec::new();
    for l in metrale_settings::groups::by_name() {
        let rule = metrale_settings::groups::rule_of(l.env).map_or("", Rule::as_str);
        rows.extend(metrale_blueprint_trust::lv_cat(&[
            l.env.as_bytes(),
            l.path.as_bytes(),
            l.ty.as_str().as_bytes(),
            rule.as_bytes(),
            l.default.as_bytes(),
            l.flag.unwrap_or("").as_bytes(),
            l.class.as_str().as_bytes(),
        ]));
    }
    Sha256::of(&[b"kbp-levers\0", &rows])
}

fn lever(id: &str, name: &str, text: &str) -> Result<LeverValue, Error> {
    let refuse = |why: String| error!(ErrorCode::BlueprintRecipe, String::from(id), why);
    let rule = metrale_settings::groups::rule_of(name)
        .ok_or_else(|| refuse(format!("env {name} is not a declared lever")))?;
    let v = match rule {
        Rule::Switch => match text {
            "1" => Some(LeverValue::Switch(true)),
            "0" => Some(LeverValue::Switch(false)),
            _ => None,
        },
        Rule::Presence => (text == "1").then_some(LeverValue::Presence),
        Rule::Int => LeverValue::int(text),
        Rule::Float => Fixed::parse(text).map(LeverValue::Fixed),
        Rule::Text => Some(LeverValue::Text(String::from(text))),
        Rule::Path => Some(LeverValue::Path(String::from(text))),
    };
    v.ok_or_else(|| {
        refuse(format!(
            "env {name}={text} is not the canonical spelling of a {} lever",
            rule.as_str()
        ))
    })
}

/// 2026-10-22: Split a recipe into blueprint parts.
pub fn project(parts: &RecipeParts) -> Result<Projection, Error> {
    let mut flags = BTreeMap::new();
    let mut host = BTreeMap::new();
    for (k, v) in &parts.defaults {
        if RESOURCE_KEYS.contains(&k.as_str()) {
            host.insert(k.clone(), v.clone());
        } else {
            flags.insert(k.clone(), FlagValue::parse(v));
        }
    }
    let mut levers = BTreeMap::new();
    let mut passthrough = BTreeMap::new();
    for (k, v) in &parts.env {
        if k.starts_with(metrale_settings::spec::PREFIX) {
            levers.insert(k.clone(), lever(&parts.id, k, v)?);
        } else {
            passthrough.insert(k.clone(), v.clone());
        }
    }
    Ok(Projection {
        model: parts.model.clone(),
        world_size: parts.min_nodes,
        policy: ServePolicy {
            flags,
            chat_template: None,
        },
        settings: ServeSettings {
            registry_sha256: lever_table_digest(),
            levers,
            passthrough,
        },
        resources: Resources {
            recipe_id: parts.id.clone(),
            recipe_version: parts.version.clone(),
            runtime: parts.runtime.clone(),
            container: parts.container.clone(),
            flags: host,
        },
    })
}

impl Projection {
    /// 2026-10-22: The recipe parts these blueprint parts render to. Total.
    pub fn render(&self) -> RecipeParts {
        let mut defaults: BTreeMap<String, String> = self
            .policy
            .flags
            .iter()
            .map(|(k, v)| (k.clone(), v.render()))
            .collect();
        defaults.extend(self.resources.flags.clone());
        let mut env: BTreeMap<String, String> = self
            .settings
            .levers
            .iter()
            .map(|(k, v)| (k.clone(), v.render()))
            .collect();
        env.extend(self.settings.passthrough.clone());
        RecipeParts {
            id: self.resources.recipe_id.clone(),
            version: self.resources.recipe_version.clone(),
            model: self.model.clone(),
            runtime: self.resources.runtime.clone(),
            container: self.resources.container.clone(),
            min_nodes: self.world_size,
            defaults,
            env,
        }
    }
}

/// 2026-10-22: Render a blueprint (and its host resources) back to recipe parts. Total.
pub fn recipe_parts(bp: &Blueprint, resources: &Resources) -> RecipeParts {
    Projection {
        model: bp.checkpoint.repo.clone(),
        world_size: bp.hardware.world_size,
        policy: bp.policy.clone(),
        settings: bp.settings.clone(),
        resources: resources.clone(),
    }
    .render()
}
