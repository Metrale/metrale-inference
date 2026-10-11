// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The declarative config mapping of one architecture
//! (`kernels/circuits/<arch>.config.toml`): how a checkpoint's `config.json` becomes the arch
//! shape a circuit instantiates under (layer kinds, dims) and the math parameters the kernels
//! read, with every key the config carries classified.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - PCND: every key of the mapped object is classified (a dim, a param, a required value, a
//!   nested object, or ignorable with a stated reason). An unclassified key is refused with
//!   its path; a math-changing key the circuit does not model is refused by its rule, never
//!   silently dropped.
//! - An absent key takes the default its rule states (the Hugging Face class default, cited
//!   in the TOML); a requirement applies to a present (or defaulted) value, and an absent key
//!   without a default is simply absent. A dim with no default is required.
//! - JSON `null` is absent.

use std::collections::BTreeMap;

use serde::Deserialize;

#[path = "config_map/apply.rs"]
mod apply;

pub use apply::{MappedConfig, map_config};

/// 2026-09-30: Why a config did not map.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ConfigMapError {
    /// 2026-09-30: The mapping TOML is malformed.
    #[error("config map: {0}")]
    Schema(String),
    /// 2026-09-30: `config.json` is not a JSON object, or a nested value is the wrong type.
    #[error("config.json: {0}")]
    Json(String),
    /// 2026-09-30: A key no rule classifies.
    #[error(
        "config.json key `{key}` is not mapped for arch `{arch}`: it may change the math, so it \
         is refused until the config map classifies it"
    )]
    UnmappedKey {
        /// 2026-09-30: The arch.
        arch: String,
        /// 2026-09-30: Dotted path of the key.
        key: String,
    },
    /// 2026-09-30: A value the circuit does not model.
    #[error("config.json `{key}` = {value}: {why}")]
    Refused {
        /// 2026-09-30: Dotted path of the key.
        key: String,
        /// 2026-09-30: The value, as JSON.
        value: String,
        /// 2026-09-30: Why it is refused.
        why: String,
    },
    /// 2026-09-30: A required key is absent and its rule states no default.
    #[error("config.json lacks `{key}`, which the `{arch}` mapping requires")]
    MissingKey {
        /// 2026-09-30: The arch.
        arch: String,
        /// 2026-09-30: Dotted path of the key.
        key: String,
    },
}

/// 2026-09-30: A parsed config map.
#[derive(Debug, Clone)]
pub struct ConfigMap {
    /// 2026-09-30: The circuit arch it produces a shape for (`kernels/circuits/<arch>.toml`).
    pub arch: String,
    pub(crate) file: MapFile,
}

impl ConfigMap {
    /// 2026-09-30: Parse a mapping TOML.
    pub fn parse(text: &str) -> Result<Self, ConfigMapError> {
        let file: MapFile =
            toml::from_str(text).map_err(|e| ConfigMapError::Schema(e.to_string()))?;
        if file.schema != 1 {
            return Err(ConfigMapError::Schema(format!(
                "schema {} (this build reads 1)",
                file.schema
            )));
        }
        if file.model_types.is_empty() {
            return Err(ConfigMapError::Schema("`model_types` is empty".into()));
        }
        for v in file.variant.keys() {
            if !file.model_types.contains(v) {
                return Err(ConfigMapError::Schema(format!(
                    "variant `{v}` is not one of `model_types`"
                )));
            }
        }
        if file.nest.is_some() == file.root.is_empty() {
            return Err(ConfigMapError::Schema(
                "`[root]` classifies the top-level keys of a nested config; it is required \
                 exactly when `nest` is set"
                    .into(),
            ));
        }
        Ok(ConfigMap {
            arch: file.arch.clone(),
            file,
        })
    }

    /// 2026-09-30: The top-level `model_type` values this map serves.
    pub fn model_types(&self) -> &[String] {
        &self.file.model_types
    }

    /// 2026-09-30: Whether this map's circuit serves an engine-configured `model_type`: one of
    /// `model_types`, or a name the engine's parser gives such a checkpoint.
    pub fn serves_engine_model_type(&self, model_type: &str) -> bool {
        self.file
            .model_types
            .iter()
            .chain(&self.file.engine_model_types)
            .any(|t| t == model_type)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MapFile {
    pub schema: u32,
    pub arch: String,
    pub model_types: Vec<String>,
    /// 2026-09-30: The `ModelConfig::model_type` names the engine's config parser gives these
    /// checkpoints where it renames them (`qwen3_6_moe`, `holo3_1_moe` for a `qwen3_5_moe`
    /// config); a model configured by the engine finds its circuit by these or `model_types`.
    #[serde(default)]
    pub engine_model_types: Vec<String>,
    /// 2026-09-30: The object the model's keys live in (`text_config`); the top level's
    /// other keys are classified by `root`.
    pub nest: Option<String>,
    #[serde(default)]
    pub root: BTreeMap<String, KeyRule>,
    pub layers: LayersFile,
    pub dims: BTreeMap<String, DimRule>,
    #[serde(default)]
    pub keys: BTreeMap<String, KeyRule>,
    #[serde(default)]
    pub variant: BTreeMap<String, VariantFile>,
}

/// 2026-09-30: Per-`model_type` additions: its dims and key rules replace the base ones of
/// the same name.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct VariantFile {
    #[serde(default)]
    pub dims: BTreeMap<String, DimRule>,
    #[serde(default)]
    pub keys: BTreeMap<String, KeyRule>,
}

/// 2026-09-30: Where the layer kinds come from: exactly one listed source is present.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LayersFile {
    /// 2026-09-30: The key holding the layer count.
    pub count: String,
    /// 2026-09-30: Every layer is this kind (no per-layer source).
    pub uniform: Option<String>,
    #[serde(default)]
    pub sources: Vec<LayerSource>,
}

/// 2026-09-30: A per-layer kind source: a list of names, or a pattern string of letters.
/// 2026-10-10: A list entry may be an integer, matched by its decimal spelling
/// (`compress_ratios`: `0`, `4`, `128`).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LayerSource {
    pub key: String,
    /// 2026-09-30: List entries to layer kinds (2026-10-10: an integer entry by its decimal
    /// text, `"1" = "full_attention"`).
    #[serde(default)]
    pub values: BTreeMap<String, String>,
    /// 2026-09-30: Pattern letters to layer kinds.
    #[serde(default)]
    pub chars: BTreeMap<String, String>,
    /// 2026-10-10: A key whose value counts the entries the source lists past the text layers
    /// (DeepSeek-V4's `compress_ratios` also lists its `num_nextn_predict_layers` MTP layers).
    /// Those entries must still map, and are then dropped; absent, the source lists exactly
    /// the text layers.
    pub trailing: Option<String>,
}

/// 2026-09-30: How one dim is read.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum DimRule {
    /// 2026-09-30: A required integer key.
    Key(String),
    /// 2026-09-30: The general form.
    Full(DimFull),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DimFull {
    pub key: Option<String>,
    /// 2026-09-30: A boolean key read as 0 / 1.
    #[serde(default)]
    pub bool: bool,
    /// 2026-09-30: 1 when the key is present (not null), else 0: a switch on an optional
    /// feature whose size is another dim (`moe_latent_size`). 2026-10-10: It does not consume
    /// the key, so the value is still classified (a dim, or a key rule such as a param).
    #[serde(default)]
    pub bool_present: bool,
    /// 2026-09-30: A fixed value.
    #[serde(rename = "const")]
    pub constant: Option<u64>,
    /// 2026-09-30: When the key is absent: this mapped dim's value.
    pub or_dim: Option<String>,
    /// 2026-09-30: When the key is absent: `a / b` of two mapped dims, exactly.
    pub or_div: Option<[String; 2]>,
    /// 2026-09-30: When the key is absent: this value (the HF default).
    pub default: Option<u64>,
    /// 2026-09-30: The values the circuit models; any other is refused.
    pub allowed: Option<Vec<u64>>,
}

/// 2026-09-30: How one config key is classified.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum KeyRule {
    /// 2026-09-30: `"ignore"` (not math: ids, dropout, runtime hints) or `"param"` (carried
    /// into the params as JSON text).
    Word(String),
    /// 2026-09-30: The general form.
    Full(Box<KeyFull>),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct KeyFull {
    /// 2026-09-30: Not math, with the reason.
    pub ignore: Option<String>,
    /// 2026-09-30: Refused whenever present (not null), with the reason: math the circuit
    /// does not model.
    pub refuse: Option<String>,
    /// 2026-09-30: Carry the value into the params under this name (`true`: the key's own).
    pub param: Option<toml::Value>,
    /// 2026-09-30: The value must be one of these (after `values` renaming).
    pub require: Option<toml::Value>,
    /// 2026-09-30: Why a value outside `require` is refused.
    pub why: Option<String>,
    /// 2026-09-30: The value when absent (the HF default).
    pub default: Option<toml::Value>,

    /// 2026-09-30: Rename values (`swish` -> `silu`) before `require` and `param`.
    #[serde(default)]
    pub values: BTreeMap<String, String>,
    /// 2026-09-30: A nested object: rules for its keys, prefixed `<param>.` in the params.
    pub object: Option<BTreeMap<String, KeyRule>>,
}

#[cfg(test)]
#[path = "config_map_tests.rs"]
mod config_map_tests;
