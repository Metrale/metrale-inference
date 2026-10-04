// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: A checkpoint's layer schedule as the embedded circuit and config map state it:
//! which config keys hold the layer count and the per-layer kinds, the module path of layer `i`,
//! the layout rule and the kinds. `metrale-ml-utils` reads it to drop whole layers from a
//! checkpoint without breaking the rule the circuit checks.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Every value comes from the embedded files (`<arch>.toml`, `<arch>.config.toml`) and the
//!   config: nothing is restated here.
//! - `source_keys` lists only the per-layer sources present in this config, in the map's order.

use serde_json::Value;

use super::{ARCHES, BLOCKS, CheckpointError, config_maps, map_checkpoint};
use crate::circuit_toml::{LayoutRule, layout_rule, parse_file};
use crate::ir::LayerKind;

/// 2026-10-03: The layer schedule of one checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerSchedule {
    /// 2026-10-03: The circuit arch.
    pub arch: String,
    /// 2026-10-03: The module path of layer `i`, with `{i}` in place of the index
    /// (`model.language_model.layers.{i}`).
    pub layer_module: String,
    /// 2026-10-03: The draft head's module (`mtp.layers.0`), when the circuit has one.
    pub draft_module: Option<String>,
    /// 2026-10-03: The layout rule the circuit checks the kinds against.
    pub layout: LayoutRule,
    /// 2026-10-03: The kind of every layer, in order.
    pub layer_kinds: Vec<LayerKind>,
    /// 2026-10-03: The object of config.json the model keys live in (`text_config`).
    pub nest: Option<String>,
    /// 2026-10-03: The key holding the layer count (`num_hidden_layers`).
    pub count_key: String,
    /// 2026-10-03: The per-layer kind sources present in the config: a list key
    /// (`layer_types`) or a pattern string key (`hybrid_override_pattern`).
    pub source_keys: Vec<String>,
}

impl LayerSchedule {
    /// 2026-10-03: The module path of layer `i`.
    pub fn module_of(&self, i: usize) -> String {
        self.layer_module.replace("{i}", &i.to_string())
    }

    /// 2026-10-03: The layer index `name` (a tensor or module path) lies under, if any:
    /// `<prefix><i>.<rest>` or exactly `<prefix><i>` for the `{i}` pattern.
    pub fn layer_of(&self, name: &str) -> Option<usize> {
        let (prefix, suffix) = self.layer_module.split_once("{i}")?;
        let rest = name.strip_prefix(prefix)?;
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        if digits == 0 {
            return None;
        }
        let tail = rest[digits..].strip_prefix(suffix)?;
        if !(tail.is_empty() || tail.starts_with('.')) {
            return None;
        }
        rest[..digits].parse().ok()
    }
}

/// 2026-10-03: The layer schedule of the checkpoint `config_json` describes.
pub fn layer_schedule(config_json: &str) -> Result<LayerSchedule, CheckpointError> {
    let mapped = map_checkpoint(config_json)?;
    let config: Value = serde_json::from_str(config_json).map_err(|e| CheckpointError::Json {
        file: "config.json",
        detail: e.to_string(),
    })?;
    let maps = config_maps()?;
    let (i, map) = maps
        .iter()
        .enumerate()
        .find(|(_, m)| m.arch == mapped.arch)
        .ok_or_else(|| CheckpointError::Quant(format!("no map for arch {}", mapped.arch)))?;
    let circuit = parse_file(ARCHES[i].circuit, &BLOCKS)?;
    let layout = layout_rule(&circuit.layout)?;
    let f = &map.file;
    let obj = match &f.nest {
        Some(n) => config.get(n).unwrap_or(&Value::Null),
        None => &config,
    };
    let source_keys = f
        .layers
        .sources
        .iter()
        .filter(|s| obj.get(&s.key).is_some_and(|v| !v.is_null()))
        .map(|s| s.key.clone())
        .collect();
    Ok(LayerSchedule {
        arch: mapped.arch,
        layer_module: circuit.layer_module,
        draft_module: circuit.draft_module,
        layout,
        layer_kinds: mapped.shape.layer_kinds,
        nest: f.nest.clone(),
        count_key: f.layers.count.clone(),
        source_keys,
    })
}

#[cfg(test)]
#[path = "schedule_tests.rs"]
mod schedule_tests;
