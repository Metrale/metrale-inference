// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Renaming a checkpoint onto its kept layers: tensor names, the config's layer
//! schedule keys, and every module name in the quantization metadata.
//!
//! Owner: metrale-ml-utils.
//! Invariants:
//! - A name under a kept layer is renumbered; a name under a dropped layer is dropped; every
//!   other name is unchanged. Nothing else in the config or metadata is edited here.
//! - Patterns (`re:` entries) are left as written; `quant_meta` checks what they resolve to
//!   and pins any module whose precision they would change.

use std::collections::BTreeMap;

use metrale_circuit::LayerSchedule;
use serde_json::Value;

use crate::error::{MlError, Result};

/// 2026-10-03: What happens to one name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fate {
    /// 2026-10-03: Not under any layer: unchanged.
    Keep,
    /// 2026-10-03: Under a kept layer: the new name.
    Rename(String),
    /// 2026-10-03: Under a dropped layer.
    Drop,
}

/// 2026-10-03: The renaming of one mock.
#[derive(Debug, Clone)]
pub struct Renamer<'a> {
    schedule: &'a LayerSchedule,
    renumber: &'a BTreeMap<usize, usize>,
}

impl<'a> Renamer<'a> {
    /// 2026-10-03: Rename by `renumber` (source layer -> mock layer).
    pub fn new(schedule: &'a LayerSchedule, renumber: &'a BTreeMap<usize, usize>) -> Self {
        Renamer { schedule, renumber }
    }

    /// 2026-10-03: The fate of a tensor or module name.
    pub fn fate(&self, name: &str) -> Fate {
        let Some(layer) = self.schedule.layer_of(name) else {
            return Fate::Keep;
        };
        match self.renumber.get(&layer) {
            None => Fate::Drop,
            Some(&new) => {
                let old_prefix = self.schedule.module_of(layer);
                let new_prefix = self.schedule.module_of(new);
                Fate::Rename(format!("{new_prefix}{}", &name[old_prefix.len()..]))
            }
        }
    }

    /// 2026-10-03: The mock name of `name`, or `None` when its layer is dropped.
    pub fn name(&self, name: &str) -> Option<String> {
        match self.fate(name) {
            Fate::Keep => Some(name.to_string()),
            Fate::Rename(n) => Some(n),
            Fate::Drop => None,
        }
    }

    /// 2026-10-03: Rewrite every module name in a quantization metadata value: object keys and
    /// strings in arrays are renamed or dropped; `re:` patterns and other strings stay.
    pub fn rewrite_metadata(&self, v: &mut Value) {
        match v {
            Value::Object(map) => {
                let old = std::mem::take(map);
                for (k, mut val) in old {
                    let key = match self.string_fate(&k) {
                        Fate::Keep => k,
                        Fate::Rename(n) => n,
                        Fate::Drop => continue,
                    };
                    self.rewrite_metadata(&mut val);
                    map.insert(key, val);
                }
            }
            Value::Array(items) => {
                let old = std::mem::take(items);
                for mut item in old {
                    if let Value::String(s) = &item {
                        match self.string_fate(s) {
                            Fate::Keep => {}
                            Fate::Rename(n) => item = Value::String(n),
                            Fate::Drop => continue,
                        }
                    } else {
                        self.rewrite_metadata(&mut item);
                    }
                    items.push(item);
                }
            }
            _ => {}
        }
    }

    fn string_fate(&self, s: &str) -> Fate {
        if s.starts_with("re:") {
            Fate::Keep
        } else {
            self.fate(s)
        }
    }

    /// 2026-10-03: The mock's config.json: the layer count and every present per-layer kind
    /// source reduced to the kept layers (a list keeps its kept entries, a pattern string its
    /// kept characters), in the model object and, if the top level restates them, there too.
    /// The quantization block is rewritten by [`Renamer::rewrite_metadata`].
    pub fn rewrite_config(&self, config: &Value) -> Result<Value> {
        let mut out = config.clone();
        let kept: Vec<usize> = self.renumber.keys().copied().collect();
        let n = self.schedule.layer_kinds.len();
        match &self.schedule.nest {
            Some(nest) => {
                self.reduce_layer_keys(&mut out, &kept, n, false)?;
                let inner = out.get_mut(nest.as_str()).ok_or_else(|| {
                    MlError::Checkpoint(format!("config.json has no `{nest}` object"))
                })?;
                self.reduce_layer_keys(inner, &kept, n, true)?;
            }
            None => self.reduce_layer_keys(&mut out, &kept, n, true)?,
        }
        if let Some(qc) = out.get_mut("quantization_config") {
            self.rewrite_metadata(qc);
        }
        Ok(out)
    }

    fn reduce_layer_keys(
        &self,
        obj: &mut Value,
        kept: &[usize],
        n: usize,
        required: bool,
    ) -> Result<()> {
        let map = obj
            .as_object_mut()
            .ok_or_else(|| MlError::Checkpoint("a config object is not an object".into()))?;
        let count_key = &self.schedule.count_key;
        match map.get(count_key).and_then(Value::as_u64) {
            Some(c) if c as usize == n => {
                map.insert(count_key.clone(), Value::from(kept.len() as u64));
            }
            Some(c) => {
                return Err(MlError::Checkpoint(format!(
                    "`{count_key}` = {c} disagrees with the {n} layers the schedule reads"
                )));
            }
            None if required => {
                return Err(MlError::Checkpoint(format!("`{count_key}` is missing")));
            }
            None => {}
        }
        for key in &self.schedule.source_keys {
            let reduced = match map.get(key) {
                Some(Value::Array(a)) if a.len() == n => {
                    Value::Array(kept.iter().map(|&i| a[i].clone()).collect())
                }
                Some(Value::String(s)) if s.chars().count() == n => {
                    let chars: Vec<char> = s.chars().collect();
                    Value::String(kept.iter().map(|&i| chars[i]).collect())
                }
                Some(other) => {
                    return Err(MlError::Checkpoint(format!(
                        "`{key}` ({}) does not list one entry per layer ({n})",
                        short(other)
                    )));
                }
                None if required => {
                    return Err(MlError::Checkpoint(format!("`{key}` is missing")));
                }
                None => continue,
            };
            map.insert(key.clone(), reduced);
        }
        Ok(())
    }
}

fn short(v: &Value) -> String {
    let s = v.to_string();
    if s.chars().count() > 60 {
        format!("{}...", s.chars().take(60).collect::<String>())
    } else {
        s
    }
}

#[cfg(test)]
#[path = "rename_tests.rs"]
mod rename_tests;
