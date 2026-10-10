// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: Applying a [`ConfigMap`] to a parsed `config.json`.
//!
//! Owner: metrale-circuit.
//! Invariants: see [`crate::config_map`]; the first failure is returned.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use super::{ConfigMap, ConfigMapError, DimRule, KeyFull, KeyRule, LayerSource};
use crate::ir::{ArchShape, LayerKind};

/// 2026-09-30: What a config maps to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MappedConfig {
    /// 2026-09-30: The circuit arch.
    pub arch: String,
    /// 2026-09-30: The config's top-level `model_type`.
    pub model_type: String,
    /// 2026-09-30: Layer kinds and dims.
    pub shape: ArchShape,
    /// 2026-09-30: The math parameters, as JSON text, by name (nested ones dotted).
    pub params: BTreeMap<String, String>,
}

/// 2026-09-30: Map `config` (the whole `config.json`) under `map`.
pub fn map_config(map: &ConfigMap, config: &Value) -> Result<MappedConfig, ConfigMapError> {
    let f = &map.file;
    let root = config
        .as_object()
        .ok_or_else(|| ConfigMapError::Json("the top level is not an object".into()))?;
    let model_type = root
        .get("model_type")
        .and_then(Value::as_str)
        .ok_or_else(|| missing(map, "model_type"))?
        .to_string();
    if !f.model_types.contains(&model_type) {
        return Err(ConfigMapError::Refused {
            key: "model_type".into(),
            value: json(&Value::String(model_type)),
            why: format!("the `{}` config map serves {:?}", map.arch, f.model_types),
        });
    }
    let mut params = BTreeMap::new();
    let obj = match &f.nest {
        Some(nest) => {
            let inner = root
                .get(nest)
                .and_then(Value::as_object)
                .ok_or_else(|| missing(map, nest))?;
            let others: BTreeMap<&str, &Value> = root
                .iter()
                .filter(|(k, _)| k.as_str() != nest)
                .map(|(k, v)| (k.as_str(), v))
                .collect();
            classify(map, ("", ""), &others, &f.root, &mut params)?;
            inner
        }
        None => root,
    };
    let fields: BTreeMap<&str, &Value> = obj
        .iter()
        .filter(|(_, v)| !v.is_null())
        .map(|(k, v)| (k.as_str(), v))
        .collect();
    let variant = f.variant.get(&model_type).cloned().unwrap_or_default();
    let mut dims_rules = f.dims.clone();
    dims_rules.extend(variant.dims);
    let mut key_rules = f.keys.clone();
    key_rules.extend(variant.keys);

    let mut consumed: BTreeSet<&str> = BTreeSet::new();
    let layer_kinds = layers(map, &fields, &mut consumed)?;
    let dims = dims(map, &fields, &dims_rules, &mut consumed)?;
    let rest: BTreeMap<&str, &Value> = obj
        .iter()
        .filter(|(k, _)| !consumed.contains(k.as_str()))
        .map(|(k, v)| (k.as_str(), v))
        .collect();
    // 2026-09-30: Errors name the key's full path; params are named relative to the model's
    // object, so one name means one thing across nested and flat configs.
    let prefix = f.nest.as_ref().map_or(String::new(), |n| format!("{n}."));
    classify(map, (&prefix, ""), &rest, &key_rules, &mut params)?;
    Ok(MappedConfig {
        arch: map.arch.clone(),
        model_type,
        shape: ArchShape { layer_kinds, dims },
        params,
    })
}

fn missing(map: &ConfigMap, key: &str) -> ConfigMapError {
    ConfigMapError::MissingKey {
        arch: map.arch.clone(),
        key: key.to_string(),
    }
}

fn json(v: &Value) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| "?".into())
}

fn uint(key: &str, v: &Value) -> Result<u64, ConfigMapError> {
    v.as_u64().ok_or_else(|| {
        ConfigMapError::Json(format!("`{key}` = {} is not an unsigned integer", json(v)))
    })
}

fn layers<'a>(
    map: &ConfigMap,
    fields: &BTreeMap<&'a str, &'a Value>,
    consumed: &mut BTreeSet<&'a str>,
) -> Result<Vec<LayerKind>, ConfigMapError> {
    let l = &map.file.layers;
    let (count_key, count_v) = fields
        .get_key_value(l.count.as_str())
        .ok_or_else(|| missing(map, &l.count))?;
    consumed.insert(count_key);
    let count = uint(&l.count, count_v)? as usize;
    let kind = |name: &str, key: &str| {
        LayerKind::parse(name).ok_or_else(|| {
            ConfigMapError::Schema(format!("`{key}` maps to unknown layer kind `{name}`"))
        })
    };
    let present: Vec<(&LayerSource, &'a str, &'a Value)> = l
        .sources
        .iter()
        .filter_map(|s| {
            fields
                .get_key_value(s.key.as_str())
                .map(|(k, v)| (s, *k, *v))
        })
        .collect();
    for (_, k, _) in &present {
        consumed.insert(k);
    }
    let mut extra = 0;
    let mut kinds: Vec<LayerKind> = match (present.as_slice(), &l.uniform) {
        ([], Some(u)) => vec![kind(u, "uniform")?; count],
        ([], None) => {
            let keys: Vec<&str> = l.sources.iter().map(|s| s.key.as_str()).collect();
            return Err(missing(map, &keys.join(" | ")));
        }
        ([(src, key, v)], None) => {
            if let Some(t) = &src.trailing {
                let (tk, tv) = fields
                    .get_key_value(t.as_str())
                    .ok_or_else(|| missing(map, t))?;
                consumed.insert(tk);
                extra = uint(t, tv)? as usize;
            }
            from_source(src, key, v)?
                .into_iter()
                .map(|n| kind(&n, key))
                .collect::<Result<_, _>>()?
        }
        _ => {
            let keys: Vec<&str> = present.iter().map(|(_, k, _)| *k).collect();
            return Err(ConfigMapError::Refused {
                key: keys.join(", "),
                value: "(several)".into(),
                why: "the layer kinds must come from exactly one source".into(),
            });
        }
    };
    if kinds.len() != count + extra {
        return Err(ConfigMapError::Refused {
            key: l.count.clone(),
            value: count.to_string(),
            why: match extra {
                0 => format!("the layer-kind source lists {} layers", kinds.len()),
                _ => format!(
                    "the layer-kind source lists {} layers, not {count} plus {extra} trailing",
                    kinds.len()
                ),
            },
        });
    }
    kinds.truncate(count);
    Ok(kinds)
}

fn from_source(src: &LayerSource, key: &str, v: &Value) -> Result<Vec<String>, ConfigMapError> {
    let refuse = |value: String, why: String| ConfigMapError::Refused {
        key: key.to_string(),
        value,
        why,
    };
    if let Some(list) = v.as_array() {
        list.iter()
            .map(|e| {
                let s = match e {
                    Value::Number(n) => n.to_string(),
                    _ => e.as_str().unwrap_or_default().to_string(),
                };
                src.values.get(&s).cloned().ok_or_else(|| {
                    refuse(
                        json(e),
                        format!(
                            "a layer kind this arch maps: {:?}",
                            src.values.keys().collect::<Vec<_>>()
                        ),
                    )
                })
            })
            .collect()
    } else if let Some(s) = v.as_str() {
        s.chars()
            .map(|c| {
                src.chars.get(&c.to_string()).cloned().ok_or_else(|| {
                    refuse(
                        format!("'{c}'"),
                        format!(
                            "pattern letters this arch maps: {:?}",
                            src.chars.keys().collect::<Vec<_>>()
                        ),
                    )
                })
            })
            .collect()
    } else {
        Err(ConfigMapError::Json(format!(
            "`{key}` is neither a list nor a pattern string"
        )))
    }
}

fn dims<'a>(
    map: &ConfigMap,
    fields: &BTreeMap<&'a str, &'a Value>,
    rules: &BTreeMap<String, DimRule>,
    consumed: &mut BTreeSet<&'a str>,
) -> Result<BTreeMap<String, u64>, ConfigMapError> {
    let mut out = BTreeMap::new();
    let mut deferred = Vec::new();
    for (name, rule) in rules {
        let full = match rule {
            DimRule::Key(k) => super::DimFull {
                key: Some(k.clone()),
                bool: false,
                bool_present: false,
                constant: None,
                or_dim: None,
                or_div: None,
                default: None,
                allowed: None,
            },
            DimRule::Full(f) => f.clone(),
        };
        if let Some(c) = full.constant {
            out.insert(name.clone(), c);
            continue;
        }
        let key = full
            .key
            .as_deref()
            .ok_or_else(|| ConfigMapError::Schema(format!("dim `{name}` has no key or const")))?;
        // 2026-10-10: A presence switch does not consume its key: the value is classified by
        // another dim (`moe_latent_size`) or a key rule (`final_logit_softcapping`, a param).
        if full.bool_present {
            out.insert(name.clone(), u64::from(fields.contains_key(key)));
            continue;
        }
        if let Some((k, v)) = fields.get_key_value(key) {
            consumed.insert(k);
            let value = match (full.bool, v.as_bool()) {
                (true, Some(b)) => u64::from(b),
                _ => uint(key, v)?,
            };
            if let Some(allowed) = &full.allowed
                && !allowed.contains(&value)
            {
                return Err(ConfigMapError::Refused {
                    key: key.to_string(),
                    value: value.to_string(),
                    why: format!("the circuit models only {allowed:?}"),
                });
            }
            out.insert(name.clone(), value);
        } else if let Some(d) = full.default {
            out.insert(name.clone(), d);
        } else if full.or_dim.is_some() || full.or_div.is_some() {
            deferred.push((name.clone(), full));
        } else {
            return Err(missing(map, key));
        }
    }
    for (name, full) in deferred {
        let get = |d: &str| {
            out.get(d).copied().ok_or_else(|| {
                ConfigMapError::Schema(format!("dim `{name}` falls back to unmapped dim `{d}`"))
            })
        };
        let value = if let Some(d) = &full.or_dim {
            get(d)?
        } else if let Some([a, b]) = &full.or_div {
            let (a, b) = (get(a)?, get(b)?);
            if b == 0 || a % b != 0 {
                return Err(ConfigMapError::Refused {
                    key: full.key.clone().unwrap_or_default(),
                    value: "(absent)".into(),
                    why: format!("its fallback {a} / {b} is not exact"),
                });
            }
            a / b
        } else {
            unreachable!("deferred only with a fallback")
        };
        out.insert(name, value);
    }
    Ok(out)
}

fn to_json(v: &toml::Value) -> Value {
    serde_json::to_value(v).unwrap_or(Value::Null)
}

/// 2026-09-30: Classify every key of `obj` under `rules`, and apply each rule, present or not.
/// `prefix` is `(path prefix for errors, name prefix for params)`.
fn classify(
    map: &ConfigMap,
    prefix: (&str, &str),
    obj: &BTreeMap<&str, &Value>,
    rules: &BTreeMap<String, KeyRule>,
    params: &mut BTreeMap<String, String>,
) -> Result<(), ConfigMapError> {
    let (path_prefix, name_prefix) = prefix;
    for (k, v) in obj {
        if !v.is_null() && !rules.contains_key(*k) {
            return Err(ConfigMapError::UnmappedKey {
                arch: map.arch.clone(),
                key: format!("{path_prefix}{k}"),
            });
        }
    }
    for (key, rule) in rules {
        let path = format!("{path_prefix}{key}");
        let name = format!("{name_prefix}{key}");
        let v = obj.get(key.as_str()).copied().filter(|v| !v.is_null());
        match rule {
            KeyRule::Word(w) if w == "ignore" => {}
            KeyRule::Word(w) if w == "param" => {
                if let Some(v) = v {
                    params.insert(name, json(v));
                }
            }
            KeyRule::Word(w) => {
                return Err(ConfigMapError::Schema(format!(
                    "`{path}`: unknown rule `{w}` (ignore | param | a table)"
                )));
            }
            KeyRule::Full(f) => apply_full(map, (&path, &name), v, f, params)?,
        }
    }
    Ok(())
}

fn apply_full(
    map: &ConfigMap,
    (path, own): (&str, &str),
    v: Option<&Value>,
    f: &KeyFull,
    params: &mut BTreeMap<String, String>,
) -> Result<(), ConfigMapError> {
    if f.ignore.is_some() {
        return Ok(());
    }
    if let Some(why) = &f.refuse {
        return match v {
            Some(v) => Err(ConfigMapError::Refused {
                key: path.to_string(),
                value: json(v),
                why: why.clone(),
            }),
            None => Ok(()),
        };
    }
    let name = match &f.param {
        Some(toml::Value::String(s)) => Some(s.clone()),
        Some(toml::Value::Boolean(true)) => Some(own.to_string()),
        Some(other) => {
            return Err(ConfigMapError::Schema(format!(
                "`{path}`: `param` is {other}, want a name or true"
            )));
        }
        None => None,
    };
    if let Some(sub) = &f.object {
        let Some(v) = v else {
            return Ok(());
        };
        let o = v
            .as_object()
            .ok_or_else(|| ConfigMapError::Json(format!("`{path}` is not an object")))?;
        let fields: BTreeMap<&str, &Value> = o.iter().map(|(k, v)| (k.as_str(), v)).collect();
        let names = format!("{}.", name.unwrap_or_else(|| own.to_string()));
        return classify(map, (&format!("{path}."), &names), &fields, sub, params);
    }
    let value = match (v, &f.default) {
        (Some(v), _) => Some(v.clone()),
        (None, Some(d)) => Some(to_json(d)),
        (None, None) => None,
    };
    let Some(mut value) = value else {
        return Ok(());
    };
    if let Some(s) = value.as_str()
        && let Some(r) = f.values.get(s)
    {
        value = Value::String(r.clone());
    }
    if let Some(req) = &f.require {
        let allowed: Vec<Value> = match req {
            toml::Value::Array(a) => a.iter().map(to_json).collect(),
            one => vec![to_json(one)],
        };
        if !allowed.contains(&value) {
            return Err(ConfigMapError::Refused {
                key: path.to_string(),
                value: json(&value),
                why: f.why.clone().unwrap_or_else(|| {
                    format!("the circuit models only {}", json(&Value::Array(allowed)))
                }),
            });
        }
    }
    if let Some(n) = name {
        params.insert(n, json(&value));
    }
    Ok(())
}
