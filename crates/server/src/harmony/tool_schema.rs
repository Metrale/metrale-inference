// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Explicit bounded tool-schema subset; no coercion, repair, or external refs.
use serde_json::Value;

#[derive(Clone)]
pub struct ToolSchema {
    pub name: String,
    parameters: Value,
}
impl ToolSchema {
    pub fn new(name: &str, parameters: Value) -> Result<Self, &'static str> {
        if name.is_empty()
            || name.len() > 64
            || !name
                .bytes()
                .enumerate()
                .all(|(i, b)| b.is_ascii_alphabetic() || b == b'_' || (i > 0 && b.is_ascii_digit()))
        {
            return Err("Harmony tool names must be ASCII identifiers of at most64 characters");
        }
        if parameters.to_string().len() > 65536 {
            return Err("tool schema too large");
        }
        schema(&parameters, 0)?;
        if parameters["type"] != "object" {
            return Err("tool parameters must be object schema");
        }
        Ok(Self {
            name: name.into(),
            parameters,
        })
    }
    pub fn recipient(&self) -> String {
        format!("functions.{}", self.name)
    }
    pub fn validate(&self, args: &Value) -> Result<(), &'static str> {
        instance(&self.parameters, args, 0)
    }
}

fn schema(s: &Value, depth: usize) -> Result<(), &'static str> {
    if depth > 8 {
        return Err("tool schema nesting exceeds8");
    }
    let obj = s.as_object().ok_or("tool schema must be object")?;
    for key in obj.keys() {
        if ![
            "type",
            "properties",
            "required",
            "additionalProperties",
            "items",
            "enum",
            "description",
            "title",
            "default",
        ]
        .contains(&key.as_str())
        {
            return Err("unsupported Harmony tool schema keyword");
        }
    }
    let kind = s["type"]
        .as_str()
        .ok_or("tool schema requires one explicit type")?;
    if ![
        "object", "array", "string", "integer", "number", "boolean", "null",
    ]
    .contains(&kind)
    {
        return Err("unsupported tool schema type");
    }
    if let Some(values) = s.get("enum") {
        if values
            .as_array()
            .is_none_or(|v| v.is_empty() || v.len() > 128)
        {
            return Err("invalid tool enum");
        }
    }
    if kind == "object" {
        let props = s
            .get("properties")
            .map(|p| p.as_object().ok_or("properties must be object"))
            .transpose()?;
        if let Some(props) = props {
            if props.len() > 64 {
                return Err("too many tool properties");
            }
            for child in props.values() {
                schema(child, depth + 1)?;
            }
        }
        if let Some(required) = s.get("required") {
            let required = required.as_array().ok_or("required must be array")?;
            let mut seen = std::collections::BTreeSet::new();
            for key in required {
                let key = key.as_str().ok_or("required entries must be strings")?;
                if !seen.insert(key) || props.is_none_or(|p| !p.contains_key(key)) {
                    return Err("required property is duplicate or undeclared");
                }
            }
        }
        if s.get("additionalProperties")
            .is_some_and(|a| !a.is_boolean())
        {
            return Err("additionalProperties must be boolean");
        }
    } else if s.get("properties").is_some()
        || s.get("required").is_some()
        || s.get("additionalProperties").is_some()
    {
        return Err("object keywords on nonobject schema");
    }
    if kind == "array" {
        schema(
            s.get("items").ok_or("array requires items schema")?,
            depth + 1,
        )?;
    } else if s.get("items").is_some() {
        return Err("items on nonarray schema");
    }
    Ok(())
}

fn instance(s: &Value, v: &Value, depth: usize) -> Result<(), &'static str> {
    if depth > 8 {
        return Err("tool argument nesting exceeds8");
    }
    let valid = match s["type"].as_str() {
        Some("object") => v.is_object(),
        Some("array") => v.is_array(),
        Some("string") => v.is_string(),
        Some("boolean") => v.is_boolean(),
        Some("null") => v.is_null(),
        Some("number") => v.is_number(),
        Some("integer") => {
            v.as_i64().is_some()
                || v.as_u64().is_some()
                || v.as_f64()
                    .is_some_and(|n| n.is_finite() && n.fract() == 0.0)
        }
        _ => false,
    };
    if !valid {
        return Err("tool argument type mismatch");
    }
    if s.get("enum").is_some_and(|choices| {
        !choices
            .as_array()
            .unwrap()
            .iter()
            .any(|choice| equivalent(choice, v))
    }) {
        return Err("tool argument is outside enum");
    }
    if let Some(obj) = v.as_object() {
        if let Some(required) = s["required"].as_array() {
            for name in required {
                if !obj.contains_key(name.as_str().unwrap()) {
                    return Err("missing required tool argument");
                }
            }
        }
        for (name, value) in obj {
            if let Some(child) = s["properties"].get(name) {
                instance(child, value, depth + 1)?;
            } else if s["additionalProperties"] == false {
                return Err("unexpected tool argument");
            }
        }
    }
    if let Some(values) = v.as_array() {
        for value in values {
            instance(&s["items"], value, depth + 1)?;
        }
    }
    Ok(())
}

// 2026-10-07: JSON Schema numeric equality must not coerce booleans or collapse
// neighboring large integers through an f64 conversion. Integral float spellings
// such as1.0 may equal1 only after an exact range-checked integer conversion.
fn equivalent(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) => {
            if !a.is_f64() && !b.is_f64() {
                return a == b;
            }
            if a.is_f64() && b.is_f64() {
                return a == b;
            }
            let (integer, float) = if a.is_f64() { (b, a) } else { (a, b) };
            let f = float.as_f64().unwrap();
            if !f.is_finite() || f.fract() != 0.0 {
                return false;
            }
            if let Some(i) = integer.as_i64() {
                f >= i64::MIN as f64 && f < (i64::MAX as f64) && f as i64 == i
            } else if let Some(u) = integer.as_u64() {
                f >= 0.0 && f < (u64::MAX as f64) && f as u64 == u
            } else {
                false
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| equivalent(a, b))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, a)| b.get(key).is_some_and(|b| equivalent(a, b)))
        }
        _ => a == b,
    }
}
