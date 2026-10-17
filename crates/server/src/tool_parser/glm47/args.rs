// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Argument conversion for the GLM-4.7 call body
//! (`NAME<arg_key>K</arg_key><arg_value>V</arg_value>...`): the key/value
//! pairs as a JSON object of strings, then each string converted to the type
//! the tool's schema declares for its key.
//!
//! Owner: server (tool parsing).
//! Invariants:
//! - `convert_args` never fails and always yields an object: text outside a
//!   complete `<arg_key>`/`<arg_value>` pair is not an argument.
//! - `coerce_to_schema` changes only values whose key the tool's schema
//!   declares, and never turns a value into a non-finite number.

use serde_json::{Map, Value};

use super::super::ToolDefinition;

pub(crate) const ARG_KEY: &str = "<arg_key>";
const ARG_KEY_END: &str = "</arg_key>";
pub(super) const ARG_VALUE: &str = "<arg_value>";
pub(super) const ARG_VALUE_END: &str = "</arg_value>";

/// 2026-10-08: The complete pairs of a call body, in order. A pair is
/// `<arg_key>K</arg_key>`, optional whitespace, `<arg_value>V</arg_value>`;
/// the key ends at the first `</arg_key>` that such a value follows, so a
/// key can swallow a stray `</arg_key>`, and the value ends at the first
/// `</arg_value>`. An attempt that finds no complete pair moves on to the next
/// `<arg_key>`; a found pair resumes the scan after its `</arg_value>`.
pub(super) fn arg_pairs(raw: &str) -> Vec<(&str, &str)> {
    let mut pairs = Vec::new();
    let mut from = 0;
    while let Some(rel) = raw[from..].find(ARG_KEY) {
        let open = from + rel;
        match pair_at(raw, open + ARG_KEY.len()) {
            Some((key, value, end)) => {
                pairs.push((key, value));
                from = end;
            }
            // 2026-10-08: `<` is one byte, so `open + 1` is a char boundary.
            None => from = open + 1,
        }
    }
    pairs
}

/// 2026-10-08: The pair whose key starts at `key_start`: `(key, value, end)`,
/// `end` just past its `</arg_value>`.
fn pair_at(raw: &str, key_start: usize) -> Option<(&str, &str, usize)> {
    let mut search = key_start;
    while let Some(rel) = raw[search..].find(ARG_KEY_END) {
        let key_end = search + rel;
        let after_key = &raw[key_end + ARG_KEY_END.len()..];
        if let Some(value_tail) = after_key.trim_start().strip_prefix(ARG_VALUE) {
            let value_start = raw.len() - value_tail.len();
            // 2026-10-08: No close after this value start means none after any
            // later one either: the pair cannot complete.
            let value_len = value_tail.find(ARG_VALUE_END)?;
            let value_end = value_start + value_len;
            return Some((
                &raw[key_start..key_end],
                &raw[value_start..value_end],
                value_end + ARG_VALUE_END.len(),
            ));
        }
        search = key_end + 1;
    }
    None
}

/// 2026-10-08: The pairs as an object of strings: keys trimmed, values
/// verbatim. A repeated key keeps its first position and its last value.
pub(super) fn convert_args(raw: &str) -> Map<String, Value> {
    let mut args = Map::new();
    for (key, value) in arg_pairs(raw) {
        args.insert(key.trim().to_string(), Value::String(value.to_string()));
    }
    args
}

/// 2026-10-08: The `properties` object of the offered tool `name`; `None` when
/// the tool is not offered or declares none.
pub(super) fn tool_properties<'t>(
    tools: &'t [ToolDefinition],
    name: &str,
) -> Option<&'t Map<String, Value>> {
    tools
        .iter()
        .find(|t| t.function.name == name)?
        .function
        .parameters
        .as_ref()?
        .get("properties")?
        .as_object()
}

/// 2026-10-08: Convert each string value whose key the schema of tool `name`
/// declares to that key's type (`coerce_string`). Keys the schema does not
/// declare, and every key of a tool without a schema, keep their strings.
pub(super) fn coerce_to_schema(
    args: &mut Map<String, Value>,
    tools: &[ToolDefinition],
    name: &str,
) {
    let Some(properties) = tool_properties(tools, name) else {
        return;
    };
    for (key, value) in args.iter_mut() {
        let Some(schema) = properties.get(key).filter(|s| s.is_object()) else {
            continue;
        };
        if let Value::String(s) = value
            && let Some(converted) = coerce_string(s, &schema_types(schema))
        {
            *value = converted;
        }
    }
}

/// 2026-10-08: Every type a property schema admits: its `type` (a string or a
/// list), the types of its `enum` values, and those of its `anyOf`, `oneOf`
/// and `allOf` branches. A schema that names none admits `string`.
fn schema_types(schema: &Value) -> Vec<String> {
    let mut types = Vec::new();
    collect_types(schema, &mut types);
    if types.is_empty() {
        types.push("string".to_string());
    }
    types
}

fn collect_types(schema: &Value, types: &mut Vec<String>) {
    let Some(obj) = schema.as_object() else {
        return;
    };
    let mut add = |t: &str| {
        if !types.iter().any(|known| known == t) {
            types.push(t.to_string());
        }
    };
    match obj.get("type") {
        Some(Value::String(t)) => add(t),
        Some(Value::Array(list)) => list.iter().filter_map(Value::as_str).for_each(&mut add),
        _ => {}
    }
    if let Some(Value::Array(values)) = obj.get("enum") {
        for v in values {
            add(match v {
                Value::Null => "null",
                Value::Bool(_) => "boolean",
                Value::Number(n) if n.is_f64() => "number",
                Value::Number(_) => "integer",
                Value::String(_) => "string",
                Value::Array(_) => "array",
                Value::Object(_) => "object",
            });
        }
    }
    for branch in ["anyOf", "oneOf", "allOf"] {
        if let Some(Value::Array(choices)) = obj.get(branch) {
            for choice in choices {
                collect_types(choice, types);
            }
        }
    }
}

/// 2026-10-08: Spellings of JSON Schema types that tool schemas use in the
/// wild, mapped to the canonical name.
fn canonical_type(t: &str) -> String {
    let t = t.trim().to_lowercase();
    let canonical = match t.as_str() {
        "str" | "text" | "varchar" | "char" | "enum" => "string",
        "int" | "int32" | "int64" | "uint" | "uint32" | "uint64" | "long" | "short"
        | "unsigned" => "integer",
        "float" | "float32" | "float64" | "double" => "number",
        "bool" => "boolean",
        "dict" => "object",
        "arr" | "list" | "sequence" => "array",
        _ => return t,
    };
    canonical.to_string()
}

/// 2026-10-08: The value `raw` stands for under `types`, or `None` to keep the
/// string. The first type that converts wins, in the order null, integer,
/// number, boolean, object, array, string; a schema without `string` that no
/// type converts falls back to `raw` as JSON. A non-finite number is never
/// produced: such a `raw` stays a string.
fn coerce_string(raw: &str, types: &[String]) -> Option<Value> {
    let types: Vec<String> = types.iter().map(|t| canonical_type(t)).collect();
    let has = |t: &str| types.iter().any(|known| known == t);
    if has("null") && raw.to_lowercase() == "null" {
        return Some(Value::Null);
    }
    if has("integer")
        && let Some(n) = parse_integer(raw)
    {
        return Some(n);
    }
    if has("number")
        && let Ok(f) = raw.trim().parse::<f64>()
        && f.is_finite()
    {
        return Some(number_value(f));
    }
    if has("boolean") {
        match raw.trim().to_lowercase().as_str() {
            "true" | "1" => return Some(Value::Bool(true)),
            "false" | "0" => return Some(Value::Bool(false)),
            _ => {}
        }
    }
    if (has("object") || has("array"))
        && let Ok(parsed) = serde_json::from_str::<Value>(raw)
    {
        return Some(parsed);
    }
    if has("string") {
        return None;
    }
    serde_json::from_str::<Value>(raw).ok()
}

/// 2026-10-08: A decimal integer, with optional sign and surrounding
/// whitespace.
fn parse_integer(raw: &str) -> Option<Value> {
    let t = raw.trim();
    if let Ok(i) = t.parse::<i64>() {
        return Some(Value::from(i));
    }
    t.parse::<u64>().ok().map(Value::from)
}

/// 2026-10-08: A finite float as JSON: an integral value in `i64` range as an
/// integer (`3.0` → `3`), anything else as a float.
fn number_value(f: f64) -> Value {
    // 2026-10-08: 2^63 bounds the exact `i64` range of an integral f64.
    const I64_BOUND: f64 = 9_223_372_036_854_775_808.0;
    if f.fract() == 0.0 && f.abs() < I64_BOUND {
        Value::from(f as i64)
    } else {
        serde_json::Number::from_f64(f).map_or(Value::Null, Value::Number)
    }
}
