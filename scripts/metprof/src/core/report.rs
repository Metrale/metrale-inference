// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: render a command result as JSON or as plain text (pure).
//!
//! Text form: scalar fields as `key: value` lines, nested objects indented
//! under their key, then every list of row objects as a table. The values are
//! the same in both forms; only the layout differs.

use serde::Serialize;
use serde_json::ser::{PrettyFormatter, Serializer};
use serde_json::{Map, Value};

use crate::core::error::Result;
use crate::fail;

fn scalar(v: &Value, fixed: bool) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Value::Number(n) if n.is_f64() && fixed => format!("{:.3}", n.as_f64().unwrap_or(f64::NAN)),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn is_scalar(v: &Value) -> bool {
    !matches!(v, Value::Array(_) | Value::Object(_))
}

fn cell(v: &Value) -> String {
    match v {
        Value::Array(items) if items.iter().all(is_scalar) => {
            let inner: Vec<String> = items.iter().map(|x| scalar(x, false)).collect();
            format!("({})", inner.join(","))
        }
        other => scalar(other, true),
    }
}

fn is_table(v: &Value) -> bool {
    matches!(v, Value::Array(items) if !items.is_empty() && items.iter().all(Value::is_object))
}

fn table(rows: &[Value]) -> Vec<String> {
    let mut cols: Vec<&str> = Vec::new();
    for r in rows.iter().filter_map(Value::as_object) {
        for k in r.keys() {
            if !cols.contains(&k.as_str()) {
                cols.push(k);
            }
        }
    }
    let cells: Vec<Vec<String>> = rows
        .iter()
        .filter_map(Value::as_object)
        .map(|r| {
            cols.iter()
                .map(|c| r.get(*c).map(cell).unwrap_or_default())
                .collect()
        })
        .collect();
    let width: Vec<usize> = cols
        .iter()
        .enumerate()
        .map(|(i, c)| {
            cells
                .iter()
                .map(|row| row[i].chars().count())
                .fold(c.chars().count(), usize::max)
        })
        .collect();
    let line = |vals: Vec<&str>| -> String {
        let parts: Vec<String> = vals
            .iter()
            .zip(&width)
            .map(|(v, w)| format!("{v:>w$}"))
            .collect();
        format!("  {}", parts.join("  "))
    };
    let mut out = vec![line(cols.clone())];
    out.extend(
        cells
            .iter()
            .map(|row| line(row.iter().map(String::as_str).collect())),
    );
    out
}

pub fn render_text(result: &Map<String, Value>, prefix: &str) -> String {
    let mut lines: Vec<String> = Vec::new();
    for (k, v) in result {
        match v {
            Value::Object(m) => {
                lines.push(format!("{prefix}{k}:"));
                lines.push(render_text(m, &format!("{prefix}  ")));
            }
            v if !is_table(v) => lines.push(format!("{prefix}{k}: {}", cell(v))),
            _ => {}
        }
    }
    for (k, v) in result {
        if let Value::Array(rows) = v
            && is_table(v)
        {
            lines.push(format!("{prefix}{k} ({} rows):", rows.len()));
            lines.extend(table(rows).into_iter().map(|l| format!("{prefix}{l}")));
        }
    }
    lines.join("\n")
}

pub fn render_json(result: &Map<String, Value>) -> Result<String> {
    let mut buf = Vec::new();
    let mut ser = Serializer::with_formatter(&mut buf, PrettyFormatter::with_indent(b" "));
    if let Err(e) = result.serialize(&mut ser) {
        fail!("cannot render the result as JSON: {e}");
    }
    match String::from_utf8(buf) {
        Ok(s) => Ok(s),
        Err(e) => fail!("cannot render the result as JSON: {e}"),
    }
}

pub fn render(result: &Map<String, Value>, as_json: bool) -> Result<String> {
    if as_json {
        render_json(result)
    } else {
        Ok(render_text(result, ""))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn obj(v: Value) -> Map<String, Value> {
        v.as_object().cloned().unwrap_or_default()
    }

    #[test]
    fn text_puts_scalars_first_then_tables_with_right_aligned_columns() {
        let r = obj(json!({
            "rows": [{"kernel": "a", "ms": 1.5}, {"kernel": "bbb", "ms": 10.25, "grid": [4, 1, 1]}],
            "n": 3, "share": 0.5, "none": null, "ok": true, "empty": [], "sub": {"x": 1}
        }));
        let want = "n: 3\nshare: 0.500\nnone: None\nok: True\nempty: ()\nsub:\n  x: 1\nrows (2 rows):\n  \
                    kernel      ms     grid\n       a   1.500         \n     bbb  10.250  (4,1,1)";
        assert_eq!(render_text(&r, ""), want);
    }

    #[test]
    fn json_keeps_key_order_with_one_space_indent() {
        let r = obj(json!({"z": 1, "a": [1.5]}));
        assert_eq!(
            render_json(&r).ok().as_deref(),
            Some("{\n \"z\": 1,\n \"a\": [\n  1.5\n ]\n}")
        );
    }
}
