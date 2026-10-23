// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The `kg-bundle/v1` JSONL form and its digest (queries part §4). Objects are written
//! with recursively sorted keys by a canonical writer, so the bytes do not depend on serde_json's
//! map feature (`preserve_order` may be unified in by another crate of the same build).
//!
//! Owner: metrale-kg-derive.
//! Invariants:
//! - Line order: nodes by id, edges by (from, rel name, to), texts by (node, field), then one
//!   `meta` line.
//! - The digest is `sha256(concat(sha256(line)))` over every node, edge and text line: a
//!   complete fingerprint of the derived graph; `meta` is excluded (it carries the digest).

use crate::bundle::{Bundle, Edge, Node, SCHEMA, Text};
use crate::hash::{Sha256, sha256};
use crate::prelude::*;
use serde_json::Value;

/// 2026-10-23: Append `v` to `out` as canonical JSON (sorted keys, no whitespace).
pub fn canon(v: &Value, out: &mut String) {
    match v {
        Value::Object(o) => {
            let mut keys: Vec<&String> = o.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_str(k, out);
                out.push(':');
                canon(&o[k.as_str()], out);
            }
            out.push('}');
        }
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                canon(x, out);
            }
            out.push(']');
        }
        other => out.push_str(&other.to_string()),
    }
}

fn push_str(s: &str, out: &mut String) {
    out.push_str(&Value::String(s.to_string()).to_string());
}

fn field(out: &mut String, k: &str, first: bool) {
    if !first {
        out.push(',');
    }
    push_str(k, out);
    out.push(':');
}

/// 2026-10-23: One node line.
pub fn node_line(n: &Node) -> String {
    let mut s = String::from("{\"t\":\"node\"");
    field(&mut s, "id", false);
    push_str(&n.id.to_string(), &mut s);
    field(&mut s, "label", false);
    push_str(n.id.label().name(), &mut s);
    field(&mut s, "src", false);
    push_str(&n.src, &mut s);
    field(&mut s, "sha", false);
    push_str(&n.sha.hex(), &mut s);
    field(&mut s, "p", false);
    let obj: serde_json::Map<String, Value> =
        n.p.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    canon(&Value::Object(obj), &mut s);
    s.push('}');
    s
}

/// 2026-10-23: One edge line.
pub fn edge_line(e: &Edge) -> String {
    let mut s = String::from("{\"t\":\"edge\"");
    field(&mut s, "from", false);
    push_str(&e.from.to_string(), &mut s);
    field(&mut s, "rel", false);
    push_str(e.rel.name(), &mut s);
    field(&mut s, "to", false);
    push_str(&e.to.to_string(), &mut s);
    field(&mut s, "w", false);
    s.push_str(&Value::from(e.w).to_string());
    s.push('}');
    s
}

/// 2026-10-23: One text line.
pub fn text_line(t: &Text) -> String {
    let mut s = String::from("{\"t\":\"text\"");
    field(&mut s, "node", false);
    push_str(&t.node.to_string(), &mut s);
    field(&mut s, "field", false);
    push_str(&t.field, &mut s);
    field(&mut s, "sha", false);
    push_str(&t.sha.hex(), &mut s);
    field(&mut s, "text", false);
    push_str(&t.text, &mut s);
    s.push('}');
    s
}

/// 2026-10-23: The node, edge and text lines, in bundle order.
pub fn record_lines(b: &Bundle) -> Vec<String> {
    let mut out = Vec::with_capacity(b.nodes.len() + b.edges.len() + b.texts.len());
    out.extend(b.nodes.iter().map(node_line));
    out.extend(b.edges.iter().map(edge_line));
    out.extend(b.texts.iter().map(text_line));
    out
}

/// 2026-10-23: The bundle digest over [`record_lines`].
pub fn digest(b: &Bundle) -> Sha256 {
    let mut cat = Vec::with_capacity(32 * (b.nodes.len() + b.edges.len() + b.texts.len()));
    for line in record_lines(b) {
        cat.extend_from_slice(&sha256(line.as_bytes()).0);
    }
    sha256(&cat)
}

/// 2026-10-23: The `meta` line.
pub fn meta_line(b: &Bundle) -> String {
    let sources: Vec<Value> = b
        .sources
        .iter()
        .map(|s| {
            let mut o = serde_json::Map::new();
            o.insert("path".into(), Value::String(s.path.clone()));
            o.insert("sha256".into(), Value::String(s.sha.hex()));
            o.insert(
                "git".into(),
                s.git.clone().map_or(Value::Null, Value::String),
            );
            Value::Object(o)
        })
        .collect();
    let errors: Vec<Value> = b
        .errors
        .iter()
        .map(|f| {
            let mut o = serde_json::Map::new();
            o.insert("code".into(), Value::String(f.code.to_string()));
            o.insert("at".into(), Value::String(f.at.clone()));
            o.insert("msg".into(), Value::String(f.msg.clone()));
            Value::Object(o)
        })
        .collect();
    let mut counts = serde_json::Map::new();
    counts.insert("nodes".into(), Value::from(b.nodes.len()));
    counts.insert("edges".into(), Value::from(b.edges.len()));
    counts.insert("texts".into(), Value::from(b.texts.len()));
    let mut o = serde_json::Map::new();
    o.insert("t".into(), Value::String("meta".into()));
    o.insert("schema".into(), Value::String(SCHEMA.into()));
    o.insert("digest".into(), Value::String(digest(b).hex()));
    o.insert("sources".into(), Value::Array(sources));
    o.insert("counts".into(), Value::Object(counts));
    o.insert("errors".into(), Value::Array(errors));
    let mut s = String::new();
    canon(&Value::Object(o), &mut s);
    s
}

/// 2026-10-23: The whole bundle as JSONL, newline-terminated.
pub fn render(b: &Bundle) -> String {
    let mut out = String::new();
    for line in record_lines(b) {
        out.push_str(&line);
        out.push('\n');
    }
    out.push_str(&meta_line(b));
    out.push('\n');
    out
}
