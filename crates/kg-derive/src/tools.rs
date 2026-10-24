// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The MCP tool surface (design §3): exactly five tools, their input schemas as
//! shipped, and the server instructions. This is the one definition `tools/list` serves and the P0
//! token gate measures.
//!
//! Owner: metrale-kg-derive.
//! Invariants:
//! - Five tools; every description is at most 140 characters (a unit test holds both).
//! - Declared defaults are written in the schema (PCND); the tool layer applies exactly these.

use crate::prelude::*;
use serde_json::Value;

/// 2026-10-23: The `kg_status` section names, in schema order.
pub const SECTIONS: &[&str] = &[
    "progress",
    "blocked",
    "tainted",
    "claims",
    "owed_lkb",
    "dangling",
    "calibration",
    "freshness",
    "schema",
    "files",
];

/// 2026-10-23: The sections of a default `kg_status` (none named): compact, every list capped at
/// [`LIST_CAP`] with its total beside it. `schema` and `files` come only when named.
pub const DEFAULT_SECTIONS: &[&str] = &[
    "progress",
    "blocked",
    "tainted",
    "claims",
    "owed_lkb",
    "dangling",
    "calibration",
    "freshness",
];

/// 2026-10-23: The most items of any list in a default `kg_status`.
pub const LIST_CAP: usize = 3;

/// 2026-10-23: The `kg_record` kinds.
pub const RECORD_KINDS: &[&str] = &["ledger", "friction", "decision", "run"];

/// 2026-10-23: Declared defaults of the tool arguments (the schema says the same).
pub const DEFAULT_QUERY_LIMIT: u32 = 200;
/// 2026-10-23: Declared default `k` of `kg_search`.
pub const DEFAULT_SEARCH_K: u32 = 10;
/// 2026-10-23: Declared default `hops` of `kg_search`.
pub const DEFAULT_SEARCH_HOPS: u32 = 1;
/// 2026-10-23: Declared default `max_depth` of `kg_impact`.
pub const DEFAULT_MAX_DEPTH: u32 = 100;

/// 2026-10-23: The server instructions sent at `initialize`.
pub const INSTRUCTIONS: &str = "Derived index of campaign ledgers, frictions, decisions, designs, \
memory rules and the LKB; files stay the source of truth. Start with kg_status. kg_search before a \
new hypothesis, kg_impact before a retraction. In kg_query, n.id is the typed id; filter on n.name.";

fn tool(name: &str, description: &str, schema: &str) -> Value {
    let mut o = serde_json::Map::new();
    o.insert("name".into(), Value::String(name.into()));
    o.insert("description".into(), Value::String(description.into()));
    let s: Value = serde_json::from_str(schema).unwrap_or(Value::Null);
    o.insert("inputSchema".into(), s);
    Value::Object(o)
}

/// 2026-10-23: The `tools/list` array.
pub fn tools() -> Vec<Value> {
    let sections: Vec<String> = SECTIONS.iter().map(|s| format!("\"{s}\"")).collect();
    let kinds: Vec<String> = RECORD_KINDS.iter().map(|s| format!("\"{s}\"")).collect();
    vec![
        tool(
            "kg_query",
            "Read-only Cypher (labels, relations: kg_status schema).",
            r#"{"type":"object","required":["cypher"],"properties":{"cypher":{"type":"string"},"params":{"type":"object"},"limit":{"type":"integer"}}}"#,
        ),
        tool(
            "kg_search",
            "Have we tried this? Vector+graph recall; reports mode hybrid|graph_only.",
            r#"{"type":"object","required":["text"],"properties":{"text":{"type":"string"},"kinds":{"type":"array","items":{"type":"string"}},"k":{"type":"integer"},"hops":{"type":"integer"}}}"#,
        ),
        tool(
            "kg_impact",
            "Dependency/taint closure of an id (== ledger.py tainted/deps).",
            r#"{"type":"object","required":["id","direction"],"properties":{"id":{"type":"string"},"direction":{"enum":["downstream","upstream"]},"relations":{"type":"array","items":{"type":"string"}},"max_depth":{"type":"integer"}}}"#,
        ),
        tool(
            "kg_record",
            "Append a record to its SSOT file through its validator; fails closed. dry_run previews would_taint.",
            &format!(
                r#"{{"type":"object","required":["kind","campaign","fields"],"properties":{{"kind":{{"enum":[{}]}},"campaign":{{"type":"string"}},"fields":{{"type":"object"}},"dry_run":{{"type":"boolean"}},"acknowledge_taint":{{"type":"integer"}}}}}}"#,
                kinds.join(",")
            ),
        ),
        tool(
            "kg_status",
            "Campaign progress, blockers, taint, claims, owed LKB rows. Default: compact (lists capped); name a section for all of it.",
            &format!(
                r#"{{"type":"object","required":["campaign"],"properties":{{"campaign":{{"type":"string"}},"sections":{{"type":"array","items":{{"enum":[{}]}}}},"instance":{{"type":"string"}}}}}}"#,
                sections.join(",")
            ),
        ),
    ]
}

/// 2026-10-23: The `tools/list` result body, canonical JSON, as the token gate counts it.
pub fn tools_list_json() -> String {
    let mut o = serde_json::Map::new();
    o.insert("tools".into(), Value::Array(tools()));
    let mut s = String::new();
    crate::jsonl::canon(&Value::Object(o), &mut s);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exactly_five_tools_with_short_descriptions_and_valid_schemas() {
        let t = tools();
        assert_eq!(t.len(), 5);
        for x in &t {
            let d = x["description"].as_str().unwrap();
            assert!(d.chars().count() <= 140, "{d}");
            assert_eq!(x["inputSchema"]["type"], "object", "{}", x["name"]);
        }
        assert!(INSTRUCTIONS.split_whitespace().count() <= 120);
    }
}
