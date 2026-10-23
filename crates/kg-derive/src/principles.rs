// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `Principle` nodes from the CLAUDE.md tables (`| **ID** | rule | ... |`) merged with
//! the `## ID: title` sections of `programming-principles/decoupling.md` (rule, violation response,
//! enforcement). One node per id, defined at its table row; the decoupling text extends it.
//!
//! Owner: metrale-kg-derive.
//! Invariants: an id is upper-case letters only, 3 to 6 long (SSOT, PCND, ROTI, ...).

use crate::bundle::Builder;
use crate::id::{KgId, Label};
use crate::prelude::*;
use crate::text::table_cells;
use serde_json::Value;

fn principle_id(cell: &str) -> Option<String> {
    let id = cell.strip_prefix("**")?.strip_suffix("**")?;
    let ok = (3..=6).contains(&id.len()) && id.chars().all(|c| c.is_ascii_uppercase());
    ok.then(|| id.to_string())
}

struct Row {
    at: String,
    rule: String,
    extra: String,
}

fn sections(text: &str) -> BTreeMap<String, BTreeMap<&'static str, String>> {
    let mut out: BTreeMap<String, BTreeMap<&'static str, String>> = BTreeMap::new();
    let mut cur: Option<String> = None;
    let mut field: Option<&'static str> = None;
    for l in text.lines() {
        if let Some(h) = l.strip_prefix("## ") {
            cur = h
                .split_once(':')
                .map(|(id, _)| id.trim().to_string())
                .filter(|id| {
                    id.chars().all(|c| c.is_ascii_uppercase()) && (3..=6).contains(&id.len())
                });
            field = None;
            continue;
        }
        let Some(id) = &cur else { continue };
        for (tag, name) in [
            ("**Rule.**", "rule"),
            ("**Violation response.**", "violation_response"),
            ("**Enforcement.**", "gate"),
            ("**Gate.**", "gate"),
        ] {
            if let Some(rest) = l.strip_prefix(tag) {
                field = Some(name);
                out.entry(id.clone())
                    .or_default()
                    .insert(name, rest.trim().to_string());
            }
        }
        if l.starts_with("**")
            && !l.starts_with("**Rule.**")
            && !l.starts_with("**Violation response.**")
            && !l.starts_with("**Enforcement.**")
            && !l.starts_with("**Gate.**")
        {
            field = None;
        } else if let Some(f) = field
            && !l.starts_with("**")
            && !l.trim().is_empty()
        {
            let e = out.entry(id.clone()).or_default().entry(f).or_default();
            e.push(' ');
            e.push_str(l.trim());
        }
    }
    out
}

/// 2026-10-23: Derive the principles; returns the set of ids (for the text scanners).
pub fn derive(
    b: &mut Builder,
    tables: &[(&str, &str)],
    detail: &[(&str, &str)],
) -> BTreeSet<String> {
    let mut rows: BTreeMap<String, Row> = BTreeMap::new();
    for (path, text) in tables {
        for (i, l) in text.lines().enumerate() {
            let Some(cells) = table_cells(l) else {
                continue;
            };
            let Some(id) = cells.first().and_then(|c| principle_id(c)) else {
                continue;
            };
            let at = format!("{path}:{}", i + 1);
            let rule = cells.get(1).cloned().unwrap_or_default();
            let extra = cells.get(2..).map(|c| c.join(" | ")).unwrap_or_default();
            rows.entry(id).or_insert(Row { at, rule, extra });
        }
    }
    let mut more: BTreeMap<String, BTreeMap<&'static str, String>> = BTreeMap::new();
    for (_, text) in detail {
        for (id, f) in sections(text) {
            more.entry(id).or_default().extend(f);
        }
    }
    let mut ids = BTreeSet::new();
    for (id, row) in rows {
        let Ok(kid) = KgId::new(Label::Principle, id.clone()) else {
            continue;
        };
        let mut p = BTreeMap::new();
        p.insert("rule".into(), Value::String(row.rule.clone()));
        p.insert(
            "violation_response".into(),
            Value::String(row.extra.clone()),
        );
        let mut canon = format!("{id}\n{}\n{}", row.rule, row.extra);
        if let Some(f) = more.get(&id) {
            for (k, v) in f {
                p.insert(format!("detail_{k}"), Value::String(v.clone()));
                canon.push('\n');
                canon.push_str(v);
            }
        }
        b.define(kid.clone(), row.at, &canon, p);
        b.text(&kid, "rule", &row.rule);
        ids.insert(id);
    }
    ids
}
