// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: TOOLING.md → `Friction` nodes (ontology §2): one per table row
//! `phase | friction | cost_min | proposed_change | done`, id
//! `friction:<phase>/<sha256(friction)[..8 hex]>`. A line that is neither blank, a heading, nor a
//! table row is a `KgBadRow` finding (design §2.1: reported, not silently skipped).
//!
//! Owner: metrale-latlas-pack (the metrale pack; was metrale-kg-derive).
//! Invariants: a wording edit of a friction is a new node; the old one vanishes on re-derive.

use crate::prelude::*;
use crate::vocab::*;
use latlas_core::bundle::Builder;
use latlas_core::hash::record_sha;
use latlas_core::text::{
    backtick_spans, commit_shas, first_number, is_separator, principle_ids, table_cells,
};
use latlas_error::{ErrorCode, error};
use serde_json::Value;

/// 2026-10-23: The columns a friction row has.
pub const COLUMNS: [&str; 5] = ["phase", "friction", "cost_min", "proposed_change", "done"];

/// 2026-10-23: `yes` / `no` / `partly` / `other` from a `done` cell.
pub fn done_kind(done: &str) -> &'static str {
    let d = done.trim().to_ascii_lowercase();
    if d.starts_with("yes") || d.starts_with("done") {
        "yes"
    } else if d.starts_with("no") {
        "no"
    } else if d.starts_with("partly") {
        "partly"
    } else {
        "other"
    }
}

fn tool_of(span: &str) -> Option<String> {
    let s = span.trim();
    let first = s.split_whitespace().next()?;
    let pathish = first.contains('/')
        && [".sh", ".py", ".rs", ".md", ".toml"]
            .iter()
            .any(|e| first.ends_with(e));
    if s.starts_with("met ") {
        let cmd: Vec<&str> = s
            .split_whitespace()
            .take_while(|w| !w.starts_with('-') && !w.starts_with('<'))
            .take(3)
            .collect();
        return Some(cmd.join(" "));
    }
    pathish.then(|| first.to_string())
}

/// 2026-10-23: Derive the frictions of one TOOLING.md.
pub fn derive(
    b: &mut Builder,
    campaign: &str,
    path: &str,
    text: &str,
    principles: &BTreeSet<String>,
) {
    for (i, line) in text.lines().enumerate() {
        let at = format!("{path}:{}", i + 1);
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let Some(cells) = table_cells(t) else {
            b.error(
                &error!(
                    ErrorCode::LatlasBadRow,
                    at.clone(),
                    "not a `| phase | friction | cost_min | proposed_change | done |` row"
                ),
                &at,
            );
            continue;
        };
        if is_separator(&cells) || cells.first().is_some_and(|c| c == "phase") {
            continue;
        }
        if cells.len() != COLUMNS.len() {
            let e = error!(
                ErrorCode::LatlasBadRow,
                at.clone(),
                format!("{} cells, expected 5", cells.len())
            );
            b.error(&e, &at);
            continue;
        }
        let (phase, friction, proposed, done) = (&cells[0], &cells[1], &cells[3], &cells[4]);
        let short: String = record_sha(friction).hex().chars().take(8).collect();
        let Ok(id) = NodeId::new(labels::Friction, format!("{phase}/{short}")) else {
            b.error(
                &error!(ErrorCode::LatlasBadRow, at.clone(), "empty phase"),
                &at,
            );
            continue;
        };
        let mut p = BTreeMap::new();
        for (k, v) in COLUMNS.iter().zip(&cells) {
            p.insert((*k).to_string(), Value::String(v.clone()));
        }
        p.insert("campaign".into(), Value::String(campaign.to_string()));
        p.insert(
            "cost_min_n".into(),
            first_number(&cells[2]).map_or(Value::Null, Value::from),
        );
        p.insert(
            "done_kind".into(),
            Value::String(done_kind(done).to_string()),
        );
        b.define(id.clone(), at.clone(), t, p);
        for span in backtick_spans(proposed) {
            if let Some(tool) = tool_of(&span).and_then(|k| NodeId::new(labels::Tool, k).ok()) {
                b.edge(&id, rels::Proposes, &tool, 1.0, &at);
            }
        }
        for c in commit_shas(done) {
            if let Ok(cid) = NodeId::new(labels::Commit, c) {
                b.edge(&id, rels::ResolvedBy, &cid, 1.0, &at);
            }
        }
        for pid in principle_ids(&format!("{friction} {proposed}"), principles) {
            if let Ok(to) = NodeId::new(labels::Principle, pid) {
                b.edge(&id, rels::Touches, &to, 1.0, &at);
            }
        }
        b.text(&id, "friction", friction);
        b.text(&id, "proposed_change", proposed);
    }
}
