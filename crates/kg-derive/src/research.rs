// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: GRAPH-THEORY-MAP.md and MATHBOMB-CARRYOVER.md → `Idea` and `Paper` nodes
//! (ontology §2): ranked experiments (`idea:gtm-<n>-<slug>`, status `ranked`), the deferred and
//! rejected bullets (`idea:gtm-<slug>`), the mathbomb result rows (`idea:mathbomb-<n>`) and process
//! ideas (`idea:mathbomb-process-<n>`); papers from the "Sources" lines (verified) and from any
//! other arXiv id the rows cite (implicit).
//!
//! Owner: metrale-kg-derive.
//! Invariants: an idea's `RESTS_ON` papers are the arXiv ids in its own row text.

use crate::bundle::Builder;
use crate::id::{KgId, Label, Rel};
use crate::prelude::*;
use crate::text::{arxiv_ids, backtick_spans, design_refs, is_separator, slug, table_cells};
use serde_json::Value;

fn bold(s: &str) -> Option<&str> {
    let a = s.find("**")?;
    let rest = &s[a + 2..];
    let b = rest.find("**")?;
    Some(&rest[..b])
}

fn idea(b: &mut Builder, at: &str, key: String, status: &str, fields: &[(&str, &str)], raw: &str) {
    let Ok(id) = KgId::new(Label::Idea, key) else {
        return;
    };
    let mut p = BTreeMap::new();
    p.insert("status".into(), Value::String(status.to_string()));
    for (k, v) in fields {
        p.insert((*k).to_string(), Value::String((*v).to_string()));
    }
    b.define(id.clone(), at.to_string(), raw, p);
    b.text(&id, "row", raw);
    for a in arxiv_ids(raw) {
        if let Ok(pid) = KgId::new(Label::Paper, format!("arxiv:{a}")) {
            b.edge(&id, Rel::RestsOn, &pid, 1.0, at);
        }
    }
    let lands = fields
        .iter()
        .find(|(k, _)| *k == "lands_as")
        .map_or("", |(_, v)| *v);
    for span in backtick_spans(lands) {
        if span.starts_with("met ") {
            let cmd: Vec<&str> = span
                .split_whitespace()
                .take_while(|w| !w.starts_with('-'))
                .take(3)
                .collect();
            if let Ok(t) = KgId::new(Label::Tool, cmd.join(" ")) {
                b.edge(&id, Rel::LandsAs, &t, 1.0, at);
            }
        }
    }
    for d in design_refs(lands) {
        if let Ok(t) = KgId::new(Label::Design, d) {
            b.edge(&id, Rel::LandsAs, &t, 1.0, at);
        }
    }
}

/// 2026-10-23: Derive GRAPH-THEORY-MAP.md.
pub fn graph_theory_map(b: &mut Builder, path: &str, text: &str) {
    let mut section = "";
    for (i, l) in text.lines().enumerate() {
        let at = format!("{path}:{}", i + 1);
        if let Some(h) = l.strip_prefix("## ") {
            section = h;
            continue;
        }
        if section.starts_with("Ranked experiments") {
            let Some(c) = table_cells(l) else { continue };
            if c.len() < 6 || is_separator(&c) || c[0] == "#" {
                continue;
            }
            let title = bold(&c[1]).unwrap_or(&c[1]);
            let key = format!("gtm-{}-{}", c[0], slug(title, 3));
            let f = [
                ("title", title),
                ("theory", c[2].as_str()),
                ("lands_as", c[3].as_str()),
                ("gate", c[4].as_str()),
                ("effort", c[5].as_str()),
            ];
            idea(b, &at, key, "ranked", &f, l.trim());
        } else if section.starts_with("Deferred") || section.starts_with("Evaluated and rejected") {
            let Some(item) = l.strip_prefix("- ") else {
                continue;
            };
            let Some(title) = bold(item) else { continue };
            let status = if section.starts_with("Deferred") {
                "deferred"
            } else {
                "rejected"
            };
            let key = format!("gtm-{}", slug(title, 4));
            idea(b, &at, key, status, &[("title", title)], item);
        } else if section.starts_with("Sources") {
            let Some(item) = l.strip_prefix("- ") else {
                continue;
            };
            let list = item.split_once(":**").map_or(item, |(_, r)| r);
            for part in list.split('·') {
                let part = part.trim();
                for a in arxiv_ids(part) {
                    let title = part.replace(a.as_str(), "").trim().to_string();
                    let Ok(id) = KgId::new(Label::Paper, format!("arxiv:{a}")) else {
                        continue;
                    };
                    let mut p = BTreeMap::new();
                    p.insert("title".into(), Value::String(title.clone()));
                    p.insert("verified".into(), Value::Bool(true));
                    b.define(id.clone(), at.clone(), part, p);
                    b.text(&id, "title", &title);
                }
            }
        }
    }
}

/// 2026-10-23: Derive MATHBOMB-CARRYOVER.md.
pub fn mathbomb(b: &mut Builder, path: &str, text: &str) {
    let mut section = "";
    for (i, l) in text.lines().enumerate() {
        let at = format!("{path}:{}", i + 1);
        if let Some(h) = l.strip_prefix("## ") {
            section = h;
            continue;
        }
        if section.starts_with("Results") {
            let Some(c) = table_cells(l) else { continue };
            if c.len() < 5 || is_separator(&c) || c[0] == "#" {
                continue;
            }
            let f = [
                ("title", c[1].as_str()),
                ("lean", c[2].as_str()),
                ("relevance", c[3].as_str()),
                ("verdict", c[4].as_str()),
            ];
            idea(
                b,
                &at,
                format!("mathbomb-{}", c[0]),
                "evaluated",
                &f,
                l.trim(),
            );
        } else if section.starts_with("Process ideas") {
            let digits: String = l.chars().take_while(char::is_ascii_digit).collect();
            let Some(rest) = l[digits.len()..].strip_prefix(". ") else {
                continue;
            };
            if digits.is_empty() {
                continue;
            }
            let title = bold(rest).unwrap_or(rest).trim_end_matches('.');
            idea(
                b,
                &at,
                format!("mathbomb-process-{digits}"),
                "process",
                &[("title", title)],
                rest,
            );
        }
    }
}
