// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: CHARTER.md and DESIGN-*.md → `Decision` and `Design` nodes (ontology §2). Decisions
//! are the bullets under an `Owner decisions (<date>)` line or heading and under
//! `## Decisions <date>`, and each `## Re-plan <date> (...): <title>` section; designs are the
//! headings of each DESIGN file, keyed by their section number (`DESIGN-x.md#2.1`) or slug.
//!
//! Owner: metrale-kg-derive.
//! Invariants:
//! - Ids are `decision:<date>/<slug of the first words>`; a repeated slug in one source takes a
//!   `-2`, `-3` suffix in file order, so a re-derive is stable while the file is unchanged.
//! - A decision list under a heading that says `needed` is `open`; every other one `resolved`.

use crate::bundle::Builder;
use crate::id::{KgId, Label, Rel};
use crate::prelude::*;
use crate::text::slug;
use serde_json::Value;

fn date_in(s: &str) -> Option<String> {
    let b = s.as_bytes();
    (0..b.len().saturating_sub(9)).find_map(|i| {
        let w = &s.get(i..i + 10)?;
        let ok = w.bytes().enumerate().all(|(j, c)| match j {
            4 | 7 => c == b'-',
            _ => c.is_ascii_digit(),
        });
        ok.then(|| (*w).to_string())
    })
}

fn unique(seen: &mut BTreeMap<String, u32>, key: String) -> String {
    let n = seen.entry(key.clone()).or_insert(0);
    *n += 1;
    if *n == 1 { key } else { format!("{key}-{n}") }
}

fn item_text(line: &str) -> Option<&str> {
    let t = line;
    if let Some(r) = t.strip_prefix("- ") {
        return Some(r);
    }
    let digits = t.chars().take_while(char::is_ascii_digit).count();
    (digits > 0)
        .then(|| t[digits..].strip_prefix(". "))
        .flatten()
}

struct Ctx<'a> {
    path: &'a str,
    seen: BTreeMap<String, u32>,
    ledger: &'a BTreeSet<String>,
}

impl Ctx<'_> {
    fn decision(
        &mut self,
        b: &mut Builder,
        line: usize,
        date: &str,
        text: &str,
        status: &str,
        words: usize,
    ) -> Option<KgId> {
        let s = slug(text, words);
        if s.is_empty() {
            return None;
        }
        let key = unique(&mut self.seen, format!("{date}/{s}"));
        let id = KgId::new(Label::Decision, key).ok()?;
        let at = format!("{}:{line}", self.path);
        let mut p = BTreeMap::new();
        p.insert("text".into(), Value::String(text.trim().to_string()));
        p.insert("status".into(), Value::String(status.to_string()));
        p.insert("date".into(), Value::String(date.to_string()));
        b.define(id.clone(), at.clone(), text.trim(), p);
        b.text(&id, "text", text);
        for word in text.split(|c: char| c.is_whitespace() || matches!(c, ',' | '(' | ')' | '`')) {
            let w = word.trim_end_matches(['.', ':', ';']);
            if self.ledger.contains(w)
                && let Ok(l) = KgId::new(Label::LedgerEntry, w)
            {
                b.edge(&l, Rel::DecidedBy, &id, 1.0, &at);
            }
        }
        Some(id)
    }
}

/// 2026-10-23: Top-level list items (`- x` / `1. x`) with their indented continuation lines.
#[derive(Default)]
struct Items {
    items: Vec<(usize, String)>,
}

impl Items {
    /// 2026-10-23: Feed one line; `false` when the line ends the list.
    fn feed(&mut self, line_no: usize, l: &str) -> bool {
        if let Some(item) = item_text(l) {
            self.items.push((line_no, item.to_string()));
            return true;
        }
        if l.trim().is_empty() {
            return true;
        }
        match self.items.last_mut() {
            Some((_, text)) if l.starts_with([' ', '\t']) => {
                text.push('\n');
                text.push_str(l.trim());
                true
            }
            _ => false,
        }
    }

    fn flush(
        &mut self,
        cx: &mut Ctx<'_>,
        b: &mut Builder,
        date: &str,
        status: &str,
        sec: Option<&KgId>,
    ) {
        for (line, text) in core::mem::take(&mut self.items) {
            if let Some(d) = cx.decision(b, line, date, &text, status, 4)
                && let Some(sec) = sec
            {
                let at = format!("{}:{line}", cx.path);
                b.edge(sec, Rel::DecidedBy, &d, 1.0, &at);
            }
        }
    }
}

/// 2026-10-23: Derive the decisions of a CHARTER.md.
pub fn charter(b: &mut Builder, path: &str, text: &str, ledger: &BTreeSet<String>) {
    let mut cx = Ctx {
        path,
        seen: BTreeMap::new(),
        ledger,
    };
    let lines: Vec<&str> = text.lines().collect();
    let mut list: Option<String> = None;
    let mut items = Items::default();
    for (i, l) in lines.iter().enumerate() {
        if let Some(date) = list.clone() {
            if !l.starts_with("## ") && items.feed(i + 1, l) {
                continue;
            }
            items.flush(&mut cx, b, &date, "resolved", None);
            list = None;
        }
        if let Some(h) = l.strip_prefix("## ") {
            if let Some(rest) = h.strip_prefix("Decisions ") {
                list = date_in(rest);
            } else if h.starts_with("Re-plan ") {
                replan(&mut cx, b, &lines, i, h);
            }
        } else if l.starts_with("Owner decisions") {
            list = date_in(l);
        }
    }
    if let Some(date) = list {
        items.flush(&mut cx, b, &date, "resolved", None);
    }
}

fn replan(cx: &mut Ctx<'_>, b: &mut Builder, lines: &[&str], i: usize, h: &str) {
    let date = date_in(h).unwrap_or_else(|| "undated".into());
    let title = h.rsplit_once("): ").map_or(h, |(_, t)| t);
    let mut body = String::from(title);
    for l in lines[i + 1..].iter().take_while(|l| !l.starts_with("## ")) {
        body.push('\n');
        body.push_str(l);
    }
    let key = unique(&mut cx.seen, format!("{date}/{}", slug(title, 6)));
    if let Ok(id) = KgId::new(Label::Decision, key) {
        let at = format!("{}:{}", cx.path, i + 1);
        let mut p = BTreeMap::new();
        p.insert("text".into(), Value::String(body.trim().to_string()));
        p.insert("status".into(), Value::String("resolved".into()));
        p.insert("date".into(), Value::String(date));
        b.define(id.clone(), at, body.trim(), p);
        b.text(&id, "text", &body);
    }
}

fn section_key(title: &str) -> String {
    let first = title.split_whitespace().next().unwrap_or("");
    let num = first.trim_end_matches('.');
    if !num.is_empty() && num.chars().all(|c| c.is_ascii_digit() || c == '.') {
        num.to_string()
    } else {
        slug(title, 6)
    }
}

/// 2026-10-23: Derive the sections and owner decisions of one DESIGN file. Lines inside code
/// fences are body text, never headings.
pub fn design(b: &mut Builder, path: &str, text: &str, ledger: &BTreeSet<String>) {
    let file = path.rsplit('/').next().unwrap_or(path);
    let file_date = text
        .lines()
        .take(6)
        .find_map(date_in)
        .unwrap_or_else(|| "undated".into());
    let mut cx = Ctx {
        path,
        seen: BTreeMap::new(),
        ledger,
    };
    let mut sections: BTreeMap<String, u32> = BTreeMap::new();
    let mut current: Option<KgId> = None;
    let mut list: Option<(String, &'static str)> = None;
    let mut items = Items::default();
    let mut fence = false;
    let mut h1 = false;
    for (i, l) in text.lines().enumerate() {
        if l.trim_start().starts_with("```") {
            fence = !fence;
        }
        let hashes = if fence {
            0
        } else {
            l.chars().take_while(|c| *c == '#').count()
        };
        let heading = hashes > 0 && l[hashes..].starts_with(' ');
        if let Some((date, status)) = list.clone() {
            if !heading && items.feed(i + 1, l) {
                continue;
            }
            items.flush(&mut cx, b, &date, status, current.as_ref());
            list = None;
        }
        if !heading {
            continue;
        }
        let title = l[hashes..].trim();
        let key = if hashes == 1 && !h1 {
            h1 = true;
            file.to_string()
        } else {
            format!("{file}#{}", unique(&mut sections, section_key(title)))
        };
        if let Ok(id) = KgId::new(Label::Design, key) {
            let mut p = BTreeMap::new();
            p.insert("title".into(), Value::String(title.to_string()));
            p.insert("file".into(), Value::String(file.to_string()));
            b.define(id.clone(), format!("{path}:{}", i + 1), title, p);
            current = Some(id);
        }
        if title.contains("Owner decisions") {
            let status = if title.contains("needed") {
                "open"
            } else {
                "resolved"
            };
            list = Some((date_in(title).unwrap_or_else(|| file_date.clone()), status));
        }
    }
    if let Some((date, status)) = list {
        items.flush(&mut cx, b, &date, status, current.as_ref());
    }
}
