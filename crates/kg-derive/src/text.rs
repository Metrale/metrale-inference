// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Token scanners the adapters share: markdown table cells, env levers, commit shas,
//! principle ids, design references, arXiv ids, wikilinks and slugs. Hand-written (no regex
//! engine in the core); each returns tokens in first-occurrence order without duplicates.
//!
//! Owner: metrale-kg-derive.
//! Invariants: scanners never allocate per byte and never panic on any input.

use crate::prelude::*;

fn push_unique(out: &mut Vec<String>, s: &str) {
    if !out.iter().any(|x| x == s) {
        out.push(s.to_string());
    }
}

/// 2026-10-23: The cells of a markdown table row (`| a | b |`), trimmed; a `|` inside a backtick
/// span or escaped as `\|` does not split. `None` when the line is not a table row.
pub fn table_cells(line: &str) -> Option<Vec<String>> {
    let t = line.trim();
    let inner = t.strip_prefix('|')?;
    let inner = inner.strip_suffix('|').unwrap_or(inner);
    let mut cells = Vec::new();
    let mut cur = String::new();
    let mut tick = false;
    let mut prev_backslash = false;
    for c in inner.chars() {
        match c {
            '`' => {
                tick = !tick;
                cur.push(c);
            }
            '|' if !tick && !prev_backslash => {
                cells.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(c),
        }
        prev_backslash = c == '\\';
    }
    cells.push(cur.trim().to_string());
    Some(cells)
}

/// 2026-10-23: Whether a table row is the `|---|---|` separator.
pub fn is_separator(cells: &[String]) -> bool {
    cells
        .iter()
        .all(|c| !c.is_empty() && c.chars().all(|ch| matches!(ch, '-' | ':' | ' ')))
}

fn words(text: &str, ok: impl Fn(char) -> bool) -> impl Iterator<Item = &str> {
    text.split(move |c: char| !ok(c)).filter(|w| !w.is_empty())
}

/// 2026-10-23: `METRALE_[A-Z0-9_]+` env lever names.
pub fn env_levers(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for w in words(text, |c| {
        c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'
    }) {
        if let Some(rest) = w.strip_prefix("METRALE_") {
            let w = w.trim_end_matches('_');
            if !rest.trim_end_matches('_').is_empty() {
                push_unique(&mut out, w);
            }
        }
    }
    out
}

/// 2026-10-23: Commit-sha-shaped tokens: 7 to 40 lowercase hex digits holding a digit and a
/// letter (so a bare number or a word is not a sha).
pub fn commit_shas(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for w in words(text, |c| c.is_ascii_alphanumeric()) {
        let hex = w
            .chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c));
        let digit = w.chars().any(|c| c.is_ascii_digit());
        let letter = w.chars().any(|c| c.is_ascii_alphabetic());
        if (7..=40).contains(&w.len()) && hex && digit && letter {
            push_unique(&mut out, w);
        }
    }
    out
}

/// 2026-10-23: The ids of `known` that appear as whole upper-case words in `text`.
pub fn principle_ids(text: &str, known: &BTreeSet<String>) -> Vec<String> {
    let mut out = Vec::new();
    for w in words(text, |c| c.is_ascii_uppercase() || c.is_ascii_digit()) {
        if known.contains(w) {
            push_unique(&mut out, w);
        }
    }
    out
}

/// 2026-10-23: Design references: `DESIGN-<name>.md` optionally followed by `§<section>`, as
/// `(file, Some(section))` keys `DESIGN-x.md#6` / `DESIGN-x.md`.
pub fn design_refs(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find("DESIGN-") {
        let tail = &rest[i..];
        let end = tail
            .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')))
            .unwrap_or(tail.len());
        let name = tail[..end].trim_end_matches('.');
        if name.ends_with(".md") {
            let after = tail[end..].trim_start();
            let sec = after.strip_prefix('§').map(|s| {
                let e = s
                    .find(|c: char| !(c.is_ascii_digit() || c == '.'))
                    .unwrap_or(s.len());
                s[..e].trim_end_matches('.')
            });
            match sec {
                Some(s) if !s.is_empty() => push_unique(&mut out, &format!("{name}#{s}")),
                _ => push_unique(&mut out, name),
            }
        }
        rest = &tail[end.max(1)..];
    }
    out
}

/// 2026-10-23: arXiv ids `NNNN.NNNNN` (4 digits, a dot, 4 or 5 digits).
pub fn arxiv_ids(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for w in words(text, |c| c.is_ascii_digit() || c == '.') {
        let w = w.trim_matches('.');
        if let Some((a, b)) = w.split_once('.') {
            let digits = |s: &str| s.chars().all(|c| c.is_ascii_digit());
            if a.len() == 4 && (4..=5).contains(&b.len()) && digits(a) && digits(b) {
                push_unique(&mut out, w);
            }
        }
    }
    out
}

/// 2026-10-23: `[[name]]` wikilink targets (an `|alias` part is dropped).
pub fn wikilinks(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find("[[") {
        let tail = &rest[i + 2..];
        let Some(j) = tail.find("]]") else { break };
        let target = tail[..j].split('|').next().unwrap_or("").trim();
        if !target.is_empty() && !target.contains('\n') {
            push_unique(&mut out, target);
        }
        rest = &tail[j + 2..];
    }
    out
}

/// 2026-10-23: `](name.md)` markdown link targets without a directory part, as `name`.
pub fn md_links(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find("](") {
        let tail = &rest[i + 2..];
        let Some(j) = tail.find(')') else { break };
        let t = &tail[..j];
        if let Some(stem) = t.strip_suffix(".md")
            && !stem.is_empty()
            && !stem.contains('/')
            && !stem.contains(' ')
        {
            push_unique(&mut out, stem);
        }
        rest = &tail[j..];
    }
    out
}

/// 2026-10-23: The first `n` words of `text` as a lowercase `-`-joined slug of `[a-z0-9]`.
pub fn slug(text: &str, n: usize) -> String {
    let lower = text.to_ascii_lowercase();
    let parts: Vec<&str> = words(&lower, |c| c.is_ascii_alphanumeric())
        .take(n)
        .collect();
    parts.join("-")
}

/// 2026-10-23: The first decimal number in `text` (`<=315 us` → 315.0).
pub fn first_number(text: &str) -> Option<f64> {
    let b = text.as_bytes();
    let start = b.iter().position(u8::is_ascii_digit)?;
    let mut end = start;
    let mut dot = false;
    while end < b.len() && (b[end].is_ascii_digit() || (b[end] == b'.' && !dot)) {
        dot |= b[end] == b'.';
        end += 1;
    }
    let neg = start > 0 && b[start - 1] == b'-';
    let v: f64 = text[start..end].trim_end_matches('.').parse().ok()?;
    Some(if neg { -v } else { v })
}

/// 2026-10-23: The text of every backtick span.
pub fn backtick_spans(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut parts = text.split('`');
    parts.next();
    while let Some(inner) = parts.next() {
        if !inner.trim().is_empty() {
            push_unique(&mut out, inner.trim());
        }
        parts.next();
    }
    out
}
