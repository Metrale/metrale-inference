// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: LEDGER.tsv → `LedgerEntry` nodes and their edges (ontology §2, §3): every column
//! verbatim, plus the parsed fields the queries read (verdict word, predicted µs, measured
//! intervals, ordered reference lists). `ledger.py check` findings become `KgBadLedgerRow`.
//!
//! Owner: metrale-latlas-pack (the metrale pack; was metrale-kg-derive).
//! Invariants:
//! - A row ledger.py would reject for its field count yields no node (as ledger.py skips it), and
//!   the finding names its line.
//! - A BLOCKED row whose text names no blocker gets an explicit `Unresolved` blocker: the gap is
//!   shown, never filled in.

use crate::claims;
use crate::ledger::{RETRACTS, Table, refs};
use crate::prelude::*;
use crate::vocab::*;
use latlas_core::bundle::Builder;
use latlas_core::text::{commit_shas, design_refs, env_levers, number_with_unit, principle_ids};
use latlas_error::{ErrorCode, error};
use serde_json::Value;

/// 2026-10-23: Verdict words a row may carry without a claim (`gate` = `-`; queries part §3).
pub const NO_CLAIM_VERDICTS: &[&str] = &["BLOCKED", "ABORT", "N/A"];

/// 2026-10-23: The verdict word: the leading upper-case token (`NULL (no win...)` → `NULL`,
/// `n/a` → `N/A`), or `UNPARSED` when the column starts with none.
pub fn verdict_kind(verdict: &str) -> String {
    let t = verdict.trim();
    if t.eq_ignore_ascii_case("n/a") || t.to_ascii_lowercase().starts_with("n/a ") {
        return "N/A".to_string();
    }
    let w: String = t
        .chars()
        .take_while(|c| c.is_ascii_uppercase() || *c == '_' || *c == '-')
        .collect();
    let w = w.trim_end_matches('-');
    if w.len() >= 2 {
        w.to_string()
    } else {
        "UNPARSED".to_string()
    }
}

/// 2026-10-23: `<isl>k <est>% [<lo>,<hi>]` intervals of a measured column, by ISL.
pub fn measured(text: &str) -> BTreeMap<String, Vec<f64>> {
    let mut out = BTreeMap::new();
    let toks: Vec<&str> = text.split_whitespace().collect();
    for i in 0..toks.len().saturating_sub(1) {
        let isl = toks[i].trim_matches(|c: char| c == ';' || c == ',' || c == ':');
        let num = isl
            .strip_suffix('k')
            .filter(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()));
        let Some(_) = num else { continue };
        let Some(est) = toks[i + 1]
            .strip_suffix('%')
            .and_then(|e| e.parse::<f64>().ok())
        else {
            continue;
        };
        let mut v = vec![est];
        if let Some(ci) = toks.get(i + 2).and_then(|t| t.strip_prefix('[')) {
            let ci = ci.trim_end_matches([';', ',']).trim_end_matches(']');
            if let Some((lo, hi)) = ci.split_once(',')
                && let (Ok(lo), Ok(hi)) = (lo.parse::<f64>(), hi.parse::<f64>())
            {
                v.extend([lo, hi]);
            }
        }
        out.entry(isl.to_string()).or_insert(v);
    }
    out
}

fn profile_names(col: &str) -> Vec<String> {
    let first = col.split_whitespace().next().unwrap_or("");
    let ok = !first.is_empty()
        && first != "-"
        && first
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '.')
        && first.chars().any(|c| c.is_ascii_digit() || c == '-');
    if ok {
        vec![first.to_string()]
    } else {
        Vec::new()
    }
}

/// 2026-10-23: The ids a `blocked on <id>` phrase names (`blocked on gate:test/x`, `blocked on
/// p0-split`), in order.
fn blockers(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let lower = text.to_ascii_lowercase();
    let mut from = 0;
    while let Some(i) = lower[from..].find("blocked on ") {
        let start = from + i + "blocked on ".len();
        let tok: String = text[start..]
            .chars()
            .take_while(|c| !c.is_whitespace() && !matches!(c, ',' | ';' | ')' | '('))
            .collect();
        let tok = tok.trim_end_matches('.').to_string();
        if !tok.is_empty() {
            out.push(tok);
        }
        from = start;
    }
    out
}

/// 2026-10-23: Derive one campaign's ledger into `b`; returns the row ids it defined.
pub fn derive(
    b: &mut Builder,
    campaign: &str,
    path: &str,
    text: &str,
    principles: &BTreeSet<String>,
) -> BTreeSet<String> {
    let table = Table::parse(text);
    for msg in table.check() {
        b.error(
            &error!(ErrorCode::LatlasBadLedgerRow, path.to_string(), msg),
            path,
        );
    }
    let h = table.header.clone();
    let col = |r: &Vec<String>, name: &str| -> String {
        h.iter()
            .position(|x| x == name)
            .and_then(|i| r.get(i))
            .cloned()
            .unwrap_or_default()
    };
    let lines: Vec<&str> = text.trim_end_matches('\n').split('\n').skip(1).collect();
    let known: BTreeSet<String> = table.links().into_iter().map(|l| l.id).collect();
    let resolved: BTreeSet<String> = table
        .rows
        .iter()
        .flat_map(|r| refs(&col(r, "resolves")))
        .collect();
    let mut ids = BTreeSet::new();
    for (i, r) in table.rows.iter().enumerate() {
        if r.len() != h.len() || r.len() <= RETRACTS {
            continue;
        }
        let line = i + 2;
        let at = format!("{path}:{line}");
        let Ok(id) = NodeId::new(labels::LedgerEntry, r[2].clone()) else {
            b.error(
                &error!(ErrorCode::LatlasBadLedgerRow, at.clone(), "empty id"),
                &at,
            );
            continue;
        };
        ids.insert(r[2].clone());
        let mut p = BTreeMap::new();
        for (k, v) in h.iter().zip(r) {
            p.insert(k.clone(), Value::String(v.clone()));
        }
        let verdict = verdict_kind(&col(r, "verdict"));
        if verdict == "UNPARSED" {
            let e = error!(
                ErrorCode::LatlasBadRow,
                at.clone(),
                "the verdict column starts with no verdict word (columns shifted?)"
            );
            b.error(&e, &at);
        }
        let pred = col(r, "predicted_us_saved");
        let predicted = number_with_unit(&pred, "us");
        let meas = measured(&col(r, "measured_delta"));
        p.insert("campaign".into(), Value::String(campaign.to_string()));
        p.insert("row".into(), Value::from(i + 1));
        p.insert("verdict_kind".into(), Value::String(verdict.clone()));
        p.insert(
            "predicted_us".into(),
            predicted.map_or(Value::Null, Value::from),
        );
        let mv: serde_json::Map<String, Value> = meas
            .iter()
            .map(|(k, v)| (k.clone(), Value::from(v.clone())))
            .collect();
        p.insert(
            "measured".into(),
            if mv.is_empty() {
                Value::Null
            } else {
                Value::Object(mv)
            },
        );
        let deps = refs(&col(r, "depends_on"));
        let rets = refs(&col(r, "retracts"));
        p.insert("depends_on_ids".into(), Value::from(deps.clone()));
        p.insert("retracts_ids".into(), Value::from(rets.clone()));
        b.define(
            id.clone(),
            at.clone(),
            lines.get(i).copied().unwrap_or(""),
            p,
        );
        let mut link = |rel: Rel, label: Label, key: &str| {
            if let Ok(to) = NodeId::new(label, key) {
                b.edge(&id, rel, &to, 1.0, &at);
            }
        };
        for d in &deps {
            link(rels::DependsOn, labels::LedgerEntry, d);
        }
        for x in &rets {
            link(rels::Retracts, labels::LedgerEntry, x);
        }
        for pr in profile_names(&col(r, "profile")) {
            link(rels::Pins, labels::Profile, &pr);
        }
        for c in commit_shas(&col(r, "commit")) {
            link(rels::ImplementedBy, labels::Commit, &c);
        }
        let hyp = col(r, "hypothesis");
        let why = col(r, "why_differed");
        for lv in env_levers(&hyp) {
            link(rels::Toggles, labels::Lever, &lv);
        }
        let prose = format!("{hyp}\n{pred}\n{why}");
        for pid in principle_ids(&prose, principles) {
            link(rels::Touches, labels::Principle, &pid);
        }
        for d in design_refs(&prose) {
            link(rels::Cites, labels::Design, &d);
        }
        for x in refs(&col(r, "resolves")) {
            if let Ok(from) = NodeId::new(labels::LedgerEntry, x) {
                b.edge(&from, rels::ResolvedBy, &id, 1.0, &at);
            }
        }
        b.text(&id, "hypothesis", &hyp);
        b.text(&id, "why_differed", &why);
        if verdict == "BLOCKED" && !resolved.contains(&r[2]) {
            blocked(b, &id, &at, &format!("{} {why}", col(r, "verdict")), &known);
        }
        if h.iter().any(|x| x == "gate") {
            claims::derive_row(b, &id, &at, &col(r, "gate"), &verdict);
        }
    }
    ids
}

fn blocked(b: &mut Builder, id: &NodeId, at: &str, text: &str, known: &BTreeSet<String>) {
    let mut any = false;
    for tok in blockers(text) {
        let target = NodeId::parse::<crate::Metrale>(&tok).ok().or_else(|| {
            known
                .contains(&tok)
                .then(|| NodeId::new(labels::LedgerEntry, tok.clone()).ok())
                .flatten()
        });
        if let Some(t) = target {
            b.edge(id, rels::BlockedBy, &t, 1.0, at);
            any = true;
        }
    }
    if !any {
        let Ok(u) = NodeId::new(labels::Unresolved, format!("gate:blocker-of-{}", id.key())) else {
            return;
        };
        let mut p = BTreeMap::new();
        p.insert(
            "target".into(),
            Value::String("the blocker of a BLOCKED row".into()),
        );
        p.insert(
            "reason".into(),
            Value::String("no `blocked on <id>` in verdict or why_differed".into()),
        );
        b.define(u.clone(), at.to_string(), &u.to_string(), p);
        b.edge(id, rels::BlockedBy, &u, 1.0, at);
        b.error(
            &error!(ErrorCode::LatlasDanglingRef, at.to_string(), u.to_string()),
            at,
        );
    }
}
