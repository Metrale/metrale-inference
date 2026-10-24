// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The closures and counts the graph answers (queries part §1) as pure functions over
//! a [`Bundle`]: LatticeDB's executor has no aggregation, `OPTIONAL MATCH` or `NOT (pattern)`, so
//! these run here and are golden-tested without the DB.
//!
//! Owner: metrale-kg-derive.
//! Invariants:
//! - The ledger closures read the bundle's `LedgerEntry` nodes through [`ledger_links`] and run
//!   the one implementation in [`crate::ledger`], so `kg_impact` equals `ledger.py` by
//!   construction and the P0 gate proves it on the real file.

use crate::bundle::{Bundle, Node};
use crate::id::{KgId, Label, Rel};
use crate::ledger::{self, Links};
use crate::ledger_adapter::NO_CLAIM_VERDICTS;
use crate::prelude::*;
use metrale_error::{ErrorCode, Result, error};
use serde_json::Value;

fn strs(n: &Node, key: &str) -> Vec<String> {
    n.p.get(key)
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn s<'a>(n: &'a Node, key: &str) -> &'a str {
    n.p.get(key).and_then(Value::as_str).unwrap_or("")
}

/// 2026-10-23: The ledger rows of `campaign`, in file order.
pub fn ledger_rows<'a>(b: &'a Bundle, campaign: &str) -> Vec<&'a Node> {
    let mut rows: Vec<&Node> = b
        .of(Label::LedgerEntry)
        .filter(|n| s(n, "campaign") == campaign)
        .collect();
    rows.sort_by_key(|n| n.p.get("row").and_then(Value::as_u64).unwrap_or(u64::MAX));
    rows
}

/// 2026-10-23: The closures' input, read from the bundle.
pub fn ledger_links(b: &Bundle, campaign: &str) -> Vec<Links> {
    ledger_rows(b, campaign)
        .into_iter()
        .map(|n| Links {
            id: n.id.key().to_string(),
            depends_on: strs(n, "depends_on_ids"),
            retracts: strs(n, "retracts_ids"),
        })
        .collect()
}

/// 2026-10-23: Refuse closures over a ledger its checker rejects (ledger.py: "ledger invalid").
pub fn ledger_valid(b: &Bundle, ledger_path: &str) -> Result<()> {
    let bad = b.errors.iter().any(|f| {
        f.code == ErrorCode::KgBadLedgerRow.name() && f.at.split(':').next() == Some(ledger_path)
    });
    if bad {
        return Err(error!(
            ErrorCode::KgBadLedgerRow,
            ledger_path.to_string(),
            ledger::INVALID
        ));
    }
    Ok(())
}

/// 2026-10-23: One row of a closure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reach {
    pub id: KgId,
    pub why: String,
    pub depth: u32,
}

/// 2026-10-23: `kg_impact`: what `root` rests on (`upstream`) or what rests on it
/// (`downstream`), breadth-first to `max_depth`, following `rels` (out-edges upstream, in-edges
/// downstream, plus a downstream ledger row's out-edges to the profile it pinned, the runs it
/// measured and the decisions that cite it). Returns the closure and whether depth cut it.
pub fn impact(
    b: &Bundle,
    root: &KgId,
    downstream: bool,
    rels: &[Rel],
    max_depth: u32,
) -> Result<(Vec<Reach>, bool)> {
    if b.node(root).is_none() {
        return Err(error!(ErrorCode::KgUnknownId, root.to_string()));
    }
    let mut seen: BTreeSet<KgId> = BTreeSet::new();
    seen.insert(root.clone());
    let mut out = Vec::new();
    let mut frontier = vec![root.clone()];
    let mut truncated = false;
    for depth in 1..=max_depth.saturating_add(1) {
        let mut next = Vec::new();
        for v in &frontier {
            let mut step: Vec<(KgId, String)> = Vec::new();
            for e in &b.edges {
                if !rels.contains(&e.rel) {
                    continue;
                }
                if !downstream && e.from == *v {
                    step.push((e.to.clone(), format!("{} {}", e.rel.name(), e.to)));
                } else if downstream && e.to == *v {
                    step.push((e.from.clone(), format!("{} {v}", e.rel.name())));
                }
            }
            if downstream {
                for e in b.edges.iter().filter(|e| e.from == *v) {
                    if matches!(e.rel, Rel::Pins | Rel::MeasuredIn | Rel::DecidedBy)
                        && rels.contains(&e.rel)
                    {
                        step.push((e.to.clone(), format!("{} by {v}", e.rel.name())));
                    }
                }
            }
            for (to, why) in step {
                if seen.contains(&to) {
                    continue;
                }
                if depth > max_depth {
                    truncated = true;
                    continue;
                }
                seen.insert(to.clone());
                out.push(Reach {
                    id: to.clone(),
                    why,
                    depth,
                });
                next.push(to);
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    out.sort_by(|a, b| (a.depth, &a.id).cmp(&(b.depth, &b.id)));
    Ok((out, truncated))
}

/// 2026-10-23: Q1: every BLOCKED row no later row resolves (`RESOLVED_BY`, from the ledger's
/// `resolves` column), with its blockers and the levers it toggles.
pub fn blocked_by_gate(b: &Bundle) -> Vec<(KgId, String, Vec<KgId>, Vec<KgId>)> {
    let mut out = Vec::new();
    for n in b.of(Label::LedgerEntry) {
        if s(n, "verdict_kind") != "BLOCKED" || b.out(&n.id, Rel::ResolvedBy).next().is_some() {
            continue;
        }
        let blockers = b.out(&n.id, Rel::BlockedBy).map(|e| e.to.clone()).collect();
        let levers = b.out(&n.id, Rel::Toggles).map(|e| e.to.clone()).collect();
        out.push((n.id.clone(), s(n, "phase").to_string(), blockers, levers));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// 2026-10-23: Q2: rows whose verdict needs a claim (every verdict but BLOCKED/ABORT/n/a) and
/// that have no `CLAIMS` edge to a claim `GATED_BY` a gate. `strict` marks WIN/LOSS/NULL.
pub fn claims_without_gate(b: &Bundle) -> Vec<(KgId, String, bool)> {
    let mut out = Vec::new();
    for n in b.of(Label::LedgerEntry) {
        let v = s(n, "verdict_kind");
        if NO_CLAIM_VERDICTS.contains(&v) {
            continue;
        }
        let gated = b
            .out(&n.id, Rel::Claims)
            .any(|c| b.out(&c.to, Rel::GatedBy).next().is_some());
        if !gated {
            out.push((
                n.id.clone(),
                v.to_string(),
                matches!(v, "WIN" | "LOSS" | "NULL"),
            ));
        }
    }
    out
}

/// 2026-10-23: Ledger verdict words per phase, for `progress`.
pub fn progress(b: &Bundle, campaign: &str) -> BTreeMap<String, BTreeMap<String, u32>> {
    let mut out: BTreeMap<String, BTreeMap<String, u32>> = BTreeMap::new();
    for n in ledger_rows(b, campaign) {
        *out.entry(s(n, "phase").to_string())
            .or_default()
            .entry(s(n, "verdict_kind").to_string())
            .or_default() += 1;
    }
    out
}

/// 2026-10-23: `Unresolved` nodes with what referenced each.
pub fn dangling(b: &Bundle) -> Vec<(KgId, Vec<KgId>)> {
    unresolved(b, false)
}

/// 2026-10-23: Memory forward links (targets not written yet), with who links to them.
pub fn forward_links(b: &Bundle) -> Vec<(KgId, Vec<KgId>)> {
    unresolved(b, true)
}

fn unresolved(b: &Bundle, forward: bool) -> Vec<(KgId, Vec<KgId>)> {
    b.of(Label::Unresolved)
        .filter(|u| (u.p.get("forward") == Some(&Value::Bool(true))) == forward)
        .map(|u| {
            let mut by: Vec<KgId> = b
                .edges
                .iter()
                .filter(|e| e.to == u.id)
                .map(|e| e.from.clone())
                .collect();
            by.dedup();
            (u.id.clone(), by)
        })
        .collect()
}

fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != *cb);
            cur.push(sub.min(prev[j + 1] + 1).min(cur[j] + 1));
        }
        prev = cur;
    }
    prev[b.len()]
}

/// 2026-10-23: Ids the caller probably meant by `raw` (a bare key, a wrong prefix or a typo):
/// every node whose key is `raw`'s key exactly, else up to three nodes within edit distance 3 of
/// it, nearest first. Empty when nothing is close.
pub fn suggest(b: &Bundle, raw: &str) -> Vec<KgId> {
    let key = raw.split_once(':').map_or(raw, |(_, k)| k);
    let exact: Vec<KgId> = b
        .nodes
        .iter()
        .filter(|n| n.id.key() == key)
        .map(|n| n.id.clone())
        .collect();
    if !exact.is_empty() {
        return exact;
    }
    let mut near: Vec<(usize, &KgId)> = b
        .nodes
        .iter()
        .filter(|n| n.id.label() != Label::Unresolved)
        .map(|n| (edit_distance(key, n.id.key()), &n.id))
        .filter(|(d, _)| *d <= 3)
        .collect();
    near.sort();
    near.into_iter().take(3).map(|(_, id)| id.clone()).collect()
}

/// 2026-10-23: Parse `raw` and require the node, or refuse naming the closest ids and the
/// prefixes, so a first call never dead-ends.
pub fn resolve_id(b: &Bundle, raw: &str) -> Result<KgId> {
    let hint = |raw: &str| {
        let s = suggest(b, raw);
        if s.is_empty() {
            String::new()
        } else {
            let names: Vec<String> = s.iter().map(ToString::to_string).collect();
            format!(" (did you mean {}?)", names.join(" or "))
        }
    };
    match KgId::parse(raw) {
        Ok(id) if b.node(&id).is_some() => Ok(id),
        Ok(id) => Err(error!(ErrorCode::KgUnknownId, format!("{id}{}", hint(raw)))),
        Err(_) => Err(error!(
            ErrorCode::KgBadId,
            raw.to_string(),
            format!(
                "{}{}; prefixes: {}",
                if raw.contains(':') {
                    "unknown label prefix"
                } else {
                    "no `<prefix>:` part"
                },
                hint(raw),
                Label::prefixes()
            )
        )),
    }
}
