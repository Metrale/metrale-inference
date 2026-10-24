// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The campaign-record queries (queries part §1: Q1, Q2, progress, calibration) as
//! pure functions over a [`Bundle`]; the generic closures (impact, dangling, suggest) are the
//! core's (`latlas_core::queries`).
//!
//! Owner: metrale-latlas-pack (cut from metrale-kg-derive `queries.rs`, 2026-10-23 K0/K1).
//! Invariants:
//! - The ledger closures read the bundle's `LedgerEntry` nodes through [`ledger_links`] and run
//!   the one implementation in [`crate::ledger`], so `la_impact` equals `ledger.py` by
//!   construction and the parity gate proves it on the real file.

use crate::ledger::{self, Links};
use crate::ledger_adapter::NO_CLAIM_VERDICTS;
use crate::prelude::*;
use crate::vocab::*;
use latlas_core::bundle::{Bundle, Node};
use latlas_error::{ErrorCode, Result, error};
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
        .of(labels::LedgerEntry)
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
        f.code == ErrorCode::LatlasBadLedgerRow.name()
            && f.at.split(':').next() == Some(ledger_path)
    });
    if bad {
        return Err(error!(
            ErrorCode::LatlasBadLedgerRow,
            ledger_path.to_string(),
            ledger::INVALID
        ));
    }
    Ok(())
}

/// 2026-10-23: Q1: every BLOCKED row no later row resolves (`RESOLVED_BY`, from the ledger's
/// `resolves` column), with its blockers and the levers it toggles.
pub fn blocked_by_gate(b: &Bundle) -> Vec<(NodeId, String, Vec<NodeId>, Vec<NodeId>)> {
    let mut out = Vec::new();
    for n in b.of(labels::LedgerEntry) {
        if s(n, "verdict_kind") != "BLOCKED" || b.out(&n.id, rels::ResolvedBy).next().is_some() {
            continue;
        }
        let blockers = b
            .out(&n.id, rels::BlockedBy)
            .map(|e| e.to.clone())
            .collect();
        let levers = b.out(&n.id, rels::Toggles).map(|e| e.to.clone()).collect();
        out.push((n.id.clone(), s(n, "phase").to_string(), blockers, levers));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// 2026-10-23: Q2: rows whose verdict needs a claim (every verdict but BLOCKED/ABORT/n/a) and
/// that have no `CLAIMS` edge to a claim `GATED_BY` a gate. `strict` marks WIN/LOSS/NULL.
pub fn claims_without_gate(b: &Bundle) -> Vec<(NodeId, String, bool)> {
    let mut out = Vec::new();
    for n in b.of(labels::LedgerEntry) {
        let v = s(n, "verdict_kind");
        if NO_CLAIM_VERDICTS.contains(&v) {
            continue;
        }
        let gated = b
            .out(&n.id, rels::Claims)
            .any(|c| b.out(&c.to, rels::GatedBy).next().is_some());
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

/// 2026-10-23: How a verdict settles a predicted saving (P4 calibration, declared): `hit` = WIN;
/// `miss` = LOSS/NULL/REFUTED; every other verdict leaves it `open`.
pub fn outcome(verdict_kind: &str) -> &'static str {
    match verdict_kind {
        "WIN" => "hit",
        "LOSS" | "NULL" | "REFUTED" => "miss",
        _ => "open",
    }
}

/// 2026-10-23: One ledger row that predicted a saving (`predicted_us > 0`), for calibration.
#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    pub id: NodeId,
    pub verdict: String,
    pub predicted_us: f64,
    /// The measured column as parsed (`<isl>k <est>% [lo,hi]`); `None` when nothing parsed.
    pub measured: Option<Value>,
    /// The headroom sites it cites (`PREDICTED_BY`) with their `upside_us`.
    pub sites: Vec<(NodeId, Option<f64>)>,
}

/// 2026-10-23: Every row of `campaign` that predicted a saving, in file order (queries part
/// "Calibration"). Units are never converted: a measured % stays a %.
pub fn calibration(b: &Bundle, campaign: &str) -> Vec<Prediction> {
    ledger_rows(b, campaign)
        .into_iter()
        .filter_map(|n| {
            let predicted_us = n.p.get("predicted_us").and_then(Value::as_f64)?;
            (predicted_us > 0.0).then(|| Prediction {
                id: n.id.clone(),
                verdict: s(n, "verdict_kind").to_string(),
                predicted_us,
                measured: n.p.get("measured").filter(|v| !v.is_null()).cloned(),
                sites: b
                    .out(&n.id, rels::PredictedBy)
                    .map(|e| {
                        let up = b
                            .node(&e.to)
                            .and_then(|h| h.p.get("upside_us"))
                            .and_then(Value::as_f64);
                        (e.to.clone(), up)
                    })
                    .collect(),
            })
        })
        .collect()
}
