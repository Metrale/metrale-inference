// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: RUNS.tsv (P3; ontology part §2 `Run`, `MEASURED_IN`, `TARGETS`): one `Run` node
//! per row keyed by its content-addressed `run_id`, `ledger:<id> MEASURED_IN run:<id>` for every
//! id in `ledger_ids`, and `run TARGETS box:<box>`.
//!
//! Owner: metrale-kg-derive.
//! Invariants:
//! - The header is exactly [`RUNS_COLUMNS`]; anything else is one `KgBadRow` for the file and no
//!   nodes (a shifted table must not produce wrong edges).
//! - A row's `run_id` must be the content hash of its other columns (`record::run_line`), so an
//!   edited row is caught.

use crate::bundle::Builder;
use crate::id::{KgId, Label, Rel};
use crate::ledger::refs;
use crate::prelude::*;
use crate::record::{RUNS_COLUMNS, run_id};
use metrale_error::{ErrorCode, error};
use serde_json::Value;

/// 2026-10-23: Derive `path`'s runs into `b`.
pub fn derive(b: &mut Builder, path: &str, text: &str) {
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap_or("").split('\t').collect();
    if header != RUNS_COLUMNS {
        let e = error!(
            ErrorCode::KgBadRow,
            format!("{path}:1"),
            format!("the header is not `{}`", RUNS_COLUMNS.join("\\t"))
        );
        b.error(&e, path);
        return;
    }
    for (i, l) in lines.enumerate() {
        let at = format!("{path}:{}", i + 2);
        let r: Vec<&str> = l.split('\t').collect();
        let bad = |b: &mut Builder, why: String| {
            b.error(&error!(ErrorCode::KgBadRow, at.clone(), why), &at);
        };
        if r.len() != RUNS_COLUMNS.len() {
            bad(
                b,
                format!("{} fields, expected {}", r.len(), RUNS_COLUMNS.len()),
            );
            continue;
        }
        let want = run_id(&r[1..].join("\t"));
        if r[0] != want {
            bad(
                b,
                format!("run_id {} is not the hash of its columns ({want})", r[0]),
            );
            continue;
        }
        let Ok(id) = KgId::new(Label::Run, r[0]) else {
            continue;
        };
        let mut p = BTreeMap::new();
        for (k, v) in RUNS_COLUMNS.iter().zip(&r).skip(1) {
            p.insert((*k).to_string(), Value::String((*v).to_string()));
        }
        b.define(id.clone(), at.clone(), l, p);
        for lid in refs(r[9]) {
            if let Ok(from) = KgId::new(Label::LedgerEntry, lid) {
                b.edge(&from, Rel::MeasuredIn, &id, 1.0, &at);
            }
        }
        if let Ok(bx) = KgId::new(Label::Box, r[5]) {
            b.edge(&id, Rel::Targets, &bx, 1.0, &at);
        }
    }
}
