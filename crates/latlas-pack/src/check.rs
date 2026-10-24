// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The metrale pack's side of `latlas-check` (check part §1, K2): ledger rows whose
//! verdict needs a claim are the outcomes (C, the same rows `claims_without_gate` lists); a
//! `manual(...)` claim earns half; a retracting row states why in `why_differed`; and the
//! obligation rule is the LKB's four facets for every kernel an instance lands or compiles (O,
//! the `lkb_every_campaign` rule).
//!
//! Owner: metrale-latlas-pack.
//! Invariants: pure over the bundle; the rules are the ones the ledger and the LKB already state.

use crate::ledger_adapter::NO_CLAIM_VERDICTS;
use crate::owed::lkb_facets;
use crate::pack::Metrale;
use crate::prelude::*;
use crate::vocab::labels;
use latlas_check::{Checks, Key, Obligation};
use latlas_core::Bundle;
use latlas_core::bundle::Node;
use latlas_core::id::NodeId;
use serde_json::Value;

/// 2026-10-23: What a review-only claim is worth (check part §1 C: manual claims cannot reach 1).
pub const MANUAL_CREDIT: f64 = 0.5;

impl Checks for Metrale {
    fn is_outcome(n: &Node) -> bool {
        n.id.label() == labels::LedgerEntry
            && n
                .p
                .get("verdict_kind")
                .and_then(Value::as_str)
                .is_some_and(|v| !NO_CLAIM_VERDICTS.contains(&v))
    }

    fn claim_credit(b: &Bundle, c: &NodeId) -> f64 {
        match b.field(c, "kind") {
            Some("manual") => MANUAL_CREDIT,
            Some("none") | None => 0.0,
            Some(_) => 1.0,
        }
    }

    fn retraction_has_reason(b: &Bundle, r: &NodeId) -> bool {
        b.field(r, "why_differed")
            .is_some_and(|w| !w.trim().is_empty() && w.trim() != "-")
    }

    fn obligations(b: &Bundle) -> Vec<Obligation> {
        let mut out = Vec::new();
        for inst in b.of(labels::Instance) {
            let Ok(rows) = lkb_facets(b, inst.id.key()) else {
                continue;
            };
            for r in rows {
                let missing: Vec<String> = [
                    (r.has_point, "family point"),
                    (r.classified, "taxonomy component"),
                    (r.measured, "measurement"),
                    (r.evidenced, "evidence row"),
                ]
                .into_iter()
                .filter(|(ok, _)| !ok)
                .map(|(_, m)| m.to_string())
                .collect();
                out.push(Obligation {
                    rule: format!("lkb@{}", inst.id.key()),
                    subject: r.kernel,
                    missing,
                });
            }
        }
        out
    }

    /// 2026-10-24: Not keyed yet: ledger claims carry a gate, not a value, and CHARTER decisions
    /// carry no subject/slot field; X reports them as unkeyed rather than guessing a key.
    fn key(_: &Bundle, _: &Node) -> Option<Key> {
        None
    }
}
