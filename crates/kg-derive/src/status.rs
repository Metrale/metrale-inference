// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `kg_status` sections as JSON rows (design §3 tool 5), computed from the bundle by
//! [`crate::queries`]. Tables, not prose.
//!
//! Owner: metrale-kg-derive.
//! Invariants: an unknown section name is refused (`KgToolArgument`), never ignored.

use crate::bundle::Bundle;
use crate::config::Config;
use crate::id::{Label, Rel};
use crate::ledger;
use crate::prelude::*;
use crate::queries;
use crate::tools::SECTIONS;
use metrale_error::{ErrorCode, Result, error};
use serde_json::{Map, Value};

/// 2026-10-23: What the binary knows about freshness that the pure bundle cannot (stat results).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Freshness {
    /// 2026-10-23: Set when the last re-derive failed (then the bundle is the previous one).
    pub stale_reason: Option<String>,
    /// 2026-10-23: When the bundle was derived (unix ms, from the Clock port).
    pub derived_at_ms: i64,
}

fn ids(v: &[crate::id::KgId]) -> Value {
    Value::from(v.iter().map(ToString::to_string).collect::<Vec<_>>())
}

/// 2026-10-23: The Cypher features LatticeDB's executor lacks (design §1.1), for `schema`.
pub const CYPHER_UNSUPPORTED: &[&str] = &[
    "OPTIONAL MATCH",
    "MERGE",
    "UNWIND",
    "SET",
    "shortestPath",
    "function calls (count, collect, labels, length, exists)",
];

/// 2026-10-23: One section's JSON.
pub fn section(
    b: &Bundle,
    cfg: &Config,
    campaign: &str,
    name: &str,
    fresh: &Freshness,
    instance: Option<&str>,
) -> Result<Value> {
    let c = cfg.campaign(campaign)?;
    Ok(match name {
        "progress" => {
            let mut o = Map::new();
            for (phase, v) in queries::progress(b, campaign) {
                let m: Map<String, Value> =
                    v.into_iter().map(|(k, n)| (k, Value::from(n))).collect();
                o.insert(phase, Value::Object(m));
            }
            Value::Object(o)
        }
        "blocked" => Value::from(
            queries::blocked_by_gate(b)
                .into_iter()
                .map(|(id, phase, blockers, levers)| {
                    let mut o = Map::new();
                    o.insert("id".into(), Value::String(id.to_string()));
                    o.insert("phase".into(), Value::String(phase));
                    let kinds: Vec<String> = blockers
                        .iter()
                        .map(|x| x.label().name().to_string())
                        .collect();
                    o.insert("blocker_kinds".into(), Value::from(kinds));
                    o.insert("blockers".into(), ids(&blockers));
                    o.insert("levers".into(), ids(&levers));
                    Value::Object(o)
                })
                .collect::<Vec<_>>(),
        ),
        "tainted" => {
            queries::ledger_valid(b, &c.ledger)?;
            let t = ledger::tainted(&queries::ledger_links(b, campaign));
            Value::from(
                t.into_iter()
                    .map(|(id, why)| Value::from(vec![Value::String(id), Value::String(why)]))
                    .collect::<Vec<_>>(),
            )
        }
        "claims" => Value::from(
            queries::claims_without_gate(b)
                .into_iter()
                .map(|(id, v, strict)| {
                    let mut o = Map::new();
                    o.insert("id".into(), Value::String(id.to_string()));
                    o.insert("verdict".into(), Value::String(v));
                    o.insert("strict".into(), Value::Bool(strict));
                    Value::Object(o)
                })
                .collect::<Vec<_>>(),
        ),
        "dangling" => {
            let mut rows: Vec<Value> = queries::dangling(b)
                .into_iter()
                .map(|(u, by)| {
                    let mut o = Map::new();
                    o.insert("id".into(), Value::String(u.to_string()));
                    o.insert("referenced_by".into(), ids(&by));
                    Value::Object(o)
                })
                .collect();
            rows.truncate(500);
            Value::from(rows)
        }
        "owed_lkb" | "calibration" => {
            let _ = instance;
            return Err(error!(
                ErrorCode::NotImplemented,
                name.to_string(),
                "the campaign-record bundle (LKB sources arrive in P1)"
            ));
        }
        "freshness" => {
            let mut o = Map::new();
            o.insert("derived_at_ms".into(), Value::from(fresh.derived_at_ms));
            o.insert(
                "stale_reason".into(),
                fresh
                    .stale_reason
                    .clone()
                    .map_or(Value::Null, Value::String),
            );
            o.insert("sources".into(), Value::from(b.sources.len()));
            let files: Vec<Value> = b
                .sources
                .iter()
                .filter(|s| !s.path.starts_with(&format!("{}/", cfg.sources.memory_dir)))
                .map(|s| {
                    Value::from(vec![
                        Value::String(s.path.clone()),
                        Value::String(s.sha.hex()[..12].to_string()),
                    ])
                })
                .collect();
            o.insert("files".into(), Value::from(files));
            o.insert(
                "memory_dir".into(),
                Value::String(cfg.sources.memory_dir.clone()),
            );
            Value::Object(o)
        }
        "schema" => {
            let mut o = Map::new();
            let labels: Map<String, Value> = Label::ALL
                .iter()
                .map(|l| (l.name().to_string(), Value::from(b.of(*l).count())))
                .collect();
            let rels: Map<String, Value> = Rel::ALL
                .iter()
                .map(|r| {
                    (
                        r.name().to_string(),
                        Value::from(b.edges.iter().filter(|e| e.rel == *r).count()),
                    )
                })
                .collect();
            o.insert("labels".into(), Value::Object(labels));
            o.insert("relations".into(), Value::Object(rels));
            o.insert(
                "cypher_unsupported".into(),
                Value::from(CYPHER_UNSUPPORTED.to_vec()),
            );
            o.insert(
                "node_property".into(),
                Value::String("n.id is the typed id; payload fields as in the bundle".into()),
            );
            Value::Object(o)
        }
        other => {
            return Err(error!(
                ErrorCode::KgToolArgument,
                "sections",
                format!("unknown section {other}")
            ));
        }
    })
}

/// 2026-10-23: The requested sections (all when `names` is empty), keyed by name.
pub fn status(
    b: &Bundle,
    cfg: &Config,
    campaign: &str,
    names: &[String],
    fresh: &Freshness,
    instance: Option<&str>,
) -> Result<Value> {
    let all: Vec<String> = if names.is_empty() {
        SECTIONS.iter().map(|s| (*s).to_string()).collect()
    } else {
        names.to_vec()
    };
    let mut o = Map::new();
    for n in &all {
        o.insert(n.clone(), section(b, cfg, campaign, n, fresh, instance)?);
    }
    Ok(Value::Object(o))
}
