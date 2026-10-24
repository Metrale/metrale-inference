// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `kg_status` sections as JSON rows (design §3 tool 5), computed from the bundle by
//! [`crate::queries`]. Tables, not prose.
//!
//! Owner: metrale-kg-derive.
//! Invariants: an unknown section name is refused (`KgToolArgument`), never ignored.

use crate::bundle::Bundle;
use crate::config::Config;
use crate::id::{KgId, Label, Rel};
use crate::ledger;
use crate::prelude::*;
use crate::queries;
use crate::tools::{DEFAULT_SECTIONS, LIST_CAP, SECTIONS};
use metrale_error::{ErrorCode, Result, error};
use serde_json::{Map, Value};

/// 2026-10-23: What the binary knows about freshness that the pure bundle cannot (stat results).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Freshness {
    /// 2026-10-23: Set when the last re-derive failed (then the bundle is the previous one).
    pub stale_reason: Option<String>,
    /// 2026-10-23: When the bundle was derived (unix ms, from the Clock port).
    pub derived_at_ms: i64,
    /// 2026-10-23: The LKB checkout's commit the bundle was derived from (`None`: no LKB source).
    pub source_commit: Option<String>,
    /// 2026-10-23: The most recently modified source file and when (unix ms), from the host's
    /// stat results.
    pub newest_change: Option<(String, i64)>,
}

/// 2026-10-23: Every source file but the memory directory's, as `[path, sha prefix]`.
fn files(b: &Bundle, cfg: &Config) -> Vec<Value> {
    let mem = format!("{}/", cfg.sources.memory_dir);
    b.sources
        .iter()
        .filter(|s| !s.path.starts_with(&mem))
        .map(|s| {
            Value::from(vec![
                Value::String(s.path.clone()),
                Value::String(s.sha.hex()[..12].to_string()),
            ])
        })
        .collect()
}

/// 2026-10-23: The default answer's form: every list longer than [`LIST_CAP`] keeps its first
/// items, and its total goes beside it (`<key>_total`; a section that is a list becomes
/// `{total, first}`), so nothing is dropped without saying how much.
pub fn compact(v: Value) -> Value {
    match v {
        Value::Array(a) => {
            let total = a.len();
            let first: Vec<Value> = a.into_iter().take(LIST_CAP).map(compact).collect();
            let mut o = Map::new();
            o.insert("total".into(), Value::from(total));
            o.insert("first".into(), Value::from(first));
            Value::Object(o)
        }
        Value::Object(o) => {
            let mut out = Map::new();
            for (k, v) in o {
                match v {
                    Value::Array(a) if a.len() > LIST_CAP => {
                        out.insert(format!("{k}_total"), Value::from(a.len()));
                        let first: Vec<Value> =
                            a.into_iter().take(LIST_CAP).map(compact_item).collect();
                        out.insert(k, Value::from(first));
                    }
                    Value::Array(a) => {
                        out.insert(
                            k,
                            Value::from(a.into_iter().map(compact_item).collect::<Vec<_>>()),
                        );
                    }
                    other => {
                        out.insert(k, compact_item(other));
                    }
                }
            }
            Value::Object(out)
        }
        other => other,
    }
}

/// 2026-10-23: [`compact`] below the top level: an inner list keeps the same cap.
fn compact_item(v: Value) -> Value {
    match v {
        Value::Object(_) => compact(v),
        Value::Array(a) if a.len() > LIST_CAP => {
            let mut keep: Vec<Value> = a.into_iter().take(LIST_CAP).map(compact_item).collect();
            keep.push(Value::String("…".into()));
            Value::from(keep)
        }
        Value::Array(a) => Value::from(a.into_iter().map(compact_item).collect::<Vec<_>>()),
        other => other,
    }
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
            let rows = |v: Vec<(KgId, Vec<KgId>)>| -> Vec<Value> {
                let mut rows: Vec<Value> = v
                    .into_iter()
                    .map(|(u, by)| {
                        let mut o = Map::new();
                        let kind = u.key().split_once(':').map_or("", |(k, _)| k).to_string();
                        o.insert("id".into(), Value::String(u.to_string()));
                        o.insert("kind".into(), Value::String(kind));
                        o.insert("referenced_by".into(), ids(&by));
                        Value::Object(o)
                    })
                    .collect();
                rows.truncate(500);
                rows
            };
            let refs = rows(queries::dangling(b));
            let fwd = rows(queries::forward_links(b));
            let mut by_kind: BTreeMap<String, u64> = BTreeMap::new();
            for r in &refs {
                *by_kind
                    .entry(r["kind"].as_str().unwrap_or("").to_string())
                    .or_insert(0) += 1;
            }
            let mut o = Map::new();
            o.insert("count".into(), Value::from(refs.len()));
            o.insert(
                "by_kind".into(),
                Value::Object(
                    by_kind
                        .into_iter()
                        .map(|(k, n)| (k, Value::from(n)))
                        .collect(),
                ),
            );
            o.insert("refs".into(), Value::from(refs));
            let uncommitted: Vec<Value> = crate::owed::uncommitted_measurements(b)
                .iter()
                .map(|m| Value::String(m.to_string()))
                .collect();
            o.insert(
                "no_commit_in_source_count".into(),
                Value::from(uncommitted.len()),
            );
            o.insert("no_commit_in_source".into(), Value::from(uncommitted));
            o.insert("forward_links_count".into(), Value::from(fwd.len()));
            o.insert("forward_links".into(), Value::from(fwd));
            Value::Object(o)
        }

        "owed_lkb" => owed_section(b, instance)?,
        "calibration" => {
            return Err(error!(
                ErrorCode::NotImplemented,
                name.to_string(),
                "prediction calibration (P4)"
            ));
        }

        "freshness" => {
            let mut o = Map::new();
            o.insert("derived_at_ms".into(), Value::from(fresh.derived_at_ms));
            o.insert(
                "source_commit".into(),
                fresh
                    .source_commit
                    .clone()
                    .map_or(Value::Null, Value::String),
            );
            o.insert(
                "stale_reason".into(),
                fresh
                    .stale_reason
                    .clone()
                    .map_or(Value::Null, Value::String),
            );
            o.insert(
                "digest".into(),
                Value::String(crate::jsonl::digest(b).hex()),
            );
            o.insert("sources".into(), Value::from(b.sources.len()));
            o.insert("files".into(), Value::from(files(b, cfg).len()));
            o.insert(
                "newest_change".into(),
                fresh.newest_change.as_ref().map_or(Value::Null, |(p, ms)| {
                    let mut n = Map::new();
                    n.insert("path".into(), Value::String(p.clone()));
                    n.insert("ts_ms".into(), Value::from(*ms));
                    Value::Object(n)
                }),
            );
            Value::Object(o)
        }
        "files" => {
            let mut o = Map::new();
            o.insert(
                "memory_dir".into(),
                Value::String(cfg.sources.memory_dir.clone()),
            );
            o.insert("files".into(), Value::from(files(b, cfg)));
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

/// 2026-10-23: Q3 for one instance: every kernel that owes an LKB facet, landed ones first,
/// with the measurements of it that predate a change of its file (Q4's `stale_since`).
fn owed_section(b: &Bundle, instance: Option<&str>) -> Result<Value> {
    let instances: Vec<String> = b
        .of(Label::Instance)
        .map(|n| n.id.key().to_string())
        .collect();
    if instances.is_empty() {
        return Err(error!(
            ErrorCode::KgSource,
            "sources.lkb", "is not configured: the graph holds no LKB"
        ));
    }
    if let Some(f) = b.errors.iter().find(|f| f.code == "KgInventoryRefused") {
        return Err(error!(ErrorCode::KgSource, f.at.clone(), f.msg.clone()));
    }
    let Some(inst) = instance else {
        return Err(error!(
            ErrorCode::KgToolArgument,
            "instance",
            format!("is required; instances: {}", instances.join(", "))
        ));
    };
    if !instances.iter().any(|i| i == inst) {
        return Err(error!(
            ErrorCode::KgToolArgument,
            "instance",
            format!("no instance `{inst}`; instances: {}", instances.join(", "))
        ));
    }
    let rows = crate::owed::owed_lkb_rows(b, inst)?;
    let landed = rows.iter().filter(|r| r.landed).count();
    let out: Vec<Value> = rows
        .iter()
        .map(|r| {
            let mut o = Map::new();
            o.insert("kernel".into(), Value::String(r.kernel.to_string()));
            o.insert("landed".into(), Value::Bool(r.landed));
            o.insert("has_point".into(), Value::Bool(r.has_point));
            o.insert("classified".into(), Value::Bool(r.classified));
            o.insert("measured".into(), Value::Bool(r.measured));
            o.insert("evidenced".into(), Value::Bool(r.evidenced));
            let stale: Vec<Value> = b
                .into(&r.kernel, Rel::Measures)
                .filter_map(|e| {
                    b.field(&e.from, "stale_since")
                        .map(|c| Value::String(format!("{} (file changed at {c})", e.from)))
                })
                .collect();
            o.insert("stale_since".into(), Value::from(stale));
            Value::Object(o)
        })
        .collect();
    let mut o = Map::new();
    o.insert("instance".into(), Value::String(inst.to_string()));
    o.insert("owed".into(), Value::from(rows.len()));
    o.insert("owed_landed".into(), Value::from(landed));
    o.insert("rows".into(), Value::from(out));
    Ok(Value::Object(o))
}

/// 2026-10-23: A section that cannot be computed, as a declared mode (PCND): the others are
/// still returned.
fn unavailable(e: &metrale_error::Error) -> Value {
    let mut o = Map::new();
    o.insert("unavailable".into(), Value::Bool(true));
    o.insert("code".into(), Value::String(e.code().name().to_string()));
    o.insert("reason".into(), Value::String(e.to_string()));
    Value::Object(o)
}

/// 2026-10-23: The requested sections (all when `names` is empty), keyed by name. A section
/// that cannot be computed is returned as `{unavailable, code, reason}`, never failing the rest;
/// an unknown campaign or section name is refused.
pub fn status(
    b: &Bundle,
    cfg: &Config,
    campaign: &str,
    names: &[String],
    fresh: &Freshness,
    instance: Option<&str>,
) -> Result<Value> {
    let short = names.is_empty();
    let all: Vec<String> = if short {
        DEFAULT_SECTIONS.iter().map(|s| (*s).to_string()).collect()
    } else {
        names.to_vec()
    };
    let instance = instance.or(cfg.campaign(campaign)?.default_instance.as_deref());
    if let Some(bad) = all.iter().find(|n| !SECTIONS.contains(&n.as_str())) {
        return Err(error!(
            ErrorCode::KgToolArgument,
            "sections",
            format!("no section `{bad}`; sections: {}", SECTIONS.join(", "))
        ));
    }
    let mut o = Map::new();
    for n in &all {
        let v = section(b, cfg, campaign, n, fresh, instance).unwrap_or_else(|e| unavailable(&e));
        o.insert(n.clone(), if short { compact(v) } else { v });
    }
    Ok(Value::Object(o))
}
