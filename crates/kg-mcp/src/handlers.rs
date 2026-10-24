// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The five tools (design §3), JSON in, JSON out, shared by the MCP server and the CLI
//! so both run the same code. Argument defaults are the declared ones of `metrale_kg_derive::tools`.
//!
//! Owner: metrale-kg-mcp.
//! Invariants:
//! - Every call runs the freshness check first and returns the envelope (`App::envelope`).
//! - An argument outside its schema is `KgToolArgument`, never coerced.

use crate::app::App;
use crate::cypher;
use metrale_error::{ErrorCode, Result, error};
use metrale_kg_derive::id::{Label, Rel};
use metrale_kg_derive::ledger::{self, render_deps};
use metrale_kg_derive::queries::{self, impact};
use metrale_kg_derive::status::status;
use metrale_kg_derive::tools::{DEFAULT_MAX_DEPTH, DEFAULT_QUERY_LIMIT};
use serde_json::{Map, Value};
use std::time::Duration;

fn arg_str<'a>(a: &'a Value, k: &str) -> Result<&'a str> {
    a.get(k)
        .and_then(Value::as_str)
        .ok_or_else(|| error!(ErrorCode::KgToolArgument, k.to_string(), "required string"))
}

fn arg_u32(a: &Value, k: &str, default: u32, max: u32) -> Result<u32> {
    match a.get(k) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .filter(|n| (1..=max).contains(n))
            .ok_or_else(|| {
                error!(
                    ErrorCode::KgToolArgument,
                    k.to_string(),
                    format!("an integer in 1..={max}")
                )
            }),
    }
}

fn arg_strs(a: &Value, k: &str) -> Result<Vec<String>> {
    match a.get(k) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(v)) => v
            .iter()
            .map(|x| {
                x.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| error!(ErrorCode::KgToolArgument, k.to_string(), "strings only"))
            })
            .collect(),
        Some(_) => Err(error!(
            ErrorCode::KgToolArgument,
            k.to_string(),
            "must be an array"
        )),
    }
}

/// 2026-10-23: Dispatch one tool call.
pub async fn call(app: &mut App, tool: &str, args: &Value) -> Result<Value> {
    app.refresh().await;
    let body = match tool {
        "kg_query" => kg_query(app, args).await?,
        "kg_impact" => kg_impact(app, args)?,
        "kg_status" => {
            let campaign = arg_str(args, "campaign")?;
            let sections = arg_strs(args, "sections")?;
            let instance = args.get("instance").and_then(Value::as_str);
            let mut v = status(
                &app.bundle,
                &app.cfg,
                campaign,
                &sections,
                &app.fresh,
                instance,
            )?;
            if sections.is_empty() || sections.iter().any(|s| s == "freshness") {
                let prov: Map<String, Value> = app
                    .settings
                    .provenance
                    .iter()
                    .map(|(k, l)| ((*k).to_string(), Value::String(l.name().into())))
                    .collect();
                let mut st = Map::new();
                st.insert(
                    "global_layer".into(),
                    Value::String(app.settings.global.describe()),
                );
                st.insert(
                    "project_layer".into(),
                    Value::String(app.cfg.origin.clone()),
                );
                st.insert("provenance".into(), Value::Object(prov));
                if let Value::Object(o) = &mut v {
                    o.insert("settings".into(), Value::Object(st));
                }
            }
            v
        }
        "kg_search" => crate::search::kg_search(app, args).await?,
        "kg_record" => crate::record::kg_record(app, args).await?,
        other => {
            return Err(error!(
                ErrorCode::KgToolArgument,
                "name",
                format!("no tool {other}")
            ));
        }
    };
    Ok(app.envelope(body))
}

async fn kg_query(app: &mut App, args: &Value) -> Result<Value> {
    let text = arg_str(args, "cypher")?.to_string();
    cypher::read_only(&text)?;
    let cap = app.cfg.limits.max_rows.min(1000);
    let limit = arg_u32(args, "limit", DEFAULT_QUERY_LIMIT.min(cap), cap)?;
    let params = cypher::params(args.get("params"))?;
    let store = app.store.clone();
    let ms = app.cfg.limits.cypher_timeout_ms;
    let job = tokio::task::spawn_blocking(move || store.blocking_lock().query(&text, params));
    let (cols, mut rows) = match tokio::time::timeout(Duration::from_millis(ms), job).await {
        Err(_) => {
            return Err(error!(
                ErrorCode::KgCypherRejected,
                format!("exceeded cypher_timeout_ms {ms}")
            ));
        }
        Ok(Err(e)) => return Err(error!(ErrorCode::KgStore, "query", e.to_string())),
        Ok(Ok(Err(e))) => return Err(error!(ErrorCode::KgCypherRejected, e.to_string())),
        Ok(Ok(Ok(r))) => r,
    };
    let limit = usize::try_from(limit).unwrap_or(usize::MAX);
    let truncated = rows.len() > limit;
    rows.truncate(limit);
    let mut o = Map::new();
    o.insert("columns".into(), Value::from(cols));
    o.insert("rows".into(), Value::from(rows));
    o.insert("truncated".into(), Value::Bool(truncated));
    Ok(Value::Object(o))
}

/// 2026-10-23: The default relations of `kg_impact` for a root of `label`.
pub fn default_rels(label: Label, downstream: bool) -> Vec<Rel> {
    match (label, downstream) {
        (Label::LedgerEntry, false) => vec![Rel::DependsOn],
        (Label::LedgerEntry, true) => {
            vec![Rel::DependsOn, Rel::Pins, Rel::MeasuredIn, Rel::DecidedBy]
        }
        _ => Rel::ALL.to_vec(),
    }
}

fn kg_impact(app: &App, args: &Value) -> Result<Value> {
    let root = queries::resolve_id(&app.bundle, arg_str(args, "id")?)?;
    let downstream = match arg_str(args, "direction")? {
        "downstream" => true,
        "upstream" => false,
        other => {
            return Err(error!(
                ErrorCode::KgToolArgument,
                "direction",
                format!("{other} is not downstream|upstream")
            ));
        }
    };
    let named = arg_strs(args, "relations")?;
    let rels: Vec<Rel> = if named.is_empty() {
        default_rels(root.label(), downstream)
    } else {
        named
            .iter()
            .map(|n| {
                Rel::from_name(n).ok_or_else(|| {
                    error!(
                        ErrorCode::KgToolArgument,
                        "relations",
                        format!("no relation {n}")
                    )
                })
            })
            .collect::<Result<_>>()?
    };
    let depth = arg_u32(args, "max_depth", DEFAULT_MAX_DEPTH, 100)?;
    let (reach, cut) = impact(&app.bundle, &root, downstream, &rels, depth)?;
    let b = &app.bundle;
    let closure: Vec<Value> = reach
        .iter()
        .map(|r| {
            let mut o = Map::new();
            o.insert("id".into(), Value::String(r.id.to_string()));
            o.insert("why".into(), Value::String(r.why.clone()));
            o.insert("depth".into(), Value::from(r.depth));
            if let Some(v) = b.field(&r.id, "verdict_kind") {
                o.insert("verdict".into(), Value::String(v.to_string()));
            }
            Value::Object(o)
        })
        .collect();
    let mut o = Map::new();
    o.insert("root".into(), Value::String(root.to_string()));
    o.insert("closure".into(), Value::from(closure));
    o.insert("paths_truncated".into(), Value::Bool(cut));
    if root.label() == Label::LedgerEntry {
        let campaign = b.field(&root, "campaign").unwrap_or("").to_string();
        let c = app.cfg.campaign(&campaign)?;
        queries::ledger_valid(b, &c.ledger)?;
        let links = queries::ledger_links(b, &campaign);
        let chain = ledger::deps(&links, root.key());
        let mut upstream: Vec<&str> = chain.iter().map(String::as_str).collect();
        upstream.push(root.key());
        let tainted: Vec<Value> = ledger::tainted(&links)
            .into_iter()
            .filter(|(id, _)| upstream.contains(&id.as_str()))
            .map(|(id, why)| Value::from(vec![id, why]))
            .collect();
        o.insert("tainted_by".into(), Value::from(tainted));
        o.insert(
            "ledger_deps".into(),
            Value::String(render_deps(&chain).trim_end().to_string()),
        );
    }
    Ok(Value::Object(o))
}
