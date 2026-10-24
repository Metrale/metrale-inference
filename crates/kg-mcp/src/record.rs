// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `kg_record` (design §3 tool 4, §4.1 invariant 4, P3): parse the arguments, run
//! `metrale_kg_derive::record_flow::record` over the native file and validator hosts, re-derive
//! after a write, answer with the taint preview.
//!
//! Owner: metrale-kg-mcp.
//! Invariants:
//! - The flow itself (checks, preview, fail-closed write) lives in the core; this is wiring only.

use crate::app::App;
use metrale_error::{ErrorCode, Result, error};
use metrale_io_host_kg::ledger_py::LedgerPy;
use metrale_io_host_kg::record::{FsRecord, LedgerValidator};
use metrale_kg_derive::record::Kind;
use metrale_kg_derive::record_flow::{Request, record};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// 2026-10-23: The tool.
pub async fn kg_record(app: &mut App, args: &Value) -> Result<Value> {
    let kind = Kind::parse(args.get("kind").and_then(Value::as_str).unwrap_or(""))?;
    let campaign = args.get("campaign").and_then(Value::as_str).unwrap_or("");
    let c = app.cfg.campaign(campaign)?.clone();
    let dry_run = args
        .get("dry_run")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let acknowledge_taint = match args.get("acknowledge_taint") {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            v.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| error!(ErrorCode::KgToolArgument, "acknowledge_taint", "a count"))?,
        ),
    };
    let mut fields = BTreeMap::new();
    for (k, v) in args
        .get("fields")
        .and_then(Value::as_object)
        .ok_or_else(|| error!(ErrorCode::KgToolArgument, "fields", "must be an object"))?
    {
        let s = v
            .as_str()
            .ok_or_else(|| error!(ErrorCode::KgToolArgument, k.clone(), "must be a string"))?;
        fields.insert(k.clone(), s.to_string());
    }
    let root = PathBuf::from(&app.cfg.sources.root);
    let fs = FsRecord { root: root.clone() };
    let validator = LedgerValidator {
        ledger_py: LedgerPy {
            python: PathBuf::from(&app.cfg.validator.python),
            script: root.join(&c.ledger_check),
            scratch: PathBuf::from(&app.cfg.store.scratch_dir),
        },
    };
    let req = Request {
        kind,
        fields,
        dry_run,
        acknowledge_taint,
    };
    let out = record(&fs, &validator, &c, &req).await?;
    if out.written {
        app.refresh().await;
    }
    let mut o = Map::new();
    o.insert("id".into(), Value::String(out.id));
    o.insert("file".into(), Value::String(out.file));
    o.insert("line".into(), Value::from(out.line));
    o.insert("validator".into(), Value::String(out.validator));
    if let Some(cl) = out.claims {
        o.insert("claims".into(), Value::String(cl));
    }
    o.insert("would_taint".into(), Value::from(out.would_taint));
    o.insert("dry_run".into(), Value::Bool(dry_run));
    o.insert("written".into(), Value::Bool(out.written));
    o.insert("warnings".into(), Value::from(out.warnings));
    Ok(Value::Object(o))
}
