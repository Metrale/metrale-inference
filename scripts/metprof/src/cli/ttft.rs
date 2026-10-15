// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `ttft log`, `ttft compare`, `ttft client`: the one TTFT
//! definition over a rank-0 serve log, two logs, and a live endpoint.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::cli::args::{Args, Kind, Opt, Spec};
use crate::cli::{Command, Done, Io, Run, load};
use crate::core::error::Result;
use crate::core::tables::into_map;
use crate::core::ttft::cells::{Selection, cell_rows, compare, select, summary};
use crate::core::ttft::client::{
    ClientRecord, Intent, join, parse_records, parse_url, record, records_jsonl, render_prompt,
    request, salt_text,
};
use crate::core::ttft::log::{ServeLog, parse_serve_log};
use crate::core::ttft::sse::decode_stream;
use crate::core::ttft::{TtftConfig, ttft_config};
use crate::fail;

const CONFIG: Opt = Opt {
    name: "config",
    kind: Kind::Str,
    required: true,
    metavar: "TOML",
    help: "ttft config: measured max_tokens, step token id, cells (and [client])",
};

const fn path(name: &'static str, required: bool, help: &'static str) -> Opt {
    Opt {
        name,
        kind: Kind::Str,
        required,
        metavar: "FILE",
        help,
    }
}

const fn int(name: &'static str, help: &'static str) -> Opt {
    Opt {
        name,
        kind: Kind::Int,
        required: true,
        metavar: "N",
        help,
    }
}

pub const COMMANDS: &[&Command] = &[
    &Command {
        spec: Spec {
            name: "ttft log",
            about: "per-request and per-cell server TTFT (cold/warm by cache-hit marker, first-token class)",
            positionals: &[],
            opts: &[
                path("log", true, "the rank-0 serve log"),
                CONFIG,
                path(
                    "client",
                    false,
                    "client JSONL of the same run (`ttft client --out`), joined in order",
                ),
            ],
        },
        run: Run::Report(log),
    },
    &Command {
        spec: Spec {
            name: "ttft compare",
            about: "two serve logs per cell: median delta (B - A), ranges, range-separation verdict",
            positionals: &[],
            opts: &[
                path("a", true, "rank-0 serve log, control"),
                path("b", true, "rank-0 serve log, treatment"),
                CONFIG,
                int("min-reps", "reps each side needs for a verdict"),
            ],
        },
        run: Run::Report(compare_logs),
    },
    &Command {
        spec: Spec {
            name: "ttft client",
            about: "time streamed requests of one cell: warm-ups, fresh-salt cold reps, warm re-sends",
            positionals: &[],
            opts: &[
                Opt {
                    name: "url",
                    kind: Kind::Str,
                    required: true,
                    metavar: "http://HOST:PORT",
                    help: "the serve endpoint",
                },
                CONFIG,
                Opt {
                    name: "cell",
                    kind: Kind::Str,
                    required: true,
                    metavar: "NAME",
                    help: "the config cell the prompt belongs to",
                },
                path("prompt", true, "prompt text with {salt} exactly once"),
                int("cold", "cold reps (a fresh salt each); at least 1"),
                int("warm", "warm reps (the last cold prompt again)"),
                path("out", true, "client JSONL to write"),
            ],
        },
        run: Run::Files(client),
    },
];

fn config(io: &Io, path: &str) -> Result<TtftConfig> {
    ttft_config(&load::read_toml(io, path)?).map_err(|e| e.context(path))
}

fn serve_log(io: &Io, path: &str) -> Result<ServeLog> {
    let bytes = load::read_bytes(io, path)?;
    parse_serve_log(&String::from_utf8_lossy(&bytes)).map_err(|e| e.context(path))
}

fn counts(log: &ServeLog, sel: &Selection) -> Value {
    json!({
        "requests": log.requests.len(),
        "concurrent": log.concurrent.len(),
        "unfinished": log.unfinished.len(),
        "measured": sel.measured.len(),
        "unassigned": sel.unassigned,
        "other_max_tokens": sel.other,
    })
}

fn log(a: &Args, io: &Io) -> Result<Map<String, Value>> {
    let cfg = config(io, a.str("config")?)?;
    let path = a.str("log")?;
    let parsed = serve_log(io, path)?;
    let sel = select(&parsed, &cfg).map_err(|e| e.context(path))?;
    let records: Vec<ClientRecord> = match a.opt_str("client") {
        Some(p) => parse_records(&String::from_utf8_lossy(&load::read_bytes(io, p)?))
            .map_err(|e| e.context(p))?,
        None => Vec::new(),
    };
    let pairs = join(&records, &sel)?;
    let by_measured: HashMap<usize, &ClientRecord> =
        pairs.iter().map(|(m, r)| (*m, &records[*r])).collect();
    let rows: Vec<Value> = sel
        .measured
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let mut row = json!({
                "line": m.req.line,
                "time": m.req.time,
                "cell": m.cell,
                "mode": m.mode.name(),
                "class": m.class.name(),
                "first_token": m.first_token,
                "ttft_server_ms": m.req.ttft_ms,
            });
            if let Some(r) = by_measured.get(&i) {
                row["ttft_client_ms"] = json!(r.ttft_client_ms);
                row["first_delta"] = json!(r.first_delta);
            }
            row
        })
        .collect();
    let mut out = into_map(json!({
        "counts": counts(&parsed, &sel),
        "server": cell_rows(&sel, &cfg, |m| Some(m.req.ttft_ms)),
    }));
    if a.opt_str("client").is_some() {
        let line_of: HashMap<usize, f64> = by_measured
            .iter()
            .map(|(i, r)| (sel.measured[*i].req.line, r.ttft_client_ms))
            .collect();
        out.insert("joined".to_string(), json!(pairs.len()));
        out.insert(
            "client".to_string(),
            json!(cell_rows(&sel, &cfg, |m| line_of.get(&m.req.line).copied())),
        );
    }
    out.insert("requests".to_string(), json!(rows));
    Ok(out)
}

fn compare_logs(a: &Args, io: &Io) -> Result<Map<String, Value>> {
    let cfg = config(io, a.str("config")?)?;
    let min_reps = a.int("min-reps")?;
    if min_reps < 1 {
        fail!("--min-reps must be at least 1, got {min_reps}");
    }
    let mut sides = Vec::new();
    for key in ["a", "b"] {
        let path = a.str(key)?;
        let parsed = serve_log(io, path)?;
        let sel = select(&parsed, &cfg).map_err(|e| e.context(path))?;
        sides.push((
            counts(&parsed, &sel),
            cell_rows(&sel, &cfg, |m| Some(m.req.ttft_ms)),
        ));
    }
    let rows = compare(&sides[0].1, &sides[1].1, min_reps as usize)?;
    Ok(into_map(json!({
        "min_reps": min_reps,
        "counts_a": sides[0].0,
        "counts_b": sides[1].0,
        "cells": rows,
    })))
}

fn client(a: &Args, io: &Io) -> Result<Done> {
    let cfg = config(io, a.str("config")?)?;
    let c = cfg.client()?;
    let cell = cfg.cell(a.str("cell")?)?;
    let (cold, warm) = (a.int("cold")?, a.int("warm")?);
    if cold < 1 || warm < 0 {
        fail!("--cold must be at least 1 and --warm at least 0, got {cold} and {warm}");
    }
    let ep = parse_url(a.str("url")?)?;
    let prompt_path = a.str("prompt")?;
    let template = match String::from_utf8(load::read_bytes(io, prompt_path)?) {
        Ok(t) => t,
        Err(_) => fail!("{prompt_path}: not UTF-8"),
    };
    render_prompt(&template, "").map_err(|e| e.context(prompt_path))?;
    let mut recs: Vec<ClientRecord> = Vec::new();
    let mut rep = |intent: Intent, n: u64, salt: &str| -> Result<String> {
        let max_tokens = match intent {
            Intent::Warmup => c.warmup_max_tokens,
            Intent::Cold | Intent::Warm => cfg.max_tokens,
        };
        let req = request(&ep, c, max_tokens, &render_prompt(&template, salt)?);
        let what = format!("cell {} {} rep {n}", cell.name, intent.name());
        let reads = io
            .chat
            .exchange(&ep, &req, c.timeout_s)
            .map_err(|e| e.context(&what))?;
        let streamed = decode_stream(&reads).map_err(|e| e.context(&what))?;
        recs.push(record(cell, n, intent, salt, &streamed)?);
        Ok(salt.to_string())
    };
    for n in 0..c.warmup_requests {
        rep(
            Intent::Warmup,
            n,
            &salt_text(io.salt.fresh(), c.salt_digits),
        )?;
    }
    let mut last = String::new();
    for n in 0..cold as u64 {
        last = rep(Intent::Cold, n, &salt_text(io.salt.fresh(), c.salt_digits))?;
    }
    for n in 0..warm as u64 {
        rep(Intent::Warm, n, &last)?;
    }
    let mode = |i: Intent| -> Vec<f64> {
        recs.iter()
            .filter(|r| r.intent == i)
            .map(|r| r.ttft_client_ms)
            .collect()
    };
    let reps: Vec<Value> = recs
        .iter()
        .map(|r| {
            json!({
                "mode": r.intent.name(),
                "rep": r.rep,
                "salt": r.salt,
                "ttft_client_ms": r.ttft_client_ms,
                "usage_ttft_ms": r.usage_ttft_ms,
                "first_delta": r.first_delta,
                "completion_tokens": r.completion_tokens,
            })
        })
        .collect();
    let result = into_map(json!({
        "cell": cell.name,
        "prompt_tokens": cell.prompt_tokens,
        "cold": summary(&mode(Intent::Cold)),
        "warm": summary(&mode(Intent::Warm)),
        "reps": reps,
    }));
    Ok(Done {
        result,
        files: vec![(a.str("out")?.to_string(), records_jsonl(&recs).into_bytes())],
    })
}
