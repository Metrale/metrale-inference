// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `kg_search` (design §3 tool 2, P2): bring the embedding cache up to date with the
//! bundle's texts (only new shas are sent), embed the query, and rank with
//! `metrale_kg_derive::search`. Any embedder failure answers in `graph_only` with that failure as
//! the reason, for this call only; the declared mode is unchanged.
//!
//! Owner: metrale-kg-mcp.
//! Invariants:
//! - Every result names its mode and, for `graph_only`, the reason.
//! - The query text is sent to the embedder only in `hybrid` mode and is never logged or cached.

use crate::app::App;
use crate::textstore::TextStore;
use metrale_error::{ErrorCode, Result, error};
use metrale_io_host_kg::openrouter::OpenRouter;
use metrale_io_host_std::{SystemClock, TokioTimer};
use metrale_kg_derive::embed::{Embedder, Mode, Plan, Role, cosine, embed_all, missing};
use metrale_kg_derive::hash::Sha256;
use metrale_kg_derive::id::{KgId, Label};
use metrale_kg_derive::search::{lexical, named, rank};
use metrale_kg_derive::tools::{DEFAULT_SEARCH_HOPS, DEFAULT_SEARCH_K};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// 2026-10-23: The embedder as the binary chose it.
pub enum Embed {
    Off(String),
    On {
        host: OpenRouter,
        cache: TextStore,
        plan: Plan,
        /// 2026-10-23: Texts the provider refused one by one, with why; not re-sent by this
        /// process (a restart retries them).
        refused: BTreeMap<Sha256, String>,
    },
}

/// 2026-10-23: Cache vectors for every bundle text the cache lacks; returns how many were sent.
pub async fn sync(app: &mut App) -> Result<usize> {
    let texts: Vec<_> = app
        .bundle
        .texts
        .iter()
        .map(|t| (t.sha, t.text.clone()))
        .collect();
    let Embed::On {
        host,
        cache,
        plan,
        refused,
    } = &mut app.embed
    else {
        return Ok(0);
    };
    let mut skip = cache
        .cached()
        .map_err(|e| error!(ErrorCode::KgStore, "kg_text", e.to_string()))?;
    skip.extend(refused.keys().copied());
    let todo = missing(texts.iter().map(|(s, t)| (*s, t.as_str())), &skip);
    let got = embed_all(
        &*host,
        &SystemClock,
        &TokioTimer,
        &*plan,
        Role::Document,
        &todo,
    )
    .await;
    let n = got.done.len();
    refused.extend(got.refused);
    cache
        .put(got.done)
        .await
        .map_err(|e| error!(ErrorCode::KgStore, "kg_text", e.to_string()))?;
    match got.stopped {
        Some(e) => Err(e),
        None => Ok(n),
    }
}

async fn query_vector(host: &OpenRouter, text: &str, plan: &Plan) -> Result<Vec<f32>> {
    let todo = [(metrale_kg_derive::hash::record_sha(text), text.to_string())];
    let mut got = embed_all(host, &SystemClock, &TokioTimer, plan, Role::Query, &todo).await;
    match (got.done.pop(), got.stopped, got.refused.pop()) {
        (_, Some(e), _) => Err(e),
        (Some((_, v)), None, _) => Ok(v),
        (None, None, Some((_, why))) => Err(error!(
            ErrorCode::KgEmbedProtocol,
            format!("the provider refused the query: {why}")
        )),
        (None, None, None) => Err(error!(
            ErrorCode::KgEmbedProtocol,
            "the query got no vector"
        )),
    }
}

/// 2026-10-23: Per-node cosine similarity of `q` to the node's cached texts.
fn cosines(app: &App, cache: &TextStore, q: &[f32]) -> Result<BTreeMap<KgId, f64>> {
    let vecs = cache
        .vectors()
        .map_err(|e| error!(ErrorCode::KgStore, "kg_text", e.to_string()))?;
    let mut out: BTreeMap<KgId, f64> = BTreeMap::new();
    for t in &app.bundle.texts {
        let Some(v) = vecs.get(&t.sha) else { continue };
        let s = cosine(q, v);
        let e = out.entry(t.node.clone()).or_insert(f64::MIN);
        if s > *e {
            *e = s;
        }
    }
    Ok(out)
}

fn arg_range(a: &Value, k: &str, default: u32, lo: u32, hi: u32) -> Result<u32> {
    match a.get(k) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .filter(|n| (lo..=hi).contains(n))
            .ok_or_else(|| {
                error!(
                    ErrorCode::KgToolArgument,
                    k.to_string(),
                    format!("an integer in {lo}..={hi}")
                )
            }),
    }
}

/// 2026-10-23: The tool.
pub async fn kg_search(app: &mut App, args: &Value) -> Result<Value> {
    let text = args
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| error!(ErrorCode::KgToolArgument, "text", "required string"))?
        .to_string();
    let mut kinds = Vec::new();
    for k in args
        .get("kinds")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let name = k.as_str().unwrap_or("");
        kinds.push(Label::from_name(name).ok_or_else(|| {
            error!(
                ErrorCode::KgToolArgument,
                "kinds",
                format!("no label {name}")
            )
        })?);
    }
    let k = arg_range(args, "k", DEFAULT_SEARCH_K, 1, 50)?;
    let hops = arg_range(args, "hops", DEFAULT_SEARCH_HOPS, 0, 2)?;
    let mut sims = lexical(&app.bundle, &text);
    let (mode, reason) = match &app.embed {
        Embed::Off(r) => ("graph_only", Some(r.clone())),
        Embed::On { host, .. } => match host.mode() {
            Mode::Unavailable { reason } => ("graph_only", Some(reason)),
            Mode::Available { .. } => match hybrid(app, &text).await {
                Ok(cos) => {
                    for (id, c) in cos {
                        let e = sims.entry(id).or_insert(0.0);
                        if c > *e {
                            *e = c;
                        }
                    }
                    ("hybrid", None)
                }
                Err(e) => ("graph_only", Some(format!("{}: {e}", e.code().name()))),
            },
        },
    };
    let names = named(&app.bundle, &text);
    let hits = rank(
        &app.bundle,
        &sims,
        &names,
        &kinds,
        usize::try_from(k).unwrap_or(10),
        hops,
        &app.cfg.search,
    );
    let hits: Vec<Value> = hits
        .into_iter()
        .map(|h| {
            let mut o = Map::new();
            o.insert("id".into(), Value::String(h.id.to_string()));
            o.insert(
                "kind".into(),
                Value::String(h.id.label().name().to_string()),
            );
            o.insert(
                "score".into(),
                Value::from((h.score * 1000.0).round() / 1000.0),
            );
            o.insert("snippet".into(), Value::String(h.snippet));
            o.insert("src".into(), Value::String(h.src));
            o.insert("via".into(), Value::from(h.via));
            Value::Object(o)
        })
        .collect();
    let mut o = Map::new();
    o.insert("mode".into(), Value::String(mode.to_string()));
    if let Some(r) = reason {
        o.insert("mode_reason".into(), Value::String(r));
    }
    o.insert("hits".into(), Value::from(hits));
    if let Embed::On { cache, refused, .. } = &app.embed {
        let shas: std::collections::BTreeSet<Sha256> =
            app.bundle.texts.iter().map(|t| t.sha).collect();
        let cached = cache
            .cached()
            .map(|c| c.intersection(&shas).count())
            .unwrap_or(0);
        let mut c = Map::new();
        c.insert("texts".into(), Value::from(shas.len()));
        c.insert("cached".into(), Value::from(cached));
        c.insert("refused".into(), Value::from(refused.len()));
        if let Some((_, why)) = refused.iter().next() {
            c.insert("first_refusal".into(), Value::String(why.clone()));
        }
        o.insert("cache".into(), Value::Object(c));
    }
    Ok(Value::Object(o))
}

async fn hybrid(app: &mut App, text: &str) -> Result<BTreeMap<KgId, f64>> {
    sync(app).await?;
    let Embed::On {
        host, cache, plan, ..
    } = &app.embed
    else {
        return Err(error!(ErrorCode::KgEmbedderUnavailable, "no embedder"));
    };
    let q = query_vector(host, text, plan).await?;
    cosines(app, cache, &q)
}
