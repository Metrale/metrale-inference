// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The running graph: config, the chosen hosts, the current bundle and the store, with
//! the per-call freshness check (design §4.1 invariant 5): stat every source; on any change re-read,
//! re-derive and reload; if that fails, keep the last bundle and say `stale` with the reason.
//!
//! Owner: metrale-kg-mcp.
//! Invariants:
//! - Every tool result carries `bundle_digest`, `derived_at_ms`, `stale` and `mode`.
//! - The store only ever holds the current bundle (`Store::load` replaces).

use crate::search::Embed;
use crate::store::{Store, data_dir};
use crate::textstore::TextStore;
use anyhow::Result;
use metrale_abstract_io::Clock;
use metrale_io_host_kg::fs::{FsSources, Stamp, read_config};
use metrale_io_host_kg::openrouter::OpenRouter;
use metrale_io_host_kg::settings::{Resolved, resolve_settings};
use metrale_io_host_std::SystemClock;
use metrale_kg_derive::embed::{Embedder as _, Mode, Plan};
use metrale_kg_derive::jsonl::digest;
use metrale_kg_derive::lkb_derive::LkbSources;
use metrale_kg_derive::settings::Embedder;
use metrale_kg_derive::status::Freshness;
use metrale_kg_derive::{Bundle, Config, SourceSet, derive};
use serde_json::{Map, Value};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;

/// 2026-10-23: The application state behind the CLI and the MCP server.
pub struct App {
    pub cfg: Config,
    pub host: FsSources,
    pub bundle: Bundle,
    pub digest: String,
    pub stamp: Stamp,
    pub fresh: Freshness,
    pub store: Arc<Mutex<Store>>,
    /// 2026-10-23: The embedder settings, resolved once at open (ROTI).
    pub settings: Resolved,
    /// 2026-10-23: The embedder host the settings chose, with its cache (or the declared
    /// unavailable mode and why). The mode every result reports comes from here.
    pub embed: Embed,
    clock: SystemClock,
}

/// 2026-10-23: The most recently modified source file in a stamp and its mtime (unix ms); the
/// LKB checkout's entry (`<repo>@git`, a hash, not a time) is not a file.
fn newest(stamp: &Stamp) -> Option<(String, i64)> {
    stamp
        .iter()
        .filter(|(p, _, _)| !p.ends_with("@git"))
        .max_by_key(|(_, _, ns)| *ns)
        .map(|(p, _, ns)| {
            (
                p.clone(),
                i64::try_from(*ns / 1_000_000).unwrap_or(i64::MAX),
            )
        })
}

/// 2026-10-23: The embedder host for the resolved settings: off with the reason when the settings
/// declare it unavailable or the host cannot run (no key), else the host and its cache under
/// `dir`.
async fn open_embed(e: &Embedder, dir: &Path) -> Result<Embed> {
    let s = match e {
        Embedder::Unavailable { reason } => return Ok(Embed::Off(reason.clone())),
        Embedder::Available(s) => s,
    };
    let host = OpenRouter::new(s, dir.join("embed-calls.jsonl"));
    if let Mode::Unavailable { reason } = host.mode() {
        return Ok(Embed::Off(reason));
    }
    let cache = TextStore::open(dir, &s.space(), s.dimensions).await?;
    let plan = Plan {
        batch: usize::try_from(s.batch)?,
        max_attempts: s.max_attempts,
        backoff_ms: s.backoff_ms,
        max_input_chars: usize::try_from(s.max_input_chars)?,
    };
    Ok(Embed::On {
        host,
        cache,
        plan,
        refused: Default::default(),
    })
}

/// 2026-10-23: Read and derive once (no store); also the LKB checkout's commit, if any.
pub fn derive_once(cfg: &Config, host: &FsSources) -> Result<(SourceSet, Bundle, Option<String>)> {
    let src = host.read(cfg)?;
    let facts = match &cfg.sources.lkb {
        Some(t) => Some(metrale_io_host_kg::lkb::facts_cached(
            Path::new(&cfg.sources.root),
            &LkbSources::parse(t)?,
            &cfg.validator.python,
            &Path::new(&cfg.store.scratch_dir).join("lkb-facts"),
        )?),
        None => None,
    };
    let b = derive(cfg, &src, facts.as_ref())?;
    Ok((src, b, facts.map(|f| f.head)))
}

impl App {
    /// 2026-10-23: Load the config, derive, open the store at `data_dir` (or the config's) and load.
    pub async fn open(cfg_path: &Path, data_override: Option<&Path>) -> Result<App> {
        let cfg = read_config(cfg_path)?;
        let settings = resolve_settings(&cfg.embedder)?;
        let host = FsSources::new(&cfg);
        let stamp = host.stamp(&cfg)?;
        let (_, bundle, source_commit) = derive_once(&cfg, &host)?;
        let dir = match data_override {
            Some(d) => d.to_path_buf(),
            None => data_dir(&cfg.store.data_dir)?,
        };
        let mut store = Store::open(&dir).await?;
        store.load(&bundle).await?;
        let embed = open_embed(&settings.embedder, &dir).await?;
        let clock = SystemClock;
        let fresh = Freshness {
            stale_reason: None,
            derived_at_ms: clock.now_unix_ms().0,
            source_commit,
            newest_change: newest(&stamp),
        };
        Ok(App {
            digest: digest(&bundle).hex(),
            cfg,
            host,
            bundle,
            stamp,
            fresh,
            store: Arc::new(Mutex::new(store)),
            settings,
            embed,
            clock,
        })
    }

    /// 2026-10-23: The per-call freshness check; returns whether the bundle changed.
    pub async fn refresh(&mut self) -> bool {
        let stamp = match self.host.stamp(&self.cfg) {
            Ok(s) => s,
            Err(e) => {
                self.fresh.stale_reason = Some(format!("{}: {e}", e.code().name()));
                return false;
            }
        };
        if stamp == self.stamp && self.fresh.stale_reason.is_none() {
            return false;
        }
        match derive_once(&self.cfg, &self.host) {
            Ok((_, b, source_commit)) => {
                let changed = digest(&b).hex() != self.digest;
                if let Err(e) = self.store.lock().await.load(&b).await {
                    self.fresh.stale_reason = Some(format!("KgStore: {e}"));
                    return false;
                }
                self.digest = digest(&b).hex();
                self.bundle = b;
                self.fresh = Freshness {
                    stale_reason: None,
                    derived_at_ms: self.clock.now_unix_ms().0,
                    source_commit,
                    newest_change: newest(&stamp),
                };
                self.stamp = stamp;
                changed
            }
            Err(e) => {
                self.fresh.stale_reason = Some(format!("KgStaleBundle: {e}"));
                false
            }
        }
    }

    /// 2026-10-23: Declare the embedder off for this process (`--graph-only`): the same declared
    /// mode as an unavailable embedder, with `why` as the reason.
    pub fn declare_graph_only(&mut self, why: &str) {
        self.embed = Embed::Off(why.to_string());
    }

    /// 2026-10-23: The embedder's mode as the host reports it (a key reference that yields no
    /// key is `graph_only` even though the settings ask for http), as every result reports it.
    pub fn mode(&self) -> (String, Option<String>) {
        match &self.embed {
            Embed::Off(reason) => ("graph_only".into(), Some(reason.clone())),
            Embed::On { .. } => ("hybrid".into(), None),
        }
    }

    /// 2026-10-23: Wrap a tool result in the freshness envelope.
    pub fn envelope(&self, body: Value) -> Value {
        let mut o = match body {
            Value::Object(o) => o,
            other => {
                let mut m = Map::new();
                m.insert("result".into(), other);
                m
            }
        };
        o.insert("bundle_digest".into(), Value::String(self.digest.clone()));
        o.insert(
            "derived_at_ms".into(),
            Value::from(self.fresh.derived_at_ms),
        );
        o.insert(
            "stale".into(),
            Value::Bool(self.fresh.stale_reason.is_some()),
        );
        if let Some(c) = &self.fresh.source_commit {
            o.insert("source_commit".into(), Value::String(c.clone()));
        }
        if let Some(r) = &self.fresh.stale_reason {
            o.insert("stale_reason".into(), Value::String(r.clone()));
        }
        if !o.contains_key("mode") {
            let (mode, reason) = self.mode();
            o.insert("mode".into(), Value::String(mode));
            if let Some(r) = reason {
                o.insert("mode_reason".into(), Value::String(r));
            }
        }
        Value::Object(o)
    }
}
