// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The running graph: config, the chosen hosts, the current bundle and the store, with
//! the per-call freshness check (design §4.1 invariant 5): stat every source; on any change re-read,
//! re-derive and reload; if that fails, keep the last bundle and say `stale` with the reason.
//!
//! Owner: metrale-kg-mcp.
//! Invariants:
//! - Every tool result carries `bundle_digest`, `derived_at_ms`, `stale` and `mode`.
//! - The store only ever holds the current bundle (`Store::load` replaces).

use crate::store::{Store, data_dir};
use anyhow::Result;
use metrale_abstract_io::Clock;
use metrale_io_host_kg::fs::{FsSources, Stamp, read_config};
use metrale_io_host_kg::settings::{Resolved, resolve_settings};
use metrale_io_host_std::SystemClock;
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
    clock: SystemClock,
}

/// 2026-10-23: Read and derive once (no store); also the LKB checkout's commit, if any.
pub fn derive_once(cfg: &Config, host: &FsSources) -> Result<(SourceSet, Bundle, Option<String>)> {
    let src = host.read(cfg)?;
    let facts = match &cfg.sources.lkb {
        Some(t) => Some(metrale_io_host_kg::lkb::facts(
            Path::new(&cfg.sources.root),
            &LkbSources::parse(t)?,
            &cfg.validator.python,
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
        let clock = SystemClock;
        let fresh = Freshness {
            stale_reason: None,
            derived_at_ms: clock.now_unix_ms().0,
            source_commit,
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
                self.stamp = stamp;
                self.fresh = Freshness {
                    stale_reason: None,
                    derived_at_ms: self.clock.now_unix_ms().0,
                    source_commit,
                };
                changed
            }
            Err(e) => {
                self.fresh.stale_reason = Some(format!("KgStaleBundle: {e}"));
                false
            }
        }
    }

    /// 2026-10-23: The embedder's declared mode, as every result reports it.
    pub fn mode(&self) -> (String, Option<String>) {
        match &self.settings.embedder {
            Embedder::Unavailable { reason } => ("graph_only".into(), Some(reason.clone())),
            Embedder::Available(_) => ("hybrid".into(), None),
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
