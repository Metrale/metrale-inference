// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the KB on this machine behind the `KbOpen` port: the hashed
//! embedder, the system lease, the LatticeDB sink and the sealed export, as
//! `kb.toml` configures them. `metprof kb selfcheck|replay|export|seal|verify`
//! land here.

use std::fs;
use std::path::Path;

use serde_json::{Map, Value, json};

use crate::config::{EmbedderConfig, EmbeddingConfig};
use crate::derive::{derive, run_key};
use crate::embed::Hashed;
use crate::export::EXPORT_FILE;
use crate::kb_toml::{KbConfig, KbPaths, SealConfig};
use crate::lease::{Holder, SystemLeaseEnv};
use crate::port::{Kb, KbOpen, KbReport, KbSink, Recorded, RunRecord};
use crate::schema::{Kind, MAX_SAMPLE_DEPTH, NodeRef};
use crate::seal::ExportSeal;
use crate::sink::{LatticeSink, Materialized, Sealing, block_on, busy_reason, open_durable};
use crate::spool::Spool;
use crate::{Error, Result, bail, canon, export};

/// Opens the KB of a `kb.toml`; `publisher_build` names the sealing build.
pub struct LocalKbs {
    publisher_build: String,
}

impl LocalKbs {
    pub fn new(publisher_build: &str) -> LocalKbs {
        LocalKbs {
            publisher_build: publisher_build.to_string(),
        }
    }
}

impl KbOpen for LocalKbs {
    fn open<'a>(&'a self, kb: &KbConfig, embedding: &EmbeddingConfig) -> Result<Box<dyn Kb + 'a>> {
        if embedding.embedder != EmbedderConfig::Hashed {
            bail!(
                "kb: metprof embeds with [embedder] kind = \"hashed\" only ({}); a dense \
                 embedder's client lives outside metprof",
                kb.embedding.display()
            );
        }
        let sealing = match &kb.seal {
            SealConfig::Off => Sealing::Off,
            SealConfig::KeyFile(p) => Sealing::Sealed(ExportSeal::load(p, &self.publisher_build)?),
        };
        Ok(Box::new(LocalKb {
            paths: kb.paths.clone(),
            sealing,
            embedder: Hashed::new(),
            env: SystemLeaseEnv,
        }))
    }
}

struct LocalKb {
    paths: KbPaths,
    sealing: Sealing,
    embedder: Hashed,
    env: SystemLeaseEnv,
}

fn path(p: &Path) -> Value {
    json!(p.display().to_string())
}

fn busy(result: Map<String, Value>, h: Option<Holder>) -> KbReport {
    KbReport {
        result,
        incomplete: Some(busy_reason(h)),
    }
}

fn materialized(m: Materialized) -> KbReport {
    match m {
        Materialized::Done {
            applied,
            export,
            sealed,
        } => {
            let mut r = Map::new();
            r.insert("inserted".into(), json!(applied.inserted));
            r.insert("updated".into(), json!(applied.updated));
            r.insert("unchanged".into(), json!(applied.unchanged));
            r.insert("edges_added".into(), json!(applied.edges_added));
            r.insert("edges_updated".into(), json!(applied.edges_updated));
            r.insert("export".into(), path(&export));
            r.insert(
                "sealed".into(),
                sealed.as_deref().map_or(json!("off"), path),
            );
            KbReport {
                result: r,
                incomplete: None,
            }
        }
        Materialized::Busy(h) => busy(Map::new(), h),
    }
}

impl LocalKb {
    fn sink(&self) -> LatticeSink<'_> {
        LatticeSink::new(self.paths.clone(), &self.embedder, &self.env, &self.sealing)
    }

    fn key(&self, what: &str) -> Result<&ExportSeal> {
        match &self.sealing {
            Sealing::Sealed(s) => Ok(s),
            Sealing::Off => bail!("kb {what}: kb.toml has seal = \"off\""),
        }
    }

    /// Under the lease: the sentinel's point is in the durable store with
    /// its digest, the export is the store's export, and the sealed export
    /// (unless sealing is off) opens to it.
    fn read_back(&self, sentinel: &RunRecord) -> Result<()> {
        let node = NodeRef::new(Kind::ProfileRun, run_key(sentinel)?);
        let graph = derive(sentinel, MAX_SAMPLE_DEPTH)?;
        let Some(want) = graph.records.iter().find(|r| r.node() == node) else {
            bail!("selfcheck: the sentinel derives no profile run");
        };
        let want = canon::content_sha256(want)?;
        let store = block_on(open_durable(&self.paths.durable_dir, &self.embedder))??;
        match store.get(&node)? {
            Some(s) if s.content_sha256 == want => {}
            Some(_) => bail!("selfcheck: the stored sentinel differs from the one recorded"),
            None => bail!("selfcheck: the sentinel is not in the durable store"),
        }
        let file = self.paths.export_dir.join(EXPORT_FILE);
        let on_disk = fs::read(&file)
            .map_err(|e| Error(format!("selfcheck: reading {}: {e}", file.display())))?;
        if on_disk != export::render(&store)? {
            bail!("selfcheck: {} is not the store's export", file.display());
        }
        if let Sealing::Sealed(s) = &self.sealing {
            s.verify(&self.paths.export_dir)?;
        }
        Ok(())
    }
}

impl KbSink for LocalKb {
    fn record(&self, run: &RunRecord) -> Result<Recorded> {
        self.sink().record(run)
    }
}

impl Kb for LocalKb {
    fn selfcheck(&self, sentinel: &RunRecord) -> Result<KbReport> {
        let rec = self.record(sentinel)?;
        let mut r = Map::new();
        r.insert("spool".into(), json!(rec.spool_path));
        if let Some(reason) = rec.reason {
            return Ok(KbReport {
                result: r,
                incomplete: Some(reason),
            });
        }
        if Spool::read(Path::new(&rec.spool_path))? != *sentinel {
            bail!("selfcheck: the spool file does not read back as the sentinel");
        }
        if let Err(h) = self.sink().leased(|| self.read_back(sentinel))? {
            return Ok(busy(r, h));
        }
        let export_dir = &self.paths.export_dir;
        r.insert("export".into(), path(&export_dir.join(EXPORT_FILE)));
        r.insert(
            "sealed".into(),
            match &self.sealing {
                Sealing::Off => json!("off"),
                Sealing::Sealed(_) => path(&export_dir.join(crate::seal::SEALED_FILE)),
            },
        );
        r.insert("read_back".into(), json!("ok"));
        Ok(KbReport {
            result: r,
            incomplete: None,
        })
    }

    fn replay(&self) -> Result<KbReport> {
        let sink = self.sink();
        let files = sink.spool().list()?.len();
        let mut report = materialized(sink.replay()?);
        report.result.insert("spool_files".into(), json!(files));
        Ok(report)
    }

    fn export(&self) -> Result<KbReport> {
        Ok(materialized(self.sink().materialize(&[])?))
    }

    fn seal(&self) -> Result<KbReport> {
        let key = self.key("seal")?;
        let mut r = Map::new();
        r.insert("kid".into(), json!(key.kid()));
        match self.sink().leased(|| key.seal(&self.paths.export_dir))? {
            Ok(sealed) => {
                r.insert("sealed".into(), path(&sealed));
                Ok(KbReport {
                    result: r,
                    incomplete: None,
                })
            }
            Err(h) => Ok(busy(r, h)),
        }
    }

    fn verify(&self) -> Result<KbReport> {
        let key = self.key("verify")?;
        let mut r = Map::new();
        r.insert("kid".into(), json!(key.kid()));
        match self.sink().leased(|| key.verify(&self.paths.export_dir))? {
            Ok(n) => {
                r.insert("verified_bytes".into(), json!(n));
                Ok(KbReport {
                    result: r,
                    incomplete: None,
                })
            }
            Err(h) => Ok(busy(r, h)),
        }
    }
}
