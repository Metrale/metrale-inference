// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the KB sink metprof writes through (DESIGN §3.1): spool the
//! run, take the lease, materialise it into the durable collection, export,
//! release. Only the spool write can fail the call; anything after it leaves
//! the run recorded and not materialised, with the reason, for `replay`.

use std::future::Future;
use std::path::{Path, PathBuf};

use lattice_storage::DiskStorage;

use crate::derive::derive;
use crate::embed::Embedder;
use crate::lattice::{Applied, Store};
use crate::lease::{self, Acquired, Holder, LeaseEnv};
use crate::port::{KbSink, Recorded, RunRecord};
use crate::schema::MAX_SAMPLE_DEPTH;
use crate::seal::ExportSeal;
use crate::spool::Spool;
use crate::{Error, Result, export};

/// The page size of both durable storages.
pub const PAGE_SIZE: usize = 4096;
pub const WAL_DIR: &str = "wal";
pub const DATA_DIR: &str = "data";

pub use crate::kb_toml::KbPaths;

/// Whether each export is also sealed (explicit: `kb.toml` says which).
#[derive(Debug)]
pub enum Sealing {
    Off,
    Sealed(ExportSeal),
}

/// The outcome of one materialisation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Materialized {
    Done {
        applied: Applied,
        export: PathBuf,
        /// The sealed export, unless sealing is off.
        sealed: Option<PathBuf>,
    },
    /// The durable directory stayed leased by another writer.
    Busy(Option<Holder>),
}

/// Run a future to completion on a current-thread runtime.
pub fn block_on<F: Future>(f: F) -> Result<F::Output> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .map_err(|e| Error(format!("kb: runtime: {e}")))?;
    Ok(rt.block_on(f))
}

async fn disk(dir: PathBuf) -> Result<DiskStorage> {
    let s = DiskStorage::new(dir.clone(), PAGE_SIZE);
    s.init()
        .await
        .map_err(|e| Error(format!("kb: storage {}: {e}", dir.display())))?;
    Ok(s)
}

/// Open the durable collection of `embedder`'s space under `durable_dir`.
/// The caller holds the lease.
pub async fn open_durable(
    durable_dir: &Path,
    embedder: &dyn Embedder,
) -> Result<Store<DiskStorage, DiskStorage>> {
    let wal = disk(durable_dir.join(WAL_DIR)).await?;
    let data = disk(durable_dir.join(DATA_DIR)).await?;
    Store::open(embedder.space(), wal, data).await
}

pub struct LatticeSink<'a> {
    paths: KbPaths,
    embedder: &'a dyn Embedder,
    env: &'a dyn LeaseEnv,
    sealing: &'a Sealing,
}

/// Why a leased step did not run.
pub fn busy_reason(h: Option<Holder>) -> String {
    let who = match h {
        Some(h) => format!("pid {} since {}", h.pid, h.since),
        None => "a writer that has not yet stated itself".to_string(),
    };
    format!("the durable directory is leased by {who}; `metprof kb replay` catches up")
}

impl<'a> LatticeSink<'a> {
    pub fn new(
        paths: KbPaths,
        embedder: &'a dyn Embedder,
        env: &'a dyn LeaseEnv,
        sealing: &'a Sealing,
    ) -> Self {
        LatticeSink {
            paths,
            embedder,
            env,
            sealing,
        }
    }

    pub fn sealing(&self) -> &Sealing {
        self.sealing
    }

    /// Run `f` under the lease; `Err(holder)` when another writer kept it.
    pub fn leased<T>(
        &self,
        f: impl FnOnce() -> Result<T>,
    ) -> Result<std::result::Result<T, Option<Holder>>> {
        let lease = match lease::acquire(&self.paths.durable_dir, self.env)? {
            Acquired::Held(l) => l,
            Acquired::Busy(h) => return Ok(Err(h)),
        };
        let out = f();
        let released = lease.release();
        let out = out?;
        released?;
        Ok(Ok(out))
    }

    pub fn paths(&self) -> &KbPaths {
        &self.paths
    }

    pub fn spool(&self) -> Spool {
        Spool::new(&self.paths.spool_dir)
    }

    /// Under the lease: derive and apply `runs` in order, then export (with
    /// no runs, only export).
    pub fn materialize(&self, runs: &[RunRecord]) -> Result<Materialized> {
        Ok(self
            .leased(|| block_on(self.load(runs)).and_then(|r| r))?
            .unwrap_or_else(Materialized::Busy))
    }

    async fn load(&self, runs: &[RunRecord]) -> Result<Materialized> {
        let mut store = open_durable(&self.paths.durable_dir, self.embedder).await?;
        let mut applied = Applied::default();
        for run in runs {
            let graph = derive(run, MAX_SAMPLE_DEPTH)?;
            applied.add(store.apply(&graph, self.embedder).await?);
        }
        let export = export::write(&store, &self.paths.export_dir)?;
        let sealed = match self.sealing {
            Sealing::Off => None,
            Sealing::Sealed(s) => Some(s.seal(&self.paths.export_dir)?),
        };
        Ok(Materialized::Done {
            applied,
            export,
            sealed,
        })
    }

    /// Materialise every spool file, in name order (idempotent).
    pub fn replay(&self) -> Result<Materialized> {
        let runs = self
            .spool()
            .list()?
            .iter()
            .map(|p| Spool::read(p))
            .collect::<Result<Vec<_>>>()?;
        self.materialize(&runs)
    }
}

impl KbSink for LatticeSink<'_> {
    fn record(&self, run: &RunRecord) -> Result<Recorded> {
        let spool_path = self.spool().write(run)?.display().to_string();
        let reason = match self.materialize(std::slice::from_ref(run)) {
            Ok(Materialized::Done { .. }) => None,
            Ok(Materialized::Busy(h)) => Some(busy_reason(h)),
            Err(e) => Some(e.0),
        };
        Ok(Recorded {
            spool_path,
            materialized: reason.is_none(),
            reason,
        })
    }
}
