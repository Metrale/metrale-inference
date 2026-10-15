// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: what metprof hands the KB sink (DESIGN §3.1), as pure types,
//! and the two ports the CLI gains: the sink and the clock. A `RunRecord` is
//! what the spool keeps, one per run; everything the KB derives from a run is
//! re-derived from it.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::config::EmbeddingConfig;
use crate::kb_toml::KbConfig;
use crate::{Result, bail, canon};

/// The schema of a spooled run record.
pub const RUN_SCHEMA: &str = "metrale.kkb/1";
/// The value metprof writes for a fact it cannot know.
pub const UNKNOWN: &str = "unknown";

/// One argv word. A file argument is never kept as a path: only its content
/// digest and its base name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Arg {
    Word(String),
    File {
        sha256: String,
        basename: String,
    },
    /// A directory argument: only its base name (its content is not one file).
    Dir {
        basename: String,
    },
}

/// A file the command wrote: its base name, digest, and for a JSON file
/// (the kernel tree) its parsed content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileRef {
    pub name: String,
    pub sha256: String,
    pub json: Option<Value>,
}

/// Who recorded the run, from `kb.toml` (explicit, never guessed).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Identity {
    pub box_alias: String,
    pub campaign: String,
    pub hardware_class: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stamps {
    pub recorded_at: String,
    pub metprof_version: String,
    pub metprof_git_sha: String,
    /// The family table's digest, or `unknown` for a command without one.
    pub config_sha256: String,
    pub capture_sha256: Option<String>,
    pub identity: Identity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunRecord {
    pub schema: String,
    pub command: String,
    pub args: Vec<Arg>,
    pub result: Map<String, Value>,
    pub files: Vec<FileRef>,
    pub stamps: Stamps,
}

impl RunRecord {
    pub fn check(&self) -> Result<()> {
        if self.schema != RUN_SCHEMA {
            bail!("run record: schema {:?} is not {RUN_SCHEMA:?}", self.schema);
        }
        if self.command.is_empty() {
            bail!("run record: empty command");
        }
        Ok(())
    }

    /// SHA-256 of the record's canonical JSON (names its spool file).
    pub fn content_sha256(&self) -> Result<String> {
        canon::digest(self)
    }
}

/// The outcome of a sink write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recorded {
    pub spool_path: String,
    /// True when the run also reached the durable store and the export.
    pub materialized: bool,
    /// Why it did not (None exactly when `materialized`).
    pub reason: Option<String>,
}

/// The KB sink port. A run is spooled before anything else; an error means
/// it was not spooled (the caller exits 3 either way when not materialised).
pub trait KbSink {
    fn record(&self, run: &RunRecord) -> Result<Recorded>;
}

/// The injected clock: the current time, RFC 3339 UTC.
pub trait Clock {
    fn now_rfc3339(&self) -> Result<String>;
}

/// What a `metprof kb` subcommand reports. `incomplete` says why the KB is
/// not in the state asked for (the run was spooled but not materialised, or
/// another writer held the lease); the CLI exits 3 with it.
#[derive(Debug, Clone, PartialEq)]
pub struct KbReport {
    pub result: Map<String, Value>,
    pub incomplete: Option<String>,
}

/// What `kb verify` found when it could look (an `Err` around it means it
/// could not: no key, or the lease or a read failed).
#[derive(Debug, Clone, PartialEq)]
pub enum Verify {
    /// The sealed export verified (`incomplete` when another writer held
    /// the lease, and nothing was looked at).
    Report(KbReport),
    /// The sealed export is missing, does not open (tampered with, or
    /// sealed under another key) or does not seal the current export.
    Failed(String),
}

/// One configured KB: the sink and the maintenance the CLI offers.
pub trait Kb: KbSink {
    /// The whole write path with `sentinel`, then read it back from the
    /// spool, the durable store, the export and the sealed export.
    fn selfcheck(&self, sentinel: &RunRecord) -> Result<KbReport>;
    /// Materialise every spool file (idempotent), export and seal.
    fn replay(&self) -> Result<KbReport>;
    /// Export (and seal) the durable store as it is.
    fn export(&self) -> Result<KbReport>;
    /// Seal the current export again.
    fn seal(&self) -> Result<KbReport>;
    /// Open the sealed export and require it to equal the export.
    fn verify(&self) -> Result<Verify>;
}

/// The port that opens the KB a `kb.toml` (and its `embedding.toml`) names.
/// The configuration comes from argv, so the CLI holds this, not a sink.
pub trait KbOpen {
    fn open<'a>(&'a self, kb: &KbConfig, embedding: &EmbeddingConfig) -> Result<Box<dyn Kb + 'a>>;
}
