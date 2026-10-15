// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the spool (DESIGN §3.1 step 1): one file per run,
//! `<recorded_at>-<content_sha256[..12]>.kkb.jsonl`, holding the run record's
//! canonical JSON and a newline. It is written before anything else and only
//! ever by an atomic rename, so a spool file is always whole; everything the
//! store holds is re-derived from it (`metprof kb replay`).

use std::fs;
use std::path::{Path, PathBuf};

use crate::atomic::{self, STAGED_PREFIX, Staged};
use crate::port::RunRecord;
use crate::{Error, Result, bail, canon};

pub const SPOOL_SUFFIX: &str = ".kkb.jsonl";

/// The spool file name of `run`. `recorded_at` becomes part of a path, so
/// anything but an RFC 3339 character is refused.
pub fn file_name(run: &RunRecord) -> Result<String> {
    let at = &run.stamps.recorded_at;
    let rfc3339 = |c: char| c.is_ascii_alphanumeric() || matches!(c, ':' | '.' | '+' | '-');
    if at.is_empty() || !at.chars().all(rfc3339) {
        bail!("spool: recorded_at {at:?} is not an RFC 3339 time");
    }
    let sha = run.content_sha256()?;
    Ok(format!("{at}-{}{SPOOL_SUFFIX}", &sha[..12]))
}

/// The bytes of a spool file: canonical JSON and one newline.
pub fn encode(run: &RunRecord) -> Result<Vec<u8>> {
    run.check()?;
    let mut b = canon::canonical(run)?;
    b.push(b'\n');
    Ok(b)
}

/// A run record from spool bytes. Only exactly what `encode` writes is
/// accepted, so a truncated or edited file fails instead of half-loading.
pub fn decode(bytes: &[u8]) -> Result<RunRecord> {
    let Some(body) = bytes.strip_suffix(b"\n") else {
        bail!("spool: a record is one line ending in a newline");
    };
    let run: RunRecord =
        serde_json::from_slice(body).map_err(|e| Error(format!("spool: record: {e}")))?;
    if encode(&run)? != bytes {
        bail!("spool: the record is not in canonical form");
    }
    Ok(run)
}

/// A spool directory.
#[derive(Debug, Clone)]
pub struct Spool {
    dir: PathBuf,
}

impl Spool {
    pub fn new(dir: impl Into<PathBuf>) -> Spool {
        Spool { dir: dir.into() }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Write the record under a temporary name; it is not in the spool until
    /// the returned value is committed.
    pub fn stage(&self, run: &RunRecord) -> Result<Staged> {
        atomic::stage(&self.dir, &file_name(run)?, &encode(run)?)
    }

    /// Spool `run`; the path of its spool file.
    pub fn write(&self, run: &RunRecord) -> Result<PathBuf> {
        self.stage(run)?.commit()
    }

    /// The spool files, in name order (staged temporaries are not files of
    /// the spool).
    pub fn list(&self) -> Result<Vec<PathBuf>> {
        let entries = fs::read_dir(&self.dir)
            .map_err(|e| Error(format!("spool: reading {}: {e}", self.dir.display())))?;
        let mut out = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| Error(format!("spool: {e}")))?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !name.starts_with(STAGED_PREFIX) && name.ends_with(SPOOL_SUFFIX) {
                out.push(entry.path());
            }
        }
        out.sort();
        Ok(out)
    }

    /// Read one spool file.
    pub fn read(path: &Path) -> Result<RunRecord> {
        let bytes =
            fs::read(path).map_err(|e| Error(format!("spool: reading {}: {e}", path.display())))?;
        decode(&bytes).map_err(|e| Error(format!("{}: {e}", path.display())))
    }
}
