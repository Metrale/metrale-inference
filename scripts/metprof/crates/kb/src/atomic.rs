// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: whole-file replacement in two steps. `stage` writes the bytes
//! to a hidden temporary name in the destination directory and fsyncs them;
//! `commit` renames it onto the final name (atomic on POSIX) and fsyncs the
//! directory. A crash between the two leaves only the hidden temporary file:
//! a reader of the final name sees the old file or the new one, never a part.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::{Error, Result};

/// The prefix every staged temporary file carries (readers skip it).
pub const STAGED_PREFIX: char = '.';

fn io(what: &str, path: &Path, e: std::io::Error) -> Error {
    Error(format!("{what} {}: {e}", path.display()))
}

/// Bytes written and fsynced under a temporary name, not yet visible.
#[derive(Debug)]
#[must_use = "a staged file is invisible until it is committed"]
pub struct Staged {
    tmp: PathBuf,
    dest: PathBuf,
}

/// Write `bytes` beside `dir/name` under a hidden temporary name; the
/// directory is created when missing.
pub fn stage(dir: &Path, name: &str, bytes: &[u8]) -> Result<Staged> {
    if name.is_empty() || name.starts_with(STAGED_PREFIX) || name.contains(['/', '\\']) {
        return Err(Error(format!(
            "atomic write: {name:?} is not a plain file name"
        )));
    }
    fs::create_dir_all(dir).map_err(|e| io("creating", dir, e))?;
    let tmp = dir.join(format!("{STAGED_PREFIX}{name}.tmp-{}", std::process::id()));
    let mut f = File::create(&tmp).map_err(|e| io("creating", &tmp, e))?;
    f.write_all(bytes).map_err(|e| io("writing", &tmp, e))?;
    f.sync_all().map_err(|e| io("syncing", &tmp, e))?;
    Ok(Staged {
        tmp,
        dest: dir.join(name),
    })
}

impl Staged {
    /// Make the file visible under its final name.
    pub fn commit(self) -> Result<PathBuf> {
        fs::rename(&self.tmp, &self.dest).map_err(|e| io("renaming", &self.tmp, e))?;
        if let Some(dir) = self.dest.parent() {
            sync_dir(dir)?;
        }
        Ok(self.dest)
    }
}

/// Persist a directory's entries (a rename is durable only once its
/// directory is synced). Done on Linux, where `fsync` of a directory is
/// defined; elsewhere the rename's durability is the file system's.
pub fn sync_dir(dir: &Path) -> Result<()> {
    if cfg!(target_os = "linux") {
        File::open(dir)
            .and_then(|d| d.sync_all())
            .map_err(|e| io("syncing", dir, e))?;
    }
    Ok(())
}

/// Replace `dir/name` with `bytes`.
pub fn write(dir: &Path, name: &str, bytes: &[u8]) -> Result<PathBuf> {
    stage(dir, name, bytes)?.commit()
}
