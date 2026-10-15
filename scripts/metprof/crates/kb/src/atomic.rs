// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: whole-file replacement in two steps. `stage` writes the bytes
//! to a hidden temporary name in the destination directory and fsyncs them;
//! `commit` renames it onto the final name (atomic on POSIX) and fsyncs the
//! directory. A crash between the two leaves only the hidden temporary file:
//! a reader of the final name sees the old file or the new one, never a part.

use std::fs::{self, File};
use std::io::{ErrorKind, Write};
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

/// Write `bytes` beside `dir/name` under a hidden temporary name. `dir` must
/// exist (`create_dir` makes it): a directory that vanished is not remade.
pub fn stage(dir: &Path, name: &str, bytes: &[u8]) -> Result<Staged> {
    if name.is_empty() || name.starts_with(STAGED_PREFIX) || name.contains(['/', '\\']) {
        return Err(Error(format!(
            "atomic write: {name:?} is not a plain file name"
        )));
    }
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

/// Create `dir` and its missing ancestors, syncing the parent of each
/// directory created (a new entry is durable only once its parent is synced).
pub fn create_dir(dir: &Path) -> Result<()> {
    let mut missing = Vec::new();
    let mut at = Some(dir);
    while let Some(d) = at.filter(|d| !d.as_os_str().is_empty()) {
        match fs::metadata(d) {
            Ok(m) if m.is_dir() => break,
            Ok(_) => return Err(Error(format!("creating {}: not a directory", d.display()))),
            Err(e) if e.kind() == ErrorKind::NotFound => missing.push(d),
            Err(e) => return Err(io("reading", d, e)),
        }
        at = d.parent();
    }
    for d in missing.into_iter().rev() {
        match fs::create_dir(d) {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::AlreadyExists && d.is_dir() => continue,
            Err(e) => return Err(io("creating", d, e)),
        }
        match d.parent().filter(|p| !p.as_os_str().is_empty()) {
            Some(p) => sync_dir(p)?,
            None => sync_dir(Path::new("."))?,
        }
    }
    Ok(())
}

/// Replace `dir/name` with `bytes`.
pub fn write(dir: &Path, name: &str, bytes: &[u8]) -> Result<PathBuf> {
    stage(dir, name, bytes)?.commit()
}
