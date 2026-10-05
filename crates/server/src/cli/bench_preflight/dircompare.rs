// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: I/O adapter for the non-vacuous-comparison check: count the regular files
//! under a directory, recursively, so a byte-identity claim over an output tree (nested
//! shard/run directories, not a flat dir) is counted honestly.
//!
//! Owner: server CLI (`met bench preflight`).
//! Invariants:
//! - Counts regular files only (`FileType::is_file`), not directories or symlinks, so an
//!   empty directory full of empty subdirectories still counts as zero.
//! - An unreadable directory is an error, not a zero count: a FAIL from a missing/
//!   unreadable path must read as "could not check", not as "empty, therefore vacuous" —
//!   both fail the gate, but `core::evaluate` reports the real cause.

use std::path::Path;

/// 2026-10-05: The number of regular files under `dir`, walked recursively.
pub fn count_files(dir: &Path) -> Result<usize, String> {
    if !dir.exists() {
        return Err(format!("{} does not exist", dir.display()));
    }
    if !dir.is_dir() {
        return Err(format!("{} is not a directory", dir.display()));
    }
    let mut total = 0usize;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let entries = std::fs::read_dir(&d).map_err(|e| format!("{}: {e}", d.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("{}: {e}", d.display()))?;
            let file_type = entry
                .file_type()
                .map_err(|e| format!("{}: {e}", entry.path().display()))?;
            if file_type.is_dir() {
                stack.push(entry.path());
            } else if file_type.is_file() {
                total += 1;
            }
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "metrale-preflight-dircompare-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// 2026-10-05: Path A — an empty directory counts as zero, not an error: `core` is
    /// where "zero is a refusal" is decided, not here.
    #[test]
    fn an_empty_directory_counts_as_zero() {
        let dir = tmp("empty");
        assert_eq!(count_files(&dir).unwrap(), 0);
    }

    /// 2026-10-05: Path B — files nested under subdirectories are counted, so a run's
    /// sharded output tree is not undercounted as empty.
    #[test]
    fn nested_files_are_counted_recursively() {
        let dir = tmp("nested");
        std::fs::create_dir_all(dir.join("a/b")).unwrap();
        std::fs::write(dir.join("top.json"), "{}").unwrap();
        std::fs::write(dir.join("a/one.json"), "{}").unwrap();
        std::fs::write(dir.join("a/b/two.json"), "{}").unwrap();
        assert_eq!(count_files(&dir).unwrap(), 3);
    }

    /// 2026-10-05: Path C — a nonexistent path is an error, not a silent zero: the report
    /// must say "does not exist", not "empty".
    #[test]
    fn a_nonexistent_path_is_an_error() {
        let err = count_files(&std::env::temp_dir().join("metrale-preflight-does-not-exist-9f3c"))
            .unwrap_err();
        assert!(err.contains("does not exist"), "{err}");
    }

    /// 2026-10-05: A path that exists but is a file, not a directory, is also an error.
    #[test]
    fn a_file_path_is_not_a_directory_error() {
        let dir = tmp("not-a-dir");
        let file = dir.join("f.txt");
        std::fs::write(&file, "x").unwrap();
        let err = count_files(&file).unwrap_err();
        assert!(err.contains("not a directory"), "{err}");
    }
}
