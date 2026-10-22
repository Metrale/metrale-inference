// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: [`FsRepo`]: metrale-circuit's source port ([`Repo`]) over a checkout on disk, so
//! the circuit core reads nothing itself (SBIO). `met circuit` chooses it; the circuit's own
//! tests keep a fixture tree.
//!
//! Owner: metrale-io-host-fs-repo.
//! Invariants:
//! - Paths in and out are repo-relative and `/`-separated; a listing is sorted.
//! - Every failure is `ErrorCode::CircuitSource` `(path, the OS reason)`.
#![forbid(unsafe_code)]

use std::path::PathBuf;

use metrale_circuit::venn::Repo;
use metrale_error::{Error, ErrorCode, error};

/// 2026-10-22: The repository rooted at `root`.
#[derive(Clone, Debug)]
pub struct FsRepo {
    /// 2026-10-22: The checkout's root directory.
    pub root: PathBuf,
}

fn source(what: String, e: &std::io::Error) -> Error {
    error!(ErrorCode::CircuitSource, what, e.to_string())
}

impl Repo for FsRepo {
    fn read(&self, rel: &str) -> metrale_error::Result<String> {
        std::fs::read_to_string(self.root.join(rel)).map_err(|e| source(rel.to_string(), &e))
    }

    fn list(&self, rel: &str) -> metrale_error::Result<Vec<String>> {
        let mut out = Vec::new();
        let mut stack = vec![self.root.join(rel)];
        while let Some(dir) = stack.pop() {
            let entries =
                std::fs::read_dir(&dir).map_err(|e| source(dir.display().to_string(), &e))?;
            for entry in entries {
                let path = entry
                    .map_err(|e| source(dir.display().to_string(), &e))?
                    .path();
                if path.is_dir() {
                    stack.push(path);
                } else if let Ok(r) = path.strip_prefix(&self.root) {
                    out.push(r.to_string_lossy().replace('\\', "/"));
                }
            }
        }
        out.sort();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::FsRepo;
    use metrale_circuit::venn::Repo;
    use metrale_error::ErrorCode;

    fn repo() -> (tempfile::TempDir, FsRepo) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("kernels/b/c")).unwrap();
        std::fs::write(dir.path().join("kernels/b/c/z.toml"), "z").unwrap();
        std::fs::write(dir.path().join("kernels/a.toml"), "a = 1\n").unwrap();
        let root = dir.path().to_path_buf();
        (dir, FsRepo { root })
    }

    #[test]
    fn reads_and_lists_repo_relative_sorted_paths() {
        let (_d, r) = repo();
        assert_eq!(r.read("kernels/a.toml").unwrap(), "a = 1\n");
        assert_eq!(
            r.list("kernels").unwrap(),
            ["kernels/a.toml", "kernels/b/c/z.toml"]
        );
    }

    #[test]
    fn a_missing_file_is_a_circuit_source_error_naming_the_path() {
        let (_d, r) = repo();
        let e = r.read("kernels/none.toml").unwrap_err();
        assert_eq!(e.code(), ErrorCode::CircuitSource);
        assert!(e.to_string().starts_with("kernels/none.toml: "), "{e}");
        assert_eq!(
            r.list("nowhere").unwrap_err().code(),
            ErrorCode::CircuitSource
        );
    }
}
