// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The I/O side of the kernel → model map: a working tree or a git revision read
//! through `metrale_circuit::venn::Repo`, and [`load`], which never fails (a map that does not
//! load is [`KernelMap::Unavailable`], which reaches every model).
//!
//! Owner: bench gate.
//! Invariants:
//! - [`FsRepo`] is the one filesystem `Repo`; `met circuit venn` reads through it too.
//! - [`GitRepo`] reads blobs at one revision and never touches the working tree.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use metrale_circuit::venn::Repo;
use metrale_circuit::venn::affected_repo::instance_uses;

use super::model_scope::KernelMap;
use super::owns::{Owners, from_bench_toml};

/// 2026-09-29: The repository on disk (moved here from `met circuit venn` 2026-10-15).
pub struct FsRepo {
    pub root: PathBuf,
}

impl Repo for FsRepo {
    fn read(&self, rel: &str) -> Result<String, String> {
        std::fs::read_to_string(self.root.join(rel)).map_err(|e| format!("{rel}: {e}"))
    }

    fn list(&self, rel: &str) -> Result<Vec<String>, String> {
        let mut out = Vec::new();
        let mut stack = vec![self.root.join(rel)];
        while let Some(dir) = stack.pop() {
            let entries = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            for entry in entries {
                let path = entry.map_err(|e| e.to_string())?.path();
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

/// 2026-10-15: The repository at one git revision.
pub struct GitRepo {
    pub root: PathBuf,
    pub rev: String,
}

impl GitRepo {
    fn git(&self, args: &[&str]) -> Result<String, String> {
        let out = Command::new("git")
            .arg("-C")
            .arg(&self.root)
            .args(args)
            .stdin(Stdio::null())
            .output()
            .map_err(|e| format!("git: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "git {}: {}",
                args.join(" "),
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        String::from_utf8(out.stdout).map_err(|e| format!("git {}: {e}", args.join(" ")))
    }
}

impl Repo for GitRepo {
    fn read(&self, rel: &str) -> Result<String, String> {
        self.git(&["show", &format!("{}:{rel}", self.rev)])
    }

    fn list(&self, rel: &str) -> Result<Vec<String>, String> {
        let text = self.git(&["ls-tree", "-r", "--name-only", &self.rev, "--", rel])?;
        Ok(text.lines().map(str::to_string).collect())
    }
}

/// 2026-10-15: The paths `head` changed since it and `base` last shared a commit (`base...head`).
pub fn changed_paths(root: &Path, base: &str, head: &str) -> Result<Vec<String>, String> {
    let repo = GitRepo {
        root: root.to_path_buf(),
        rev: head.to_string(),
    };
    let text = repo.git(&["diff", "--name-only", &format!("{base}...{head}")])?;
    Ok(text.lines().map(str::to_string).collect())
}

/// 2026-10-15: The commit `head` and `base` last shared, so a change is what `head` adds and
/// not what `base` gained since.
pub fn merge_base(root: &Path, base: &str, head: &str) -> Result<String, String> {
    let repo = GitRepo {
        root: root.to_path_buf(),
        rev: head.to_string(),
    };
    let sha = repo.git(&["merge-base", base, head])?.trim().to_string();
    if sha.is_empty() {
        return Err(format!("git merge-base {base} {head}: no common commit"));
    }
    Ok(sha)
}

/// 2026-10-15: The kernel → model map of the tree `repo` reads.
pub fn load(repo: &dyn Repo) -> KernelMap {
    match instance_uses(repo) {
        Ok((manifests, uses)) => KernelMap::Derived { manifests, uses },
        Err(e) => KernelMap::Unavailable(e.to_string()),
    }
}

/// 2026-10-16: Every `owns` list of the tree `repo` reads, by model directory, from the
/// BENCH.toml files `bench::load_all` reads (`kernels/<hw>/<model>/BENCH.toml` under a hardware
/// directory holding `HARDWARE.toml`). Never fails: a tree that does not list owns nothing,
/// and a file that does not read or parse owns nothing, so its paths reach every model.
pub fn owners(repo: &dyn Repo) -> Owners {
    let Ok(files) = repo.list("kernels") else {
        return Owners::new();
    };
    let mut out = Owners::new();
    for path in &files {
        let parts: Vec<&str> = path.split('/').collect();
        let [_, hw, model, "BENCH.toml"] = parts[..] else {
            continue;
        };
        if !files.contains(&format!("kernels/{hw}/HARDWARE.toml")) {
            continue;
        }
        let Ok(owns) = repo.read(path).and_then(|t| from_bench_toml(&t)) else {
            continue;
        };
        out.entry(model.to_string()).or_default().extend(owns);
    }
    out.retain(|_, entries| !entries.is_empty());
    out
}

#[cfg(test)]
#[path = "kernel_map_tests.rs"]
mod kernel_map_tests;
