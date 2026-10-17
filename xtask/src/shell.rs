// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The I/O shell: finds the repository root and reads the manifests a workspace
//! names. Everything it returns is a value the rules judge without touching the disk.

use crate::manifest::{self, Manifest, Workspace};
use std::path::{Path, PathBuf};

/// 2026-10-17: The repository root: the directory above this crate.
pub fn repo_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "xtask has no parent directory".to_owned())
}

/// 2026-10-17: This crate's fixture trees, one workspace per directory.
pub fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

/// 2026-10-17: The workspace rooted at `root`: its `Cargo.toml` and each member's.
pub fn load_workspace(root: &Path) -> Result<Workspace, String> {
    let root_manifest = load(root, "Cargo.toml")?;
    let members = manifest::member_dirs(&root_manifest)?
        .iter()
        .map(|dir| load(root, &format!("{dir}/Cargo.toml")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Workspace {
        root: root_manifest,
        members,
    })
}

fn load(root: &Path, rel: &str) -> Result<Manifest, String> {
    let path = root.join(rel);
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("{rel}: cannot read {}: {e}", path.display()))?;
    manifest::parse(rel, &text)
}
