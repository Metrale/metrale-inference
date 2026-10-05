// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The repository's `kernels/` manifests and sources (`*.toml`, `*.cu`, `*.cuh`,
//! `*.h`) as they were when this binary was built, so code that reads the repository
//! (`met circuit memory`'s model, which `met serve` also runs) works without a checkout.
//!
//! Owner: kernel tree (build embedding).
//! Invariants:
//! - [`materialize`] returns `<dir>/<SHA256>/`, a directory named by the embedded tree's own
//!   digest: two binaries with different trees never share one.
//! - The returned tree is verified, every time: exactly the embedded files, byte for byte, and
//!   nothing else. A tree that differs (a planted or edited file, an extra file, a missing one)
//!   is moved aside and unpacked again.
//! - An unpack is written under a temporary name and renamed into place, so an interrupted one
//!   never leaves a directory under the final name, and concurrent serves race safely.
//! - No embedded path is absolute or has a `..` component: unpacking cannot write outside the
//!   tree.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};

const PACKED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernel_tree.bin"));

/// 2026-10-02: SHA-256 of the unpacked records: the tree's identity.
pub const SHA256: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel_tree.sha256"));

/// 2026-10-02: Every embedded file, `(repository-relative path, bytes)`, sorted by path.
pub fn files() -> Result<Vec<(String, Vec<u8>)>> {
    let mut raw = Vec::new();
    std::io::Read::read_to_end(&mut flate2::read::DeflateDecoder::new(PACKED), &mut raw)
        .context("inflate the embedded kernel tree")?;
    let mut out = Vec::new();
    let mut at = 0usize;
    let mut take = |n: usize| -> Result<&[u8]> {
        let end = at.checked_add(n).filter(|&e| e <= raw.len());
        let end = end.context("the embedded kernel tree is truncated")?;
        let s = &raw[at..end];
        at = end;
        Ok(s)
    };
    loop {
        let Ok(len) = take(4) else { break };
        let len = u32::from_le_bytes(len.try_into().expect("4 bytes")) as usize;
        let path = String::from_utf8(take(len)?.to_vec()).context("embedded path")?;
        let n = u64::from_le_bytes(take(8)?.try_into().expect("8 bytes")) as usize;
        let bytes = take(n)?.to_vec();
        check_path(&path)?;
        out.push((path, bytes));
    }
    Ok(out)
}

fn check_path(rel: &str) -> Result<()> {
    let p = Path::new(rel);
    ensure!(
        !rel.is_empty() && p.components().all(|c| matches!(c, Component::Normal(_))),
        "embedded path `{rel}` is not a plain relative path"
    );
    Ok(())
}

/// 2026-10-02: The embedded tree unpacked and verified under `dir/<SHA256>/`, a repository root
/// that has the `kernels/` files and nothing else.
pub fn materialize(dir: &Path) -> Result<PathBuf> {
    materialize_tree(dir, SHA256, &files()?)
}

/// 2026-10-03: Every regular file under `root`, repository-relative with `/`, and its bytes.
fn on_disk(root: &Path) -> Result<BTreeMap<String, Vec<u8>>> {
    let mut out = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).with_context(|| format!("read {}", d.display()))? {
            let p = e?.path();
            let meta = std::fs::symlink_metadata(&p)?;
            if meta.is_dir() {
                stack.push(p);
            } else {
                let rel = p.strip_prefix(root).expect("under the root");
                let rel = rel.to_string_lossy().replace('\\', "/");
                // 2026-10-03: A symlink or other non-regular entry is never part of the tree;
                // recording it as unreadable makes the comparison fail.
                let bytes = if meta.is_file() {
                    std::fs::read(&p)?
                } else {
                    b"\0not a regular file".to_vec()
                };
                out.insert(rel, bytes);
            }
        }
    }
    Ok(out)
}

/// 2026-10-03: Whether `root` holds exactly `files`.
fn verified(root: &Path, files: &[(String, Vec<u8>)]) -> Result<bool> {
    if !root.is_dir() {
        return Ok(false);
    }
    let disk = on_disk(root)?;
    Ok(disk.len() == files.len()
        && files
            .iter()
            .all(|(rel, bytes)| disk.get(rel).is_some_and(|d| d == bytes)))
}

/// 2026-10-03: [`materialize`] for any tree: `files` under `dir/<digest>/`, verified, unpacked
/// again when it differs.
fn materialize_tree(dir: &Path, digest: &str, files: &[(String, Vec<u8>)]) -> Result<PathBuf> {
    ensure!(
        digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()),
        "kernel tree digest `{digest}` is not a SHA-256"
    );
    let root = dir.join(digest);
    if verified(&root, files)? {
        return Ok(root);
    }
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let pid = std::process::id();
    if root.exists() {
        // 2026-10-03: Moved aside first, so no reader is left with a half-removed tree.
        let stale = dir.join(format!("{digest}.stale-{pid}"));
        std::fs::rename(&root, &stale)
            .with_context(|| format!("move the unverified {} aside", root.display()))?;
        std::fs::remove_dir_all(&stale).with_context(|| format!("remove {}", stale.display()))?;
    }
    let tmp = dir.join(format!("{digest}.partial-{pid}"));
    if tmp.exists() {
        std::fs::remove_dir_all(&tmp).with_context(|| format!("remove {}", tmp.display()))?;
    }
    for (rel, bytes) in files {
        check_path(rel)?;
        let path = tmp.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        std::fs::write(&path, bytes).with_context(|| format!("write {}", path.display()))?;
    }
    match std::fs::rename(&tmp, &root) {
        Ok(()) => {}
        // 2026-10-03: Another process renamed its own unpack into place first.
        Err(_) if root.is_dir() => {
            std::fs::remove_dir_all(&tmp).with_context(|| format!("remove {}", tmp.display()))?;
        }
        Err(e) => bail!("rename {} to {}: {e}", tmp.display(), root.display()),
    }
    ensure!(
        verified(&root, files)?,
        "{} does not hold the kernel tree {digest} after unpacking it",
        root.display()
    );
    Ok(root)
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
