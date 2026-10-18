// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The purity gates' I/O shell: lists a tree's files through git, reads them, asks
//! cargo for the resolved graph and builds a crate for a target. Every judgement is made on the
//! values these return, in the gate modules.

use crate::metadata::{self, Graph};
use std::path::Path;
use std::process::{Command, Output};

fn run(cmd: &mut Command, what: &str) -> Result<Output, String> {
    cmd.output().map_err(|e| format!("{what}: cannot run: {e}"))
}

fn stdout_ok(out: Output, what: &str) -> Result<String, String> {
    if !out.status.success() {
        return Err(format!(
            "{what}: {}\n{}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    String::from_utf8(out.stdout).map_err(|e| format!("{what}: output is not UTF-8: {e}"))
}

/// 2026-10-17: Every file under `root` git tracks or would track (untracked files that are not
/// ignored count, so a new file cannot dodge a gate before it is added), relative to `root`.
pub fn list_files(root: &Path) -> Result<Vec<String>, String> {
    let what = format!("git ls-files in {}", root.display());
    let out = run(
        Command::new("git").arg("-C").arg(root).args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ]),
        &what,
    )?;
    let text = stdout_ok(out, &what)?;
    let mut files: Vec<String> = text
        .split('\0')
        .filter(|f| !f.is_empty())
        .map(str::to_owned)
        .collect();
    files.sort_unstable();
    files.dedup();
    Ok(files)
}

/// 2026-10-17: Whether `rel` exists under `root` (a tracked file deleted in the working tree is
/// listed by git but has nothing to judge).
pub fn exists(root: &Path, rel: &str) -> bool {
    root.join(rel).is_file()
}

/// 2026-10-17: The bytes of `rel` under `root`.
pub fn read_bytes(root: &Path, rel: &str) -> Result<Vec<u8>, String> {
    std::fs::read(root.join(rel)).map_err(|e| format!("{rel}: cannot read: {e}"))
}

/// 2026-10-17: The text of `rel` under `root`.
pub fn read_text(root: &Path, rel: &str) -> Result<String, String> {
    String::from_utf8(read_bytes(root, rel)?).map_err(|e| format!("{rel}: not UTF-8: {e}"))
}

/// 2026-10-17: The host target triple, as `rustc -vV` reports it.
pub fn host_triple() -> Result<String, String> {
    let text = stdout_ok(
        run(Command::new("rustc").arg("-vV"), "rustc -vV")?,
        "rustc -vV",
    )?;
    text.lines()
        .find_map(|l| l.strip_prefix("host: "))
        .map(str::to_owned)
        .ok_or_else(|| "rustc -vV: no `host:` line".to_owned())
}

/// 2026-10-17: The workspace graph at `root` resolved for `platform`.
pub fn cargo_metadata(root: &Path, platform: &str) -> Result<Graph, String> {
    let what = format!("cargo metadata --filter-platform {platform}");
    let out = run(
        Command::new("cargo")
            .current_dir(root)
            .args([
                "metadata",
                "--format-version",
                "1",
                "--filter-platform",
                platform,
            ])
            .arg("--manifest-path")
            .arg(root.join("Cargo.toml")),
        &what,
    )?;
    metadata::parse(platform, &stdout_ok(out, &what)?)
}

/// 2026-10-17: How a crate's build for a target ended.
pub enum Built {
    Ok,
    /// The first error rustc or cargo printed.
    Failed(String),
}

/// 2026-10-17: `cargo build -p package --target target` in the workspace at `root`, into
/// `target_dir` when given (the selftest's throwaway workspaces) or cargo's own otherwise. A
/// target that is not installed is an error, not a failed build: every crate would fail.
pub fn cargo_build(
    root: &Path,
    package: &str,
    target: &str,
    target_dir: Option<&Path>,
) -> Result<Built, String> {
    let what = format!("cargo build -p {package} --target {target}");
    let mut cmd = Command::new("cargo");
    cmd.current_dir(root)
        .args(["build", "--quiet", "-p", package, "--target", target])
        .arg("--manifest-path")
        .arg(root.join("Cargo.toml"));
    if let Some(dir) = target_dir {
        cmd.arg("--target-dir").arg(dir);
    }
    let out = run(&mut cmd, &what)?;
    if out.status.success() {
        return Ok(Built::Ok);
    }
    let stderr = String::from_utf8_lossy(&out.stderr);
    if stderr.contains("target may not be installed") {
        return Err(format!(
            "{what}: the `{target}` target is not installed (rustup target add {target})"
        ));
    }
    let first = stderr
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with("error") && !l.contains("could not compile"))
        .or_else(|| stderr.lines().map(str::trim).find(|l| !l.is_empty()))
        .unwrap_or("no output")
        .to_owned();
    Ok(Built::Failed(first))
}

/// 2026-10-17: Replace `to` with a copy of the tree at `from` (the selftest builds its fixture
/// workspaces in a throwaway copy so cargo never writes into the source tree).
pub fn copy_tree(from: &Path, to: &Path) -> Result<(), String> {
    if to.exists() {
        std::fs::remove_dir_all(to).map_err(|e| format!("{}: cannot remove: {e}", to.display()))?;
    }
    copy_dir(from, to)
}

fn copy_dir(from: &Path, to: &Path) -> Result<(), String> {
    std::fs::create_dir_all(to).map_err(|e| format!("{}: cannot create: {e}", to.display()))?;
    let entries =
        std::fs::read_dir(from).map_err(|e| format!("{}: cannot list: {e}", from.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("{}: cannot list: {e}", from.display()))?;
        let src = entry.path();
        let dst = to.join(entry.file_name());
        let kind = entry
            .file_type()
            .map_err(|e| format!("{}: {e}", src.display()))?;
        if kind.is_dir() {
            copy_dir(&src, &dst)?;
        } else {
            std::fs::copy(&src, &dst)
                .map_err(|e| format!("{}: cannot copy: {e}", src.display()))?;
        }
    }
    Ok(())
}
