// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Build script for metrale-server: bakes the build's own git provenance
//! (commit + dirty flag) into the binary as `rustc-env`s, read back by `cli.rs`
//! (`BUILD_GIT_SHA`, `BUILD_GIT_DIRTY`) and printed by `met --version`.
//!
//! This exists for the `met bench preflight` binary-provenance check (lever journal,
//! PR #126): a stale pre-head binary measured in a timed A/B is a real incident this
//! embeds against. The embedded commit is what `build.rs` saw when it last ran, which —
//! since this script declares no `rerun-if-changed` for the git lookup — is every build
//! (cargo's default when a build script emits no `rerun-if` instruction at all is to run
//! it every time), so the embedded commit cannot go stale between commits the way a
//! `rerun-if-changed`-gated one could.
//!
//! Owner: server CLI (build).
//! Invariants:
//! - Fails closed, never silently: a build outside a git checkout (or with git missing)
//!   embeds the literal string `"unknown"` for the sha and `"unknown"` for the dirty
//!   flag, which can never equal a real `--expect-head` sha and is never read as "not
//!   dirty" (`BUILD_GIT_DIRTY` is a three-valued `&str`, not a `bool` that would have to
//!   pick a default). It does not panic the build: a packaged source tarball with no
//!   `.git` must still compile, just without a working provenance check.
//! - `METRALE_BUILD_GIT_SHA` is the full 40-hex commit, not the `--short=10` form
//!   `metrale_bench::gate::git_sha` reads at runtime: a preflight check comparing against
//!   an `--expect-head` resolved by `git rev-parse` wants the unambiguous form.

use std::process::Command;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("set by cargo");
    // 2026-10-05: Two levels up from `crates/server`, the same convention
    // `crates/kernels/build.rs` uses.
    let workspace_root = std::path::Path::new(&manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/server is two levels below the workspace root");

    let sha = git_sha(workspace_root).unwrap_or_else(|| "unknown".to_string());
    let dirty = is_dirty(workspace_root)
        .map(|d| if d { "true" } else { "false" })
        .unwrap_or("unknown");

    println!("cargo:rustc-env=METRALE_BUILD_GIT_SHA={sha}");
    println!("cargo:rustc-env=METRALE_BUILD_GIT_DIRTY={dirty}");
}

/// 2026-10-05: The full 40-hex `HEAD` commit, or `None` when git fails or this is not a
/// git checkout.
fn git_sha(root: &std::path::Path) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "HEAD"])
        .stdin(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let sha = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (sha.len() == 40 && sha.chars().all(|c| c.is_ascii_hexdigit())).then_some(sha)
}

/// 2026-10-05: Whether the working tree has uncommitted changes (tracked, modified, or
/// staged — `git status --porcelain` with no filter, the same question
/// `metrale_bench::gate::dirty_perf_paths` asks of `PERF_PATHS` alone, asked here of the
/// whole tree since a build's provenance covers everything it compiled). `None` when git
/// fails or this is not a git checkout.
fn is_dirty(root: &std::path::Path) -> Option<bool> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["status", "--porcelain"])
        .stdin(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(!out.stdout.is_empty())
}
