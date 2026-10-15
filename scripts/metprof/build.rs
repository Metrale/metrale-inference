// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: stamp the commit metprof is built from into the binary
//! (`METPROF_GIT_SHA`, recorded with every run), or `unknown` outside a git
//! checkout. Rebuilt when HEAD moves.

use std::path::Path;
use std::process::Command;

fn git(dir: &str, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!s.is_empty()).then_some(s)
}

fn watch(path: &Path) {
    if path.exists() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let dir = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let sha = git(&dir, &["rev-parse", "HEAD"]);
    if let Some(git_dir) = git(&dir, &["rev-parse", "--absolute-git-dir"]) {
        watch(&Path::new(&git_dir).join("HEAD"));
        if let Some(r) = git(&dir, &["symbolic-ref", "-q", "HEAD"]) {
            let common = git(
                &dir,
                &["rev-parse", "--path-format=absolute", "--git-common-dir"],
            )
            .unwrap_or_else(|| git_dir.clone());
            watch(&Path::new(&git_dir).join(&r));
            watch(&Path::new(&common).join(&r));
            watch(&Path::new(&common).join("packed-refs"));
        }
    }
    println!(
        "cargo:rustc-env=METPROF_GIT_SHA={}",
        sha.as_deref().unwrap_or("unknown")
    );
}
