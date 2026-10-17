// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `ErrorCode::Generic` is a migration escape hatch whose uses may only shrink. This
//! counts the path `ErrorCode::Generic` in every crate's sources (this crate excluded) against
//! the recorded ceiling. A `use ...::ErrorCode::Generic` import followed by a bare `Generic`
//! escapes the count, so review forbids that import.

use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};

/// 2026-10-17: Recorded ceiling. No crate has migrated yet, so it is 0; a migration that needs
/// Generic raises it in the same commit, and every later commit may only lower it.
const GENERIC_CEILING: usize = 0;

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("dir entry").path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if path.is_dir() {
            if name != "target" && !name.starts_with('.') {
                rust_sources(&path, out);
            }
        } else if name.ends_with(".rs") {
            out.push(path);
        }
    }
}

#[test]
fn generic_uses_never_exceed_the_ceiling() {
    let this_crate = Path::new(env!("CARGO_MANIFEST_DIR"));
    let crates = this_crate.parent().expect("crates/ dir");
    let mut files = Vec::new();
    for entry in fs::read_dir(crates).expect("read crates/") {
        let dir = entry.expect("crate entry").path();
        if dir.is_dir() && dir != this_crate {
            rust_sources(&dir, &mut files);
        }
    }
    assert!(
        files.len() > 100,
        "found only {} .rs files under {}",
        files.len(),
        crates.display()
    );
    let mut count = 0;
    let mut sites = Vec::new();
    for f in &files {
        let text = fs::read_to_string(f).unwrap_or_else(|e| panic!("read {}: {e}", f.display()));
        let n = text.matches("ErrorCode::Generic").count();
        if n > 0 {
            count += n;
            sites.push(format!("{} ({n})", f.display()));
        }
    }
    match count.cmp(&GENERIC_CEILING) {
        Ordering::Greater => panic!(
            "ErrorCode::Generic is used {count} times, above the ceiling {GENERIC_CEILING}: \
             {sites:?}. Register a specific code in crates/error/src/code.rs instead."
        ),
        Ordering::Less => panic!(
            "ErrorCode::Generic uses fell to {count}; lower GENERIC_CEILING in {} to {count}.",
            file!()
        ),
        Ordering::Equal => {}
    }
}
