// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The env-ban I/O shell: walks a tree for `.rs` files, classifies each, and lexes
//! it. Symlinks are not followed; an unreadable or non-UTF-8 source is an error, not a pass.

use super::judge::{Kind, Source};
use super::lex;
use std::fs;
use std::path::Path;

/// 2026-10-17: Directories never scanned, relative to the scan root: third-party code and the
/// gates' own fixtures (which hold deliberate reads). `target` and dot-directories are skipped
/// at any depth.
const SKIP: [&str; 2] = ["vendor", "xtask/fixtures"];

/// 2026-10-17: Every `.rs` file under `root`, sorted by path.
pub fn scan(root: &Path) -> Result<Vec<Source>, String> {
    let mut out = Vec::new();
    walk(root, "", &mut out)?;
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

fn walk(root: &Path, rel: &str, out: &mut Vec<Source>) -> Result<(), String> {
    let dir = root.join(rel);
    let entries = fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let has_manifest = dir.join("Cargo.toml").is_file();
    for entry in entries {
        let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
        let name = entry.file_name();
        let name = name
            .to_str()
            .ok_or_else(|| format!("{}: a non-UTF-8 file name", dir.display()))?;
        let child = if rel.is_empty() {
            name.to_owned()
        } else {
            format!("{rel}/{name}")
        };
        let ty = entry.file_type().map_err(|e| format!("{child}: {e}"))?;
        if ty.is_dir() {
            if name.starts_with('.') || name == "target" || SKIP.contains(&child.as_str()) {
                continue;
            }
            walk(root, &child, out)?;
        } else if ty.is_file() && name.ends_with(".rs") {
            let text = fs::read_to_string(root.join(&child))
                .map_err(|e| format!("{child}: cannot read: {e}"))?;
            let build = name == "build.rs" || name.starts_with("build_");
            let kind = if child == super::HOST_ENV {
                Kind::HostEnv
            } else if build && has_manifest {
                Kind::BuildScript
            } else {
                Kind::Ordinary
            };
            out.push(Source {
                reads: lex::reads(&text),
                path: child,
                kind,
            });
        }
    }
    Ok(())
}
