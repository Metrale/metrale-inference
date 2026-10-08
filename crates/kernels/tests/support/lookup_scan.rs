// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: The workspace scan for literal kernel lookups (`.kernel(`, `try_kernel(`,
//! `try_target_kernel(`, `gated(`), moved here from `kernel_lookups.rs`, plus
//! `Lookup::required` (a `.kernel(...)?` call).
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! Included with `#[path]` by `kernel_lookups.rs` (every lookup names some declared kernel)
//! and `strix_hip_laguna_int4.rs` (the Laguna lookups against one target). It sits below
//! `tests/`, so cargo does not build it as a test target of its own. Each includer reads
//! part of [`Lookup`], hence `allow(dead_code)`.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    for e in read.flatten() {
        let path = e.path();
        let name = e.file_name().to_string_lossy().to_string();
        if path.is_dir() {
            if name != "tests" && name != "target" {
                rust_files(&path, out);
            }
        } else if name.ends_with(".rs")
            && !name.ends_with("_tests.rs")
            && name != "tests.rs"
            && name != "mock.rs"
        {
            out.push(path);
        }
    }
}

/// 2026-09-26: The file's text up to its first inline `#[cfg(test)] mod x {`.
/// An out-of-line `#[cfg(test)] mod x;` is skipped: its file is a test file.
fn non_test_text(text: &str) -> &str {
    let mut at = 0;
    while let Some(i) = text[at..].find("#[cfg(test)]") {
        let after = text[at + i + "#[cfg(test)]".len()..].trim_start();
        if let Some(rest) = after.strip_prefix("mod ") {
            let rest = rest.trim_start_matches(|c: char| c.is_alphanumeric() || c == '_');
            if rest.trim_start().starts_with('{') {
                return &text[..at + i];
            }
        }
        at += i + 1;
    }
    text
}

/// 2026-09-26: `NAME -> values` for every `const NAME: &str = "..."` in `text`.
fn str_consts(text: &str, out: &mut BTreeMap<String, BTreeSet<String>>) {
    for line in text.lines() {
        let Some(rest) = line.trim_start().split("const ").nth(1) else {
            continue;
        };
        let Some((name, rest)) = rest.split_once(':') else {
            continue;
        };
        let Some((ty, value)) = rest.split_once('=') else {
            continue;
        };
        if !matches!(ty.trim(), "&str" | "&'static str") {
            continue;
        }
        let value = value.trim().trim_end_matches(';');
        if let Some(v) = value.strip_prefix('"').and_then(|v| v.strip_suffix('"')) {
            out.entry(name.trim().to_string())
                .or_default()
                .insert(v.to_string());
        }
    }
}

/// 2026-09-26: The top-level comma-separated arguments of the call whose
/// opening parenthesis ends `text[..open]`, or `None` if unbalanced.
fn call_args(text: &str, open: usize) -> Option<Vec<&str>> {
    let bytes = text.as_bytes();
    let (mut depth, mut start, mut in_str) = (0usize, open + 1, false);
    let mut args = Vec::new();
    let mut i = open + 1;
    while i < bytes.len() {
        let c = bytes[i];
        if in_str {
            match c {
                b'\\' => i += 1,
                b'"' => in_str = false,
                _ => {}
            }
        } else {
            match c {
                b'"' => in_str = true,
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' if depth > 0 => depth -= 1,
                b')' => {
                    // 2026-09-26: A trailing comma leaves an empty last argument.
                    let last = text[start..i].trim();
                    if !last.is_empty() {
                        args.push(last);
                    }
                    return Some(args);
                }
                b',' if depth == 0 => {
                    args.push(text[start..i].trim());
                    start = i + 1;
                }
                _ => {}
            }
        }
        i += 1;
    }
    None
}

fn literal(arg: &str) -> Option<&str> {
    let s = arg.strip_prefix('"')?.strip_suffix('"')?;
    (!s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')).then_some(s)
}

fn is_const_name(arg: &str) -> bool {
    arg.starts_with(|c: char| c.is_ascii_uppercase())
        && arg
            .bytes()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
}

/// 2026-09-26: One lookup: where it is, the candidate module names (one for a
/// literal, every value of the constant for a constant) and the entry point.
pub struct Lookup {
    /// 2026-10-07: `path:line`, workspace-relative.
    pub site: String,
    /// 2026-10-07: Candidate module names.
    pub modules: BTreeSet<String>,
    /// 2026-10-07: The entry point.
    pub func: String,
    /// 2026-10-07: `.kernel(...)?`: a missing entry fails the caller. Every other form
    /// (`try_kernel(`, `try_target_kernel(`, `gated(`, `.kernel(...)` handled in place) probes.
    pub required: bool,
}

const CALLS: &[&str] = &[".kernel(", "try_kernel(", "try_target_kernel(", "gated("];

pub fn lookups(root: &Path) -> Vec<Lookup> {
    let mut out = Vec::new();
    let mut crates: Vec<PathBuf> = std::fs::read_dir(root.join("crates"))
        .expect("crates/")
        .flatten()
        .map(|e| e.path())
        .collect();
    crates.sort();
    let sources: Vec<Vec<(PathBuf, String)>> = crates
        .iter()
        .map(|krate| {
            let mut files = Vec::new();
            rust_files(krate, &mut files);
            files.sort();
            files
                .into_iter()
                .map(|f| {
                    let t = std::fs::read_to_string(&f)
                        .unwrap_or_else(|e| panic!("{}: {e}", f.display()));
                    (f, t)
                })
                .collect()
        })
        .collect();
    // 2026-09-26: A constant resolves in its own crate first; one imported
    // from another crate (`KQUANT_MODULE` is model-layers', read by
    // model-arch) falls back to the workspace's.
    let mut workspace_consts = BTreeMap::new();
    for (_, t) in sources.iter().flatten() {
        str_consts(t, &mut workspace_consts);
    }
    for texts in &sources {
        let mut consts = BTreeMap::new();
        for (_, t) in texts {
            str_consts(t, &mut consts);
        }
        for (file, text) in texts {
            let text = non_test_text(text);
            for call in CALLS {
                let mut at = 0;
                while let Some(i) = text[at..].find(call) {
                    let pos = at + i;
                    at = pos + call.len();
                    // 2026-09-26: `try_kernel(` also matches inside `my_try_kernel(`.
                    let prev = text[..pos].chars().next_back();
                    if !call.starts_with('.')
                        && prev.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
                    {
                        continue;
                    }
                    let Some(args) = call_args(text, pos + call.len() - 1) else {
                        continue;
                    };
                    // 2026-10-07: The `)` closing the call, then whether `?` follows it.
                    let close = args.last().map_or(pos + call.len(), |last| {
                        last.as_ptr() as usize - text.as_ptr() as usize + last.len()
                    });
                    let after = text[close..].trim_start();
                    let after = after.strip_prefix(',').unwrap_or(after).trim_start();
                    let required = *call == ".kernel("
                        && after
                            .strip_prefix(')')
                            .is_some_and(|rest| rest.trim_start().starts_with('?'));
                    let [.., module, func] = args.as_slice() else {
                        continue;
                    };
                    let Some(func) = literal(func) else {
                        continue;
                    };
                    let modules: BTreeSet<String> = if let Some(m) = literal(module) {
                        [m.to_string()].into()
                    } else if is_const_name(module) {
                        match consts
                            .get(*module)
                            .or_else(|| workspace_consts.get(*module))
                        {
                            Some(values) => values.clone(),
                            None => continue,
                        }
                    } else {
                        continue;
                    };
                    let line = text[..pos].matches('\n').count() + 1;
                    out.push(Lookup {
                        site: format!("{}:{line}", file.strip_prefix(root).unwrap().display()),
                        modules,
                        func: func.to_string(),
                        required,
                    });
                }
            }
        }
    }
    out
}
