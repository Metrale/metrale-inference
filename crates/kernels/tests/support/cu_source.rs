// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Plain-text reading of CUDA kernel sources for the tests that check what an
//! entry point declares against what its body does: entry declarations, brace-matched
//! bodies, `#define` integers, and an entry's body with the macros and `*_impl` helpers it
//! uses appended.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! Included with `#[path]` by `kernel_n_tile.rs` and `kernel_a_e4m3.rs`. It sits below
//! `tests/`, so cargo does not build it as a test target of its own. Each includer uses
//! part of it, hence `allow(dead_code)`.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/kernels is two levels below the workspace root")
        .to_path_buf()
}

pub fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// 2026-09-29: `NAME -> value` for every `#define NAME <integer>`.
pub fn int_defines(text: &str) -> BTreeMap<String, u32> {
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        if it.next() != Some("#define") {
            continue;
        }
        if let (Some(name), Some(v)) = (it.next(), it.next())
            && let Ok(v) = v.parse::<u32>()
        {
            out.insert(name.to_string(), v);
        }
    }
    out
}

/// 2026-09-29: Each `extern "C"` declaration that contains `__global__`, as
/// `(name, offset just past the name's '(')`. A declaration runs to its first `{` or `;`,
/// so a `__launch_bounds__(...)` between `__global__` and the name is part of it.
pub fn global_entries(text: &str) -> Vec<(String, usize)> {
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(i) = text[at..].find("extern \"C\"") {
        let decl_at = at + i;
        let decl_end = text[decl_at..]
            .find(['{', ';'])
            .map_or(text.len(), |p| decl_at + p);
        let decl = &text[decl_at..decl_end];
        if decl.contains("__global__") {
            let mut from = 0;
            while let Some(p) = decl[from..].find('(') {
                let paren = from + p;
                let head = decl[..paren].trim_end();
                let name: String = head
                    .chars()
                    .rev()
                    .take_while(|c| is_ident(*c))
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect();
                if !name.is_empty() && name != "__launch_bounds__" {
                    out.push((name, decl_at + paren + 1));
                    break;
                }
                from = paren + 1;
            }
        }
        at = decl_at + 1;
    }
    out
}

/// 2026-09-29: Byte offset just past `name(` in the first `extern "C" __global__` declaration
/// of `name`, if the file declares it.
pub fn entry_start(text: &str, name: &str) -> Option<usize> {
    global_entries(text)
        .into_iter()
        .find(|(n, _)| n == name)
        .map(|(_, at)| at)
}

/// 2026-09-29: The brace-matched block that opens at the first `{` at or after `from`.
pub fn block_at(text: &str, from: usize) -> Option<&str> {
    let open = from + text[from..].find('{')?;
    let mut depth = 0usize;
    for (i, c) in text[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[open..=open + i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// 2026-09-29: Identifiers in `body` directly followed by `(`, or by `<` for a template call
/// such as `helper_impl<32, true>(` (spaces allowed between).
fn called(body: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for (i, _) in body.match_indices(['(', '<']) {
        let head = body[..i].trim_end();
        let name: String = head
            .chars()
            .rev()
            .take_while(|c| is_ident(*c))
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        if !name.is_empty() {
            out.insert(name);
        }
    }
    out
}

/// 2026-09-29: The full `#define name ...` including `\`-continued lines, if `text` has one.
fn macro_def<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let mut at = 0;
    while let Some(i) = text[at..].find("#define") {
        let start = at + i;
        let rest = text[start + "#define".len()..].trim_start();
        let ident: String = rest.chars().take_while(|c| is_ident(*c)).collect();
        let mut end = start;
        loop {
            let line_end = text[end..].find('\n').map_or(text.len(), |p| end + p);
            if !text[end..line_end].trim_end().ends_with('\\') || line_end == text.len() {
                end = line_end;
                break;
            }
            end = line_end + 1;
        }
        if ident == name {
            return Some(&text[start..end]);
        }
        at = end.max(start + 1);
    }
    None
}

/// 2026-09-29: The entry's body, followed by the definitions of the upper-case macros and the
/// `*_impl` helpers it calls, recursively. `None` when the file does not declare `entry`.
pub fn expanded_body(text: &str, entry: &str) -> Option<String> {
    let body = block_at(text, entry_start(text, entry)?)?;
    let mut seen = BTreeSet::new();
    let mut out = String::new();
    expand_into(text, body, &mut seen, &mut out);
    Some(out)
}

fn expand_into(text: &str, part: &str, seen: &mut BTreeSet<String>, out: &mut String) {
    out.push_str(part);
    for name in called(part) {
        let is_macro = name.len() > 3
            && name.starts_with(|c: char| c.is_ascii_uppercase())
            && name
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
        if !(is_macro || name.ends_with("_impl")) || !seen.insert(name.clone()) {
            continue;
        }
        let def = if is_macro {
            macro_def(text, &name)
        } else {
            text.find(&format!("void {name}("))
                .and_then(|at| block_at(text, at))
        };
        if let Some(def) = def {
            expand_into(text, def, seen, out);
        }
    }
}
