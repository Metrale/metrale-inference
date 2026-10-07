// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: The "no bare asm" ratchet (#148). Inline assembly in the kernel and runtime sources
//! must go through the ISA router; until every site has moved, `kernels/ASM_RATCHET.toml` holds
//! the per-file count of bare `asm` statements, and a count may only go down.
//!
//! Owner: metrale-kernels tests.
//! Invariants:
//! - Scanned: every `.cu`, `.cuh`, `.h`, `.hpp`, `.c`, `.cpp` and `.metal` file below `kernels/`
//!   and `crates/`, except build output and the directories the ratchet lists as `exempt`
//!   (the router itself, once it exists). Symlinks are not followed: a linked file is counted
//!   where it lives.
//! - A statement is the keyword `asm`, `__asm` or `__asm__`, optionally `volatile` /
//!   `__volatile__`, then `(`, outside comments, string literals and character literals.
//! - Each file's count must equal its ratchet entry exactly, so a migration lowers the
//!   entry in the same change and the slack cannot be refilled later. A file not listed must
//!   have none, and an entry whose file has none (or no longer exists) is refused.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const EXTENSIONS: &[&str] = &["cu", "cuh", "h", "hpp", "c", "cpp", "metal"];
const SCAN_ROOTS: &[&str] = &["kernels", "crates"];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/kernels is two levels below the workspace root")
        .to_path_buf()
}

/// 2026-10-07: `src` with comments, string literals and character literals blanked out, line
/// structure kept. A raw string literal (`R"d(...)d"`) does not occur in the kernel tree; one
/// would be read as an ordinary string, which ends it early but never adds a statement.
fn strip(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'/' if b.get(i + 1) == Some(&b'/') => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if b.get(i + 1) == Some(&b'*') => {
                i += 2;
                while i < b.len() && !(b[i] == b'*' && b.get(i + 1) == Some(&b'/')) {
                    if b[i] == b'\n' {
                        out.push('\n');
                    }
                    i += 1;
                }
                i += 2;
            }
            q @ (b'"' | b'\'') => {
                i += 1;
                while i < b.len() && b[i] != q && b[i] != b'\n' {
                    i += if b[i] == b'\\' { 2 } else { 1 };
                }
                i += 1;
                out.push(' ');
            }
            c => {
                out.push(c as char);
                i += 1;
            }
        }
    }
    out
}

fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// 2026-10-07: The bare `asm` statements in `src` (see the module invariants).
fn count_asm(src: &str) -> usize {
    let s = strip(src);
    let b = s.as_bytes();
    let mut n = 0;
    let mut i = 0;
    while i < b.len() {
        if !is_ident(b[i]) {
            i += 1;
            continue;
        }
        let start = i;
        while i < b.len() && is_ident(b[i]) {
            i += 1;
        }
        if !matches!(&s[start..i], "asm" | "__asm" | "__asm__") {
            continue;
        }
        let mut j = i;
        let skip_ws = |j: &mut usize| {
            while *j < b.len() && b[*j].is_ascii_whitespace() {
                *j += 1;
            }
        };
        skip_ws(&mut j);
        let q = j;
        while j < b.len() && is_ident(b[j]) {
            j += 1;
        }
        if !matches!(&s[q..j], "" | "volatile" | "__volatile__") {
            continue;
        }
        skip_ws(&mut j);
        if b.get(j) == Some(&b'(') {
            n += 1;
        }
    }
    n
}

/// 2026-10-07: Repo-relative path -> bare `asm` count, for every scanned file with at least one.
fn scan(root: &Path, exempt: &[String]) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    let mut stack: Vec<PathBuf> = SCAN_ROOTS.iter().map(|r| root.join(r)).collect();
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            let rel = p
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            let Ok(ft) = e.file_type() else { continue };
            if ft.is_symlink() || exempt.iter().any(|x| rel.starts_with(x.as_str())) {
                continue;
            }
            if ft.is_dir() {
                if e.file_name() != "target" {
                    stack.push(p);
                }
                continue;
            }
            let ext = p.extension().and_then(|x| x.to_str()).unwrap_or("");
            if !EXTENSIONS.contains(&ext) {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&p) else {
                continue;
            };
            let n = count_asm(&text);
            if n > 0 {
                out.insert(rel, n);
            }
        }
    }
    out
}

fn render(counts: &BTreeMap<String, usize>) -> String {
    counts
        .iter()
        .map(|(f, n)| format!("\"{f}\" = {n}\n"))
        .collect()
}

#[test]
fn no_new_bare_asm() {
    let root = workspace_root();
    let path = root.join("kernels/ASM_RATCHET.toml");
    let text = std::fs::read_to_string(&path).expect("kernels/ASM_RATCHET.toml");
    let doc: toml::Table = toml::from_str(&text).expect("ASM_RATCHET.toml parses");
    let exempt: Vec<String> = doc
        .get("exempt")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let listed: BTreeMap<String, usize> = doc
        .get("files")
        .and_then(|v| v.as_table())
        .expect("ASM_RATCHET.toml has a [files] table")
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                v.as_integer().expect("counts are integers") as usize,
            )
        })
        .collect();
    let found = scan(&root, &exempt);

    let mut errors = Vec::new();
    for (f, &n) in &found {
        match listed.get(f) {
            None => errors.push(format!("{f}: {n} bare asm statement(s) in a file the ratchet does not list; route them through the ISA router (#148)")),
            Some(&m) if n > m => errors.push(format!("{f}: {n} bare asm statement(s), ratchet allows {m}; route the new ones through the ISA router (#148)")),
            Some(&m) if n < m => errors.push(format!("{f}: {n} bare asm statement(s), ratchet still says {m}; lower the entry to {n}")),
            _ => {}
        }
    }
    for f in listed.keys().filter(|f| !found.contains_key(*f)) {
        errors.push(format!(
            "{f}: no bare asm left (or the file is gone); remove its entry"
        ));
    }
    assert!(
        errors.is_empty(),
        "kernels/ASM_RATCHET.toml is out of date:\n  {}\n\nThe [files] table for this tree ({} statements in {} files):\n{}",
        errors.join("\n  "),
        found.values().sum::<usize>(),
        found.len(),
        render(&found)
    );
}

#[test]
fn scanner_counts_statements_not_mentions() {
    let src = r#"
        // asm volatile("comment");
        /* asm("block comment") */
        const char* s = "asm(not code)";
        char c = '"';
        asm volatile("mma.sync ..." : "=r"(d) : "r"(a));
        __asm__ __volatile__ ("cp.async ...");
        asm ("cvt ...");
        int my_asm(int x); my_asm(1); asmfoo(2);
        #define F(x) asm volatile("prmt" : "=r"(x))
    "#;
    assert_eq!(count_asm(src), 4);
}
