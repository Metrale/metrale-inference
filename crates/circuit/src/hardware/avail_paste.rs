// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Kernels whose names a macro builds by token pasting
//! (`#define ENTRY(R, S) __global__ void k_##R##_s##S(...)`, invoked as `ENTRY(16, 1)` directly
//! or through another macro), which no line of the source spells out. Each comes with the
//! line of the top-level invocation that defines it, so its guard is that line's region.
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - Only function-like macros whose body holds `__global__` and a pasted identifier directly
//!   before `(` define a kernel; arguments substitute whole parameters, as the preprocessor
//!   does, through at most [`DEPTH`] nested invocations.

/// 2026-10-22: How deep one invocation may nest macro calls before expansion stops.
use crate::prelude::*;

const DEPTH: usize = 4;

struct Macro {
    name: String,
    params: Vec<String>,
    body: String,
}

fn ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// 2026-10-22: The function-like `#define`s of `lines`, continuations joined, and per line
/// whether it belongs to one.
fn macros(lines: &[&str]) -> (Vec<Macro>, Vec<bool>) {
    let (mut out, mut inside) = (Vec::new(), vec![false; lines.len()]);
    let mut i = 0;
    while i < lines.len() {
        let define = lines[i]
            .trim_start()
            .strip_prefix('#')
            .map(str::trim_start)
            .and_then(|r| r.strip_prefix("define"));
        let Some(define) = define else {
            i += 1;
            continue;
        };
        let mut body = define.to_string();
        inside[i] = true;
        while body.trim_end().ends_with('\\') && i + 1 < lines.len() {
            body.truncate(body.trim_end().len() - 1);
            i += 1;
            inside[i] = true;
            body.push(' ');
            body.push_str(lines[i]);
        }
        i += 1;
        let head = body.trim_start();
        let name: String = head.chars().take_while(|&c| ident(c)).collect();
        if let Some((params, rest)) = head[name.len()..]
            .strip_prefix('(')
            .and_then(|r| r.split_once(')'))
        {
            out.push(Macro {
                params: params.split(',').map(|p| p.trim().to_string()).collect(),
                body: rest.to_string(),
                name,
            });
        }
    }
    (out, inside)
}

/// 2026-10-22: The pasted identifiers of a kernel-defining body: each `a##B##c` run directly
/// followed by `(`, as its pieces.
fn templates(m: &Macro) -> Vec<Vec<String>> {
    if !m.body.contains("__global__") {
        return Vec::new();
    }
    let b: Vec<char> = m.body.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if !ident(b[i]) || (i > 0 && (ident(b[i - 1]) || b[i - 1] == '#')) {
            i += 1;
            continue;
        }
        let start = i;
        while i < b.len() && (ident(b[i]) || b[i] == '#') {
            i += 1;
        }
        let run: String = b[start..i].iter().collect();
        let next = b[i..].iter().find(|c| !c.is_whitespace());
        if run.contains("##") && next == Some(&'(') {
            out.push(run.split("##").map(str::to_string).collect());
        }
    }
    out
}

/// 2026-10-22: The arguments of the invocation whose `(` opens `s`, up to its matching `)`.
fn args(s: &str) -> Option<Vec<String>> {
    let mut depth = 0usize;
    let mut cur = String::new();
    let mut out = Vec::new();
    for c in s.strip_prefix('(')?.chars() {
        match c {
            '(' => depth += 1,
            ')' if depth == 0 => {
                out.push(cur.trim().to_string());
                return Some(out);
            }
            ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
                continue;
            }
            _ => {}
        }
        cur.push(c);
    }
    None
}

/// 2026-10-22: The invocations of a macro of `all` in `text`, with their arguments.
fn calls<'m>(text: &str, all: &'m [Macro]) -> Vec<(&'m Macro, Vec<String>)> {
    let mut out = Vec::new();
    for m in all {
        for (at, _) in text.match_indices(m.name.as_str()) {
            let before = text[..at].chars().next_back();
            if before.is_some_and(|c| ident(c) || c == '#') {
                continue;
            }
            let rest = text[at + m.name.len()..].trim_start();
            if let Some(a) = args(rest) {
                out.push((m, a));
            }
        }
    }
    out
}

fn expand(m: &Macro, a: &[String], all: &[Macro], depth: usize, out: &mut Vec<String>) {
    let sub = |p: &str| {
        m.params
            .iter()
            .position(|q| q == p)
            .and_then(|i| a.get(i))
            .map_or(p.to_string(), Clone::clone)
    };
    for t in templates(m) {
        out.push(t.iter().map(|p| sub(p)).collect());
    }
    if depth == 0 {
        return;
    }
    for (inner, ia) in calls(&m.body, all) {
        let ia: Vec<String> = ia.iter().map(|x| sub(x)).collect();
        expand(inner, &ia, all, depth - 1, out);
    }
}

/// 2026-10-22: Every kernel a top-level macro invocation in `lines` defines by pasting, with
/// the invocation's line.
pub(super) fn pasted_entries(lines: &[&str]) -> Vec<(String, usize)> {
    let (all, inside) = macros(lines);
    if all.iter().all(|m| templates(m).is_empty()) {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate().filter(|(i, _)| !inside[*i]) {
        let line = line.trim_start();
        for m in &all {
            let Some(a) = line
                .strip_prefix(m.name.as_str())
                .map(str::trim_start)
                .and_then(args)
            else {
                continue;
            };
            let mut names = Vec::new();
            expand(m, &a, &all, DEPTH, &mut names);
            out.extend(names.into_iter().map(|n| (n, i)));
        }
    }
    out
}
