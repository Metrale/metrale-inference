// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Which paths in a file read the process environment, once its `use` declarations
//! are resolved. Pure: tokens in, `(char index, call)` out.
//!
//! A read is a path that is, or resolves through the file's `use` declarations to:
//! - `env::{var, var_os, vars, vars_os}` or `levers::{var, var_os, vars, vars_os}`, under any
//!   prefix (`std::env::var`, `metrale_config::levers::var`);
//! - anything in `metrale_gpu_sys::env`, including the module itself: its key-chain helpers
//!   (`first_set` and the rest) read the environment;
//! - a glob import of `std::env`, `levers` or `metrale_gpu_sys::env`.
//!
//! Renamed imports resolve: after `use std::env as e;`, `e::var` is `std::env::var`, and
//! `use std::env::var as get;` is itself a read. `env!`/`option_env!` are compile-time (the
//! Build class) and are not reads.
//!
//! Bindings are collected file-wide, not per scope, and a path is a read when it, or any
//! binding it can resolve through, matches. A binding in one function can therefore make a
//! path elsewhere a read: a false positive, never a miss.

use super::lex::{Tok, Token};
use std::collections::BTreeMap;

/// 2026-10-17: Modules whose `var*` functions read the process environment.
const MODULES: [&str; 2] = ["env", LEVERS];
const LEVERS: &str = "levers";
const FUNCTIONS: [&str; 4] = ["var", "var_os", "vars", "vars_os"];

/// 2026-10-17: Globs that import a read: `std::env::*`, and `levers::*` under any prefix. A
/// bare or crate-local `env::*` names a module of that name (2018 paths), not std's, unless
/// the file bound `env` to `std::env`.
const STD_ENV: [&str; 2] = ["std", "env"];

/// 2026-10-17: A module every item of which reads the process environment.
const WHOLE_MODULE: [&str; 2] = ["metrale_gpu_sys", "env"];

/// 2026-10-17: Alias chains are followed this many steps (`use std::env as a; use a as b;`).
const RESOLVE_DEPTH: usize = 4;

/// 2026-10-17: A path as written, the name a `use` leaf binds (none for `as _` or a glob),
/// whether it is a glob, and the char index a read on it is reported at.
struct Named {
    path: Vec<String>,
    binds: Option<String>,
    glob: bool,
    pos: usize,
}

/// 2026-10-17: Every read among `toks`, in source order.
pub fn reads(toks: &[Token]) -> Vec<(usize, String)> {
    let named = split(toks);
    let mut bindings: BTreeMap<&str, Vec<&[String]>> = BTreeMap::new();
    for n in &named {
        if let Some(name) = &n.binds {
            bindings.entry(name).or_default().push(&n.path);
        }
    }
    let mut out: Vec<(usize, String)> = named
        .iter()
        .filter_map(|n| {
            candidates(&n.path, &bindings)
                .iter()
                .find_map(|p| banned(p, n.glob))
                .map(|call| (n.pos, call))
        })
        .collect();
    out.sort();
    out
}

/// 2026-10-17: `path` and every path it can resolve to through `bindings`.
fn candidates(path: &[String], bindings: &BTreeMap<&str, Vec<&[String]>>) -> Vec<Vec<String>> {
    let mut all = vec![path.to_vec()];
    let mut frontier = all.clone();
    for _ in 0..RESOLVE_DEPTH {
        let mut next = Vec::new();
        for p in &frontier {
            let Some((head, rest)) = p.split_first() else {
                continue;
            };
            for target in bindings.get(head.as_str()).into_iter().flatten() {
                let mut q = target.to_vec();
                q.extend_from_slice(rest);
                if !all.contains(&q) {
                    all.push(q.clone());
                    next.push(q);
                }
            }
        }
        frontier = next;
    }
    all
}

/// 2026-10-17: The call a resolved path reads the environment through, if it does.
fn banned(p: &[String], glob: bool) -> Option<String> {
    let star = if glob { "::*" } else { "" };
    if let Some(i) = p
        .windows(2)
        .position(|w| w[0] == WHOLE_MODULE[0] && w[1] == WHOLE_MODULE[1])
    {
        return Some(format!("{}{star}", p[i..].join("::")));
    }
    if glob {
        let reads = match p {
            [.., std, env] if *std == STD_ENV[0] && *env == STD_ENV[1] => true,
            [.., module] => module == LEVERS,
            [] => false,
        };
        return p.last().filter(|_| reads).map(|m| format!("{m}{star}"));
    }
    let [.., module, function] = p else {
        return None;
    };
    (MODULES.contains(&module.as_str()) && FUNCTIONS.contains(&function.as_str()))
        .then(|| format!("{module}::{function}"))
}

fn kind(toks: &[Token], i: usize) -> Option<&Tok> {
    toks.get(i).map(|t| &t.kind)
}

/// 2026-10-17: The `use` leaves and the code paths of a file. `use` after `#` is the raw
/// identifier `r#use`, not the keyword.
fn split(toks: &[Token]) -> Vec<Named> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        match &toks[i].kind {
            Tok::Ident(w)
                if w == "use" && kind(toks, i.wrapping_sub(1)) != Some(&Tok::Punct('#')) =>
            {
                let end = toks[i..]
                    .iter()
                    .position(|t| t.kind == Tok::Punct(';'))
                    .map_or(toks.len(), |p| i + p);
                tree(&toks[..end], i + 1, &[], &mut out);
                i = end + 1;
            }
            Tok::Ident(first) => {
                let (mut path, mut pos, mut j) = (vec![first.clone()], toks[i].pos, i + 1);
                while let (Some(Tok::Sep), Some(Tok::Ident(seg))) =
                    (kind(toks, j), kind(toks, j + 1))
                {
                    path.push(seg.clone());
                    pos = toks[j + 1].pos;
                    j += 2;
                }
                out.push(Named {
                    path,
                    binds: None,
                    glob: false,
                    pos,
                });
                i = j;
            }
            _ => i += 1,
        }
    }
    out
}

/// 2026-10-17: One `use` tree starting at `i` under `prefix`; returns the index after it, or
/// `i` itself when nothing there is a tree.
fn tree(toks: &[Token], mut i: usize, prefix: &[String], out: &mut Vec<Named>) -> usize {
    let mut path = prefix.to_vec();
    if kind(toks, i) == Some(&Tok::Sep) {
        i += 1;
    }
    loop {
        match kind(toks, i) {
            Some(Tok::Punct('{')) => return group(toks, i + 1, &path, out),
            Some(Tok::Punct('*')) => {
                out.push(Named {
                    path,
                    binds: None,
                    glob: true,
                    pos: toks[i].pos,
                });
                return i + 1;
            }
            Some(Tok::Ident(seg)) => {
                let pos = toks[i].pos;
                path.push(seg.clone());
                i += 1;
                if kind(toks, i) == Some(&Tok::Sep) {
                    i += 1;
                    continue;
                }
                return leaf(toks, i, path, pos, out);
            }
            _ => return i,
        }
    }
}

/// 2026-10-17: A leaf naming `path`, with its optional `as` rename at `i`.
fn leaf(
    toks: &[Token],
    mut i: usize,
    mut path: Vec<String>,
    pos: usize,
    out: &mut Vec<Named>,
) -> usize {
    if path.last().is_some_and(|s| s == "self") {
        path.pop();
    }
    let mut binds = path.last().cloned();
    if matches!(kind(toks, i), Some(Tok::Ident(w)) if w == "as") {
        binds = match kind(toks, i + 1) {
            Some(Tok::Ident(name)) if name != "_" => Some(name.clone()),
            _ => None,
        };
        i += 2;
    }
    out.push(Named {
        path,
        binds,
        glob: false,
        pos,
    });
    i
}

/// 2026-10-17: The trees of a `{ … }` group whose `{` is just before `i`; an attribute on an
/// item (`#[cfg(…)]`) is skipped. Returns the index after the closing `}`.
fn group(toks: &[Token], mut i: usize, prefix: &[String], out: &mut Vec<Named>) -> usize {
    loop {
        match kind(toks, i) {
            None => return i,
            Some(Tok::Punct('}')) => return i + 1,
            Some(Tok::Punct(',')) => i += 1,
            Some(Tok::Punct('#')) => {
                let close = toks[i..]
                    .iter()
                    .position(|t| t.kind == Tok::Punct(']'))
                    .map_or(toks.len(), |p| i + p);
                i = close + 1;
            }
            _ => {
                let next = tree(toks, i, prefix, out);
                if next == i {
                    return i;
                }
                i = next;
            }
        }
    }
}
