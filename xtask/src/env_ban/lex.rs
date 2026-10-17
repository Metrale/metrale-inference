// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Finds runtime environment reads in Rust source text. Comments, string literals
//! and char literals are blanked first, so a read named in a doc comment or a message is not a
//! read. Pure: text in, findings out.
//!
//! A read is a path ending `env::{var, var_os, vars, vars_os}` or `levers::{var, var_os, vars,
//! vars_os}`, including a brace list (`use std::env::{self, var}`). `env!`/`option_env!` are
//! compile-time and are not reads. An alias (`use std::env as e;`) is not seen; review catches
//! it, and the allowlist names the files that still read the environment.

/// 2026-10-17: One read: the 1-based line and the call as written, e.g. `env::var`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Read {
    pub line: usize,
    pub call: String,
}

/// 2026-10-17: Modules whose `var*` functions read the process environment.
const MODULES: [&str; 2] = ["env", "levers"];
const FUNCTIONS: [&str; 4] = ["var", "var_os", "vars", "vars_os"];

fn is_ident(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn blank_range(out: &mut [char], src: &[char], from: usize, to: usize) {
    let to = to.min(src.len());
    for (o, s) in out[from..to].iter_mut().zip(&src[from..to]) {
        if *s != '\n' {
            *o = ' ';
        }
    }
}

/// 2026-10-17: The end (exclusive) of a `"…"` literal whose opening quote is at `open`.
fn string_end(c: &[char], open: usize) -> usize {
    let mut j = open + 1;
    while j < c.len() {
        match c[j] {
            '\\' => j += 2,
            '"' => return j + 1,
            _ => j += 1,
        }
    }
    c.len()
}

/// 2026-10-17: The end (exclusive) of a raw string whose `r` is at `r_at`, or `None` when the
/// `r` does not start one (a raw identifier `r#name`, or a plain identifier).
fn raw_string_end(c: &[char], r_at: usize) -> Option<usize> {
    let mut j = r_at + 1;
    let mut hashes = 0usize;
    while c.get(j) == Some(&'#') {
        hashes += 1;
        j += 1;
    }
    if c.get(j) != Some(&'"') {
        return None;
    }
    j += 1;
    while j < c.len() {
        if c[j] == '"' && (1..=hashes).all(|h| c.get(j + h) == Some(&'#')) {
            return Some(j + 1 + hashes);
        }
        j += 1;
    }
    Some(c.len())
}

/// 2026-10-17: The end (exclusive) of a nested `/* … */` comment opening at `open`.
fn block_comment_end(c: &[char], open: usize) -> usize {
    let (mut j, mut depth) = (open + 2, 1usize);
    while j < c.len() && depth > 0 {
        if c[j] == '/' && c.get(j + 1) == Some(&'*') {
            depth += 1;
            j += 2;
        } else if c[j] == '*' && c.get(j + 1) == Some(&'/') {
            depth -= 1;
            j += 2;
        } else {
            j += 1;
        }
    }
    j
}

/// 2026-10-17: The end (exclusive) of a char literal opening at `open`, or `None` for a
/// lifetime or label (`'a`, `'outer:`).
fn char_end(c: &[char], open: usize) -> Option<usize> {
    if c.get(open + 1) == Some(&'\\') {
        let mut j = open + 3;
        while j < c.len() && c[j] != '\'' && c[j] != '\n' {
            j += 1;
        }
        return Some((j + 1).min(c.len()));
    }
    (c.get(open + 2) == Some(&'\'')).then_some(open + 3)
}

/// 2026-10-17: `src` with every comment, string and char literal replaced by spaces; newlines
/// are kept, so a position in the result has the same line as in `src`.
pub fn blank(src: &str) -> Vec<char> {
    let c: Vec<char> = src.chars().collect();
    let mut out = c.clone();
    let mut i = 0;
    while i < c.len() {
        let boundary = i == 0 || !is_ident(c[i - 1]);
        let end = match c[i] {
            '/' if c.get(i + 1) == Some(&'/') => Some(
                c[i..]
                    .iter()
                    .position(|&x| x == '\n')
                    .map_or(c.len(), |p| i + p),
            ),
            '/' if c.get(i + 1) == Some(&'*') => Some(block_comment_end(&c, i)),
            '"' => Some(string_end(&c, i)),
            '\'' => char_end(&c, i),
            'r' if boundary => raw_string_end(&c, i),
            'b' | 'c' if boundary && c.get(i + 1) == Some(&'r') => raw_string_end(&c, i + 1),
            'b' | 'c' if boundary && c.get(i + 1) == Some(&'"') => Some(string_end(&c, i + 1)),
            _ => None,
        };
        match end {
            Some(end) => {
                blank_range(&mut out, &c, i, end);
                i = end.max(i + 1);
            }
            None => i += 1,
        }
    }
    out
}

fn skip_ws(c: &[char], mut j: usize) -> usize {
    while j < c.len() && c[j].is_whitespace() {
        j += 1;
    }
    j
}

fn ident_at(c: &[char], j: usize) -> &[char] {
    let len = c[j..].iter().take_while(|&&x| is_ident(x)).count();
    &c[j..j + len]
}

fn is_function(ident: &[char]) -> Option<&'static str> {
    FUNCTIONS
        .into_iter()
        .find(|f| f.chars().eq(ident.iter().copied()))
}

/// 2026-10-17: Every read in `src`, in source order.
pub fn reads(src: &str) -> Vec<Read> {
    let c = blank(src);
    let line_of = |pos: usize| 1 + c[..pos].iter().filter(|&&x| x == '\n').count();
    let mut found = Vec::new();
    for i in 0..c.len() {
        if i > 0 && is_ident(c[i - 1]) {
            continue;
        }
        let ident = ident_at(&c, i);
        let Some(module) = MODULES
            .into_iter()
            .find(|m| m.chars().eq(ident.iter().copied()))
        else {
            continue;
        };
        let j = skip_ws(&c, i + module.len());
        if c.get(j) != Some(&':') || c.get(j + 1) != Some(&':') {
            continue;
        }
        let j = skip_ws(&c, j + 2);
        if c.get(j) == Some(&'{') {
            brace_list(&c, j, module, &line_of, &mut found);
        } else if let Some(f) = c.get(j).and_then(|_| is_function(ident_at(&c, j))) {
            found.push(Read {
                line: line_of(j),
                call: format!("{module}::{f}"),
            });
        }
    }
    found
}

/// 2026-10-17: Reads named directly in a brace list opening at `open` (`env::{self, var}`); a
/// nested path such as `env::{consts::var}` is not a direct item and is not counted.
fn brace_list(
    c: &[char],
    open: usize,
    module: &str,
    line_of: &impl Fn(usize) -> usize,
    found: &mut Vec<Read>,
) {
    let (mut j, mut depth, mut item_start) = (open + 1, 1usize, true);
    while j < c.len() && depth > 0 {
        match c[j] {
            '{' => depth += 1,
            '}' => depth -= 1,
            ',' if depth == 1 => item_start = true,
            x if x.is_whitespace() => {}
            x if is_ident(x) => {
                let ident = ident_at(c, j);
                let after = skip_ws(c, j + ident.len());
                let ends_item = matches!(c.get(after), Some(',' | '}'))
                    || (after < c.len() && ident_at(c, after) == ['a', 's']);
                if let Some(f) =
                    is_function(ident).filter(|_| item_start && depth == 1 && ends_item)
                {
                    found.push(Read {
                        line: line_of(j),
                        call: format!("{module}::{f}"),
                    });
                }
                item_start = false;
                j += ident.len();
                continue;
            }
            _ => item_start = false,
        }
        j += 1;
    }
}

#[cfg(test)]
#[path = "lex_tests.rs"]
mod tests;
