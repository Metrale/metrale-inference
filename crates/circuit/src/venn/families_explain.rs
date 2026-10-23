// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: A family's explanation (`explain.about`, `explain.numerics`, `explain.math` in
//! KERNEL_FAMILIES.toml): the prose and the LaTeX of what its kernels compute, the one source the
//! blueprint exports carry to the store's kernel modal.
//!
//! Owner: metrale-circuit (venn).
//! Invariants:
//! - Every family states all three; an empty one is a load error (the gate that every family has
//!   math is this loader, so the planner, the exports and every test refuse a manifest without it).
//! - `math` is a list of display equations in KaTeX syntax; `about` and `numerics` are plain prose
//!   with inline math between `$` pairs. The checks here are the structural ones a Rust loader can
//!   make (balanced `$`, braces and environments, ASCII math); whether KaTeX in strict mode
//!   renders every entry is the `kernel_math` test's (crates/circuit/tests/kernel_math.rs).
//! - The text is published: it names no source file, path or code (refused below).

use crate::prelude::*;

use serde::Deserialize;

/// 2026-10-22: Text the store publishes, so never a source path or code.
const PRIVATE: [&str; 9] = [
    ".cu",
    ".cuh",
    ".rs",
    ".ptx",
    "kernels/",
    "crates/",
    "/workspace",
    "#include",
    "__global__",
];

/// 2026-10-22: A family's explanation.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Explain {
    /// 2026-10-22: What the kernels compute and how a launch is organised.
    pub about: String,
    /// 2026-10-22: Formats, accumulation, reduction and rounding.
    pub numerics: String,
    /// 2026-10-22: Display equations (KaTeX).
    pub math: Vec<String>,
}

/// 2026-10-22: The `explain` table of one family.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ExplainFile {
    about: String,
    numerics: String,
    math: Vec<String>,
}

impl ExplainFile {
    /// 2026-10-22: The checked explanation, or why it is refused.
    pub(super) fn validate(self) -> Result<Explain, String> {
        let about = prose("explain.about", &self.about)?;
        let numerics = prose("explain.numerics", &self.numerics)?;
        if self.math.is_empty() {
            return Err("explain.math is empty: every family states its math".into());
        }
        for (i, m) in self.math.iter().enumerate() {
            tex(&format!("explain.math[{i}]"), m)?;
            if m.contains('$') {
                return Err(format!("explain.math[{i}] is display math: no `$`"));
            }
        }
        Ok(Explain {
            about,
            numerics,
            math: self.math.iter().map(|m| m.trim().to_string()).collect(),
        })
    }
}

/// 2026-10-22: Prose with inline math: non-empty, whitespace collapsed, `$` paired.
pub(crate) fn prose(key: &str, text: &str) -> Result<String, String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Err(format!("{key} is empty"));
    }
    let text = words.join(" ");
    let parts: Vec<&str> = text.split('$').collect();
    if parts.len().is_multiple_of(2) {
        return Err(format!("{key}: an unpaired `$`"));
    }
    for (i, p) in parts.iter().enumerate() {
        if i % 2 == 1 {
            tex(key, p)?;
        } else {
            public(key, p)?;
        }
    }
    Ok(text)
}

/// 2026-10-22: One LaTeX string: non-empty, ASCII (KaTeX strict refuses Unicode text in math),
/// balanced braces and `\begin`/`\end`, and nothing private.
pub(crate) fn tex(key: &str, s: &str) -> Result<(), String> {
    let s = s.trim();
    if s.is_empty() {
        return Err(format!("{key}: empty math"));
    }
    if let Some(c) = s.chars().find(|c| !c.is_ascii()) {
        return Err(format!("{key}: non-ASCII `{c}` in math"));
    }
    let mut depth = 0i64;
    let mut escaped = false;
    for c in s.chars() {
        match (escaped, c) {
            (false, '\\') => escaped = true,
            (false, '{') => depth += 1,
            (false, '}') => depth -= 1,
            _ => escaped = false,
        }
        if depth < 0 {
            return Err(format!("{key}: `}}` without `{{` in `{s}`"));
        }
    }
    if depth != 0 {
        return Err(format!("{key}: unbalanced braces in `{s}`"));
    }
    if s.matches("\\begin{").count() != s.matches("\\end{").count() {
        return Err(format!("{key}: unbalanced \\begin/\\end in `{s}`"));
    }
    public(key, s)
}

fn public(key: &str, s: &str) -> Result<(), String> {
    match PRIVATE.iter().find(|p| s.contains(*p)) {
        Some(p) => Err(format!(
            "{key} names `{p}`: the text is published, so it names no source or code"
        )),
        None => Ok(()),
    }
}

#[cfg(test)]
#[path = "families_explain_tests.rs"]
mod families_explain_tests;
