// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Module targets of a declared precision plan and how they match.
//!
//! Owner: config (quantization).
//! Invariants: none beyond the types.

use anyhow::{Context, Result};

/// 2026-09-28: One target or ignore entry.
#[derive(Debug, Clone)]
pub enum Target {
    /// 2026-09-28: An exact module path.
    Exact(String),
    /// 2026-09-28: A compressed-tensors `re:` pattern, anchored at both ends as Python's
    /// `re.match` plus the checkpoint's `$` do.
    Regex(String, regex::Regex),
    /// 2026-09-28: A ModelOpt glob (`*`, `?`).
    Glob(String),
    /// 2026-09-28: An HF `modules_to_not_convert` entry: matches any module path that
    /// contains it.
    Substring(String),
    /// 2026-09-28: A module class (`Linear`). Every module the plan is asked about is a
    /// linear layer, so `Linear` matches all of them and any other class none.
    Class(String),
}

impl Target {
    /// 2026-09-28: A compressed-tensors target or ignore entry: `re:<pattern>`, a class
    /// name (`Linear`), or an exact path.
    pub fn compressed_tensors(s: &str) -> Result<Self> {
        if let Some(p) = s.strip_prefix("re:") {
            let re = regex::Regex::new(&format!("^(?:{p})"))
                .with_context(|| format!("quantization_config: bad target pattern {s:?}"))?;
            return Ok(Target::Regex(s.to_string(), re));
        }
        if !s.contains('.') && s.chars().next().is_some_and(char::is_uppercase) {
            return Ok(Target::Class(s.to_string()));
        }
        Ok(Target::Exact(s.to_string()))
    }

    /// 2026-09-28: A ModelOpt entry: a glob when it has a wildcard, else exact.
    pub fn modelopt(s: &str) -> Self {
        if s.contains(['*', '?']) {
            Target::Glob(s.to_string())
        } else if s == "Linear" {
            Target::Class(s.to_string())
        } else {
            Target::Exact(s.to_string())
        }
    }

    /// 2026-09-28: The entry as the checkpoint wrote it.
    pub fn text(&self) -> &str {
        match self {
            Target::Exact(s)
            | Target::Regex(s, _)
            | Target::Glob(s)
            | Target::Substring(s)
            | Target::Class(s) => s,
        }
    }

    /// 2026-09-28: Does this entry name `module` (classes excluded)?
    pub fn matches_name(&self, module: &str) -> bool {
        match self {
            Target::Exact(s) => s == module,
            Target::Regex(_, re) => re.is_match(module),
            Target::Glob(g) => glob_match(g.as_bytes(), module.as_bytes()),
            Target::Substring(s) => module.contains(s.as_str()),
            Target::Class(c) => c == "Linear",
        }
    }

    /// 2026-09-28: Precedence tier when this entry matches `module`: 0 exact name, 1
    /// pattern, 2 class. `None` when it does not match.
    pub fn tier_for(&self, module: &str) -> Option<u8> {
        if !self.matches_name(module) {
            return None;
        }
        Some(match self {
            Target::Exact(_) => 0,
            Target::Regex(..) | Target::Glob(_) | Target::Substring(_) => 1,
            Target::Class(_) => 2,
        })
    }
}

/// 2026-09-28: `*` matches any run, `?` one byte; everything else literally.
fn glob_match(p: &[u8], s: &[u8]) -> bool {
    let (mut pi, mut si) = (0usize, 0usize);
    let (mut star, mut mark) = (None, 0usize);
    while si < s.len() {
        if pi < p.len() && (p[pi] == b'?' || p[pi] == s[si]) {
            pi += 1;
            si += 1;
        } else if pi < p.len() && p[pi] == b'*' {
            star = Some(pi);
            mark = si;
            pi += 1;
        } else if let Some(sp) = star {
            pi = sp + 1;
            mark += 1;
            si = mark;
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|&c| c == b'*')
}
