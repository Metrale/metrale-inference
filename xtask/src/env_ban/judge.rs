// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The env-ban verdict: which scanned files may read the environment, and whether
//! the allowlist is well-formed and current. Pure: the scanned files and the allowlist text in,
//! violations out.
//!
//! The allowlist names one file per line, sorted and unique, with no line numbers. It can only
//! shrink: a listed file that no longer reads the environment (or no longer exists, or needs
//! no entry) is itself a violation, so a fix must delete its entry, and a new entry needs a
//! CODEOWNERS review.

use super::lex::Read;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// 2026-10-17: Why a file may read the environment without an allowlist entry, or may not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// 2026-10-17: Needs an allowlist entry to read the environment.
    Ordinary,
    /// 2026-10-17: `build.rs` or a `build_*.rs` helper beside a `Cargo.toml`.
    BuildScript,
    /// 2026-10-17: The `ProcessEnv` host, the one sanctioned reader.
    HostEnv,
}

/// 2026-10-17: One scanned file: its repository-relative path (`/`-separated) and its reads.
pub struct Source {
    pub path: String,
    pub kind: Kind,
    pub reads: Vec<Read>,
}

/// 2026-10-17: Why a listed file's entry must go.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stale {
    Missing,
    NoRead,
    Exempt,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Violation {
    /// 2026-10-17: A read in a file that is not exempt and not listed.
    Unlisted { path: String, read: Read },
    /// 2026-10-17: An entry that is not a plain relative `.rs` path (`line` is the list line).
    Malformed { line: usize, entry: String },
    /// 2026-10-17: An entry not strictly after the one before it (unsorted or duplicate).
    Unsorted { line: usize, entry: String },
    /// 2026-10-17: An entry whose file does not need one.
    Stale {
        line: usize,
        entry: String,
        why: Stale,
    },
}

impl Violation {
    /// 2026-10-17: A comparable identity for the selftest: kind, path, line.
    pub fn key(&self) -> (&'static str, &str, usize) {
        match self {
            Self::Unlisted { path, read } => ("unlisted", path, read.line),
            Self::Malformed { line, entry } => ("malformed", entry, *line),
            Self::Unsorted { line, entry } => ("unsorted", entry, *line),
            Self::Stale { line, entry, why } => match why {
                Stale::Missing => ("stale-missing", entry, *line),
                Stale::NoRead => ("stale-no-read", entry, *line),
                Stale::Exempt => ("stale-exempt", entry, *line),
            },
        }
    }
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let list = super::ALLOWLIST;
        match self {
            Self::Unlisted { path, read } => write!(
                f,
                "{path}:{}: `{}` reads the process environment; read it through the Env port \
                 or the settings resolver (only {} and build scripts may call it; {list} lists \
                 the files not yet moved)",
                read.line,
                read.call,
                super::HOST_ENV
            ),
            Self::Malformed { line, entry } => write!(
                f,
                "{list}:{line}: `{entry}` is not a relative `.rs` path without a line number"
            ),
            Self::Unsorted { line, entry } => write!(
                f,
                "{list}:{line}: `{entry}` is not after the entry before it (keep the list sorted \
                 and unique)"
            ),
            Self::Stale { line, entry, why } => {
                let reason = match why {
                    Stale::Missing => "the file does not exist",
                    Stale::NoRead => "the file no longer reads the environment",
                    Stale::Exempt => "the file is exempt and needs no entry",
                };
                write!(f, "{list}:{line}: remove `{entry}`: {reason}")
            }
        }
    }
}

fn well_formed(entry: &str) -> bool {
    !entry.starts_with('/')
        && entry.ends_with(".rs")
        && !entry.contains([':', '\\', ' '])
        && entry
            .split('/')
            .all(|seg| !seg.is_empty() && seg != "." && seg != "..")
}

/// 2026-10-17: The entries of an allowlist: `(1-based line, trimmed entry)`, without blank
/// lines and `#` comments.
pub fn entries(allowlist: &str) -> impl Iterator<Item = (usize, &str)> {
    allowlist
        .lines()
        .enumerate()
        .map(|(idx, raw)| (idx + 1, raw.trim()))
        .filter(|(_, entry)| !entry.is_empty() && !entry.starts_with('#'))
}

/// 2026-10-17: Every violation in `sources` under `allowlist`, sorted.
pub fn judge(sources: &[Source], allowlist: &str) -> Vec<Violation> {
    let mut out = Vec::new();
    let by_path: BTreeMap<&str, &Source> = sources.iter().map(|s| (s.path.as_str(), s)).collect();
    let mut listed = BTreeSet::new();
    let mut prev: Option<&str> = None;
    for (line, entry) in entries(allowlist) {
        let owned = entry.to_owned();
        if !well_formed(entry) {
            out.push(Violation::Malformed { line, entry: owned });
            continue;
        }
        if prev.is_some_and(|p| entry <= p) {
            out.push(Violation::Unsorted {
                line,
                entry: owned.clone(),
            });
        }
        prev = Some(prev.map_or(entry, |p| p.max(entry)));
        listed.insert(entry);
        let why = match by_path.get(entry) {
            None => Some(Stale::Missing),
            Some(s) if s.kind != Kind::Ordinary => Some(Stale::Exempt),
            Some(s) if s.reads.is_empty() => Some(Stale::NoRead),
            Some(_) => None,
        };
        if let Some(why) = why {
            out.push(Violation::Stale {
                line,
                entry: owned,
                why,
            });
        }
    }
    for s in sources {
        if s.kind == Kind::Ordinary && !listed.contains(s.path.as_str()) {
            out.extend(s.reads.iter().map(|read| Violation::Unlisted {
                path: s.path.clone(),
                read: read.clone(),
            }));
        }
    }
    out.sort();
    out
}
