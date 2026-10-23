// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The already-read source set the derive is a pure function of (SBIO): the host reads
//! every path the config lists (and the files of the memory directory) into a [`SourceSet`].
//!
//! Owner: metrale-kg-derive.
//! Invariants:
//! - Paths are root-relative and `/`-separated; a `BTreeMap` keeps the iteration order independent
//!   of the order the host read them in.
//! - A credential-named file is `withheld`: listed by path, never read.

use crate::bundle::SourceDigest;
use crate::hash::sha256;
use crate::prelude::*;
use crate::secrets::name_is_secret;

/// 2026-10-23: The files one derive reads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SourceSet {
    /// 2026-10-23: root-relative path → text.
    pub files: BTreeMap<String, String>,
    /// 2026-10-23: Credential-named files present but never read (root-relative paths).
    pub withheld: BTreeSet<String>,
    /// 2026-10-23: root-relative path → the commit the file is at, when the host knows it.
    pub git: BTreeMap<String, String>,
}

impl SourceSet {
    /// 2026-10-23: Add a file the host read; a credential-named one is withheld instead.
    pub fn add(&mut self, path: &str, text: String) {
        if name_is_secret(path) {
            self.withheld.insert(path.to_string());
        } else {
            self.files.insert(path.to_string(), text);
        }
    }

    /// 2026-10-23: The text of `path`.
    pub fn get(&self, path: &str) -> Option<&str> {
        self.files.get(path).map(String::as_str)
    }

    /// 2026-10-23: The files and withheld names directly under directory `dir`, sorted.
    pub fn under<'a>(&'a self, dir: &'a str) -> impl Iterator<Item = (&'a str, Option<&'a str>)> {
        let prefix = format!("{}/", dir.trim_end_matches('/'));
        let mut all: Vec<(&str, Option<&str>)> = self
            .files
            .iter()
            .map(|(p, t)| (p.as_str(), Some(t.as_str())))
            .chain(self.withheld.iter().map(|p| (p.as_str(), None)))
            .filter(|(p, _)| {
                p.strip_prefix(prefix.as_str())
                    .is_some_and(|r| !r.contains('/'))
            })
            .collect();
        all.sort_by(|a, b| a.0.cmp(b.0));
        all.into_iter()
    }

    /// 2026-10-23: Every file's digest (a withheld file's covers its name only).
    pub fn digests(&self) -> Vec<SourceDigest> {
        let mut out: Vec<SourceDigest> = self
            .files
            .iter()
            .map(|(p, t)| SourceDigest {
                path: p.clone(),
                sha: sha256(t.as_bytes()),
                git: self.git.get(p).cloned(),
            })
            .collect();
        out.extend(self.withheld.iter().map(|p| SourceDigest {
            path: p.clone(),
            sha: sha256(format!("withheld:{p}").as_bytes()),
            git: None,
        }));
        out.sort();
        out
    }
}
