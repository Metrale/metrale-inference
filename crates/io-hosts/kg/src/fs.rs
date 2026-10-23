// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The configured sources on disk: [`FsSources::read`] fills a `SourceSet`,
//! [`FsSources::stamp`] is the per-call freshness check (path, size, mtime) the design asks for
//! (§4.1 invariant 5) without reading any file.
//!
//! Owner: metrale-io-host-kg.
//! Invariants:
//! - Credential-named memory files (`secrets::name_is_secret`) are stamped and listed, never read.
//! - A configured file that is missing is `KgSource`, never skipped.

use metrale_error::{ErrorCode, Result, error};
use metrale_kg_derive::lkb_derive::LkbSources;
use metrale_kg_derive::secrets::name_is_secret;
use metrale_kg_derive::{Config, SourceSet};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

/// 2026-10-23: One file's stat: root-relative path, size, mtime (ns since the epoch).
pub type Stamp = Vec<(String, u64, u128)>;

/// 2026-10-23: The file-system host over `config.sources.root`.
#[derive(Clone, Debug)]
pub struct FsSources {
    pub root: PathBuf,
}

fn source_err(path: &str, e: &std::io::Error) -> metrale_error::Error {
    error!(ErrorCode::KgSource, path.to_string(), e.to_string())
}

impl FsSources {
    /// 2026-10-23: The host for `cfg`.
    pub fn new(cfg: &Config) -> Self {
        FsSources {
            root: PathBuf::from(&cfg.sources.root),
        }
    }

    fn abs(&self, rel: &str) -> PathBuf {
        self.root.join(rel)
    }

    /// 2026-10-23: The memory directory's `.md` files, root-relative, sorted.
    pub fn memory_files(&self, cfg: &Config) -> Result<Vec<String>> {
        let dir = &cfg.sources.memory_dir;
        let rd = std::fs::read_dir(self.abs(dir)).map_err(|e| source_err(dir, &e))?;
        let mut out = Vec::new();
        for entry in rd {
            let entry = entry.map_err(|e| source_err(dir, &e))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.ends_with(".md") && entry.path().is_file() {
                out.push(format!("{}/{name}", dir.trim_end_matches('/')));
            }
        }
        out.sort();
        Ok(out)
    }

    /// 2026-10-23: The LKB files of `[sources.lkb]` (root-relative): the five manifests and the
    /// top-level `.toml` circuits, `.txt` plans and `.json` headroom reports of its directories.
    pub fn lkb_files(&self, cfg: &Config) -> Result<Vec<String>> {
        let Some(t) = &cfg.sources.lkb else {
            return Ok(Vec::new());
        };
        let l = LkbSources::parse(t)?;
        let mut out: Vec<String> = l.files().to_vec();
        for (dir, ext) in l.dirs().into_iter().zip([".toml", ".txt", ".json"]) {
            let rd = std::fs::read_dir(self.abs(&dir)).map_err(|e| source_err(&dir, &e))?;
            for entry in rd {
                let entry = entry.map_err(|e| source_err(&dir, &e))?;
                let name = entry.file_name().to_string_lossy().into_owned();
                if name.ends_with(ext) && entry.path().is_file() {
                    out.push(format!("{dir}/{name}"));
                }
            }
        }
        out.sort();
        Ok(out)
    }

    fn all_paths(&self, cfg: &Config) -> Result<Vec<String>> {
        let mut v = cfg.source_paths();
        v.extend(self.memory_files(cfg)?);
        v.extend(self.lkb_files(cfg)?);
        Ok(v)
    }

    /// 2026-10-23: Read every configured source (credential-named files are withheld unread).
    pub fn read(&self, cfg: &Config) -> Result<SourceSet> {
        let mut set = SourceSet::default();
        for rel in self.all_paths(cfg)? {
            if name_is_secret(&rel) {
                set.withheld.insert(rel);
                continue;
            }
            let text = std::fs::read_to_string(self.abs(&rel)).map_err(|e| source_err(&rel, &e))?;
            set.add(&rel, text);
        }
        Ok(set)
    }

    /// 2026-10-23: The freshness stamp of every source (no file is opened).
    pub fn stamp(&self, cfg: &Config) -> Result<Stamp> {
        let mut out = Vec::new();
        for rel in self.all_paths(cfg)? {
            let md = std::fs::metadata(self.abs(&rel)).map_err(|e| source_err(&rel, &e))?;
            let mtime = md
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos());
            out.push((rel, md.len(), mtime));
        }
        if let Some(t) = &cfg.sources.lkb {
            out.push(crate::lkb::repo_stamp(&self.root, &LkbSources::parse(t)?)?);
        }
        Ok(out)
    }
}

/// 2026-10-23: Read the config file at `path` and parse it (a relative path is resolved, so
/// `--config config.toml` and the absolute path name the same project layer).
pub fn read_config(path: &Path) -> Result<Config> {
    let shown = path.display().to_string();
    let path = std::fs::canonicalize(path).map_err(|e| source_err(&shown, &e))?;
    let text = std::fs::read_to_string(&path).map_err(|e| source_err(&shown, &e))?;
    let mut cfg = Config::parse(&text)?;
    let root = std::fs::canonicalize(&cfg.sources.root)
        .map_err(|e| source_err(&cfg.sources.root, &e))?;
    let rel = path.strip_prefix(&root).map_err(|_| {
        error!(
            ErrorCode::KgConfig,
            shown.clone(),
            "must lie under sources.root (it ingests as a node)"
        )
    })?;
    cfg.origin = rel.to_string_lossy().into_owned();
    Ok(cfg)
}
