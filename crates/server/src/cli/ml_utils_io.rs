// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The I/O side of `met ml-utils` and `met serve --mock`: checkpoint sources over a
//! local directory and over the Hub (metadata and safetensors headers only, never tensor data),
//! and a directory sink that publishes a checkpoint only once it is complete.
//!
//! Owner: server CLI.
//! Invariants:
//! - A header is read through an 8-byte length capped at 64 MiB, then exactly that many bytes.
//! - The sink writes into `<out>.partial` and renames it to `<out>` in [`FsSink::commit`]; it
//!   refuses an `<out>` that exists, so nothing is ever overwritten.

use std::io::{Read as _, Seek as _, SeekFrom, Write as _};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use metrale_ml_utils::io::{CheckpointSink, CheckpointSource};
use metrale_ml_utils::{MlError, Result as MlResult};

use crate::model_download::hf;

/// 2026-10-03: The largest safetensors header read (the engine's loaders use the same cap).
const HEADER_CAP: u64 = 64 << 20;

fn io<E: std::fmt::Display>(what: &str) -> impl FnOnce(E) -> MlError + '_ {
    move |e| MlError::Io(format!("{what}: {e}"))
}

/// 2026-10-03: A checkpoint directory (a local copy or a Hub cache snapshot).
pub(crate) struct FsCheckpoint {
    dir: PathBuf,
}

impl FsCheckpoint {
    pub(crate) fn new(dir: &Path) -> Self {
        FsCheckpoint {
            dir: dir.to_path_buf(),
        }
    }
}

impl CheckpointSource for FsCheckpoint {
    fn files(&self) -> MlResult<Vec<String>> {
        let mut out = Vec::new();
        for e in std::fs::read_dir(&self.dir).map_err(io(&self.dir.display().to_string()))? {
            let e = e.map_err(io("read_dir"))?;
            if let Some(n) = e.file_name().to_str() {
                out.push(n.to_string());
            }
        }
        out.sort();
        Ok(out)
    }

    fn read(&self, rel: &str) -> MlResult<Vec<u8>> {
        let p = self.dir.join(rel);
        std::fs::read(&p).map_err(io(&p.display().to_string()))
    }

    fn read_range(&self, rel: &str, offset: u64, len: u64) -> MlResult<Vec<u8>> {
        let p = self.dir.join(rel);
        let what = p.display().to_string();
        let mut f = std::fs::File::open(&p).map_err(io(&what))?;
        f.seek(SeekFrom::Start(offset)).map_err(io(&what))?;
        let mut buf = vec![0u8; len as usize];
        f.read_exact(&mut buf).map_err(io(&what))?;
        Ok(buf)
    }

    fn shard_headers(&self) -> MlResult<Vec<(String, Vec<u8>)>> {
        let mut out = Vec::new();
        for name in self.files()? {
            if !name.ends_with(".safetensors") {
                continue;
            }
            let p = self.dir.join(&name);
            let what = p.display().to_string();
            let mut f = std::fs::File::open(&p).map_err(io(&what))?;
            let mut len = [0u8; 8];
            f.read_exact(&mut len).map_err(io(&what))?;
            let n = u64::from_le_bytes(len);
            if n > HEADER_CAP {
                return Err(MlError::Io(format!("{what}: a {n}-byte header")));
            }
            let mut h = vec![0u8; n as usize];
            f.read_exact(&mut h).map_err(io(&what))?;
            out.push((name, h));
        }
        Ok(out)
    }
}

/// 2026-10-03: A checkpoint on the Hub at one revision, read through ranged requests.
pub(crate) struct HubCheckpoint {
    repo: String,
    revision: String,
    files: Vec<String>,
    token: Option<String>,
}

impl HubCheckpoint {
    /// 2026-10-03: The repo's current revision and file list.
    pub(crate) fn open(repo: &str) -> Result<Self> {
        let token = hf::token();
        let (revision, files) = hf::repo_info(repo, token.as_deref())
            .map_err(|e| anyhow::anyhow!("{repo}: {}", e.hint()))?;
        Ok(HubCheckpoint {
            repo: repo.to_string(),
            revision,
            files: files.into_iter().map(|f| f.name).collect(),
            token,
        })
    }

    /// 2026-10-03: The revision read.
    pub(crate) fn revision(&self) -> &str {
        &self.revision
    }

    fn fetch(&self, name: &str, span: Option<(u64, u64)>) -> MlResult<Vec<u8>> {
        hf::fetch_bytes(
            &self.repo,
            &self.revision,
            name,
            span,
            self.token.as_deref(),
        )
        .map_err(|e| MlError::Io(format!("{}/{name}: {}", self.repo, e.hint())))
    }
}

impl CheckpointSource for HubCheckpoint {
    fn files(&self) -> MlResult<Vec<String>> {
        Ok(self.files.clone())
    }

    fn read(&self, rel: &str) -> MlResult<Vec<u8>> {
        self.fetch(rel, None)
    }

    fn read_range(&self, rel: &str, offset: u64, len: u64) -> MlResult<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
        }
        self.fetch(rel, Some((offset, offset + len - 1)))
    }

    fn shard_headers(&self) -> MlResult<Vec<(String, Vec<u8>)>> {
        let mut out = Vec::new();
        for name in self.files.iter().filter(|f| f.ends_with(".safetensors")) {
            let len = self.fetch(name, Some((0, 7)))?;
            let len: [u8; 8] = len
                .as_slice()
                .try_into()
                .map_err(|_| MlError::Io(format!("{name}: {} length bytes", len.len())))?;
            let n = u64::from_le_bytes(len);
            if n == 0 || n > HEADER_CAP {
                return Err(MlError::Io(format!("{name}: a {n}-byte header")));
            }
            out.push((name.clone(), self.fetch(name, Some((8, 8 + n - 1)))?));
        }
        Ok(out)
    }
}

/// 2026-10-03: A checkpoint source for `spec` (a directory, a cached Hub id, or, with
/// `allow_network`, a Hub id), with the id and revision a resolved spec records.
pub(crate) struct OpenedSource {
    pub source: Box<dyn CheckpointSource>,
    pub id: String,
    pub revision: Option<String>,
    /// 2026-10-03: The local directory, when the source is one.
    pub dir: Option<PathBuf>,
}

/// 2026-10-03: Open `spec`: a directory holding `config.json`, a Hub id cached locally, or a Hub
/// id read over the network when `allow_network` says so.
pub(crate) fn open_source(
    spec: &str,
    cache_dir: Option<&Path>,
    allow_network: bool,
) -> Result<OpenedSource> {
    let local = crate::model_resolver::resolve_model_dir(spec, cache_dir).ok();
    if let Some(dir) = local {
        let id = metrale_circuit::venn::checkpoint_id_of(&dir.to_string_lossy())
            .unwrap_or_else(|| spec.to_string());
        let revision = dir
            .parent()
            .filter(|p| p.ends_with("snapshots"))
            .and_then(|_| dir.file_name())
            .map(|n| n.to_string_lossy().into_owned());
        return Ok(OpenedSource {
            source: Box::new(FsCheckpoint::new(&dir)),
            id,
            revision,
            dir: Some(dir),
        });
    }
    if !spec.contains('/') || Path::new(spec).exists() {
        bail!("{spec}: not a checkpoint directory and not a Hub id (org/name)");
    }
    if !allow_network {
        bail!(
            "{spec} is not in the local cache; pass --allow-network to read its metadata from the Hub"
        );
    }
    let hub = HubCheckpoint::open(spec)?;
    let revision = Some(hub.revision().to_string());
    Ok(OpenedSource {
        source: Box::new(hub),
        id: spec.to_string(),
        revision,
        dir: None,
    })
}

/// 2026-10-03: A sink into a new directory, published by [`FsSink::commit`].
pub(crate) struct FsSink {
    out: PathBuf,
    tmp: PathBuf,
    open: Option<(PathBuf, PathBuf, std::fs::File)>,
}

impl FsSink {
    /// 2026-10-03: A sink for `out`, which must not exist yet.
    pub(crate) fn create(out: &Path) -> Result<Self> {
        if out.exists() {
            bail!(
                "{} exists; a mock is never written over anything",
                out.display()
            );
        }
        let mut tmp = out.as_os_str().to_owned();
        tmp.push(".partial");
        let tmp = PathBuf::from(tmp);
        if tmp.exists() {
            std::fs::remove_dir_all(&tmp)
                .with_context(|| format!("removing the stale {}", tmp.display()))?;
        }
        std::fs::create_dir_all(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        Ok(FsSink {
            out: out.to_path_buf(),
            tmp,
            open: None,
        })
    }

    /// 2026-10-03: Publish the written directory under its final name.
    pub(crate) fn commit(self) -> Result<PathBuf> {
        if self.open.is_some() {
            bail!("a shard is still open");
        }
        std::fs::rename(&self.tmp, &self.out).with_context(|| {
            format!("renaming {} to {}", self.tmp.display(), self.out.display())
        })?;
        Ok(self.out)
    }
}

impl CheckpointSink for FsSink {
    fn write_file(&mut self, rel: &str, bytes: &[u8]) -> MlResult<()> {
        let p = self.tmp.join(rel);
        std::fs::write(&p, bytes).map_err(io(&p.display().to_string()))
    }

    fn begin_shard(&mut self, rel: &str, header: &[u8]) -> MlResult<()> {
        if self.open.is_some() {
            return Err(MlError::Io("a shard is already open".into()));
        }
        let fin = self.tmp.join(rel);
        let part = self.tmp.join(format!("{rel}.part"));
        let mut f = std::fs::File::create(&part).map_err(io(&part.display().to_string()))?;
        f.write_all(header).map_err(io(rel))?;
        self.open = Some((part, fin, f));
        Ok(())
    }

    fn append(&mut self, bytes: &[u8]) -> MlResult<()> {
        let (_, _, f) = self
            .open
            .as_mut()
            .ok_or_else(|| MlError::Io("no open shard".into()))?;
        f.write_all(bytes).map_err(io("shard"))
    }

    fn skip(&mut self, len: u64) -> MlResult<()> {
        let (_, _, f) = self
            .open
            .as_mut()
            .ok_or_else(|| MlError::Io("no open shard".into()))?;
        let end = f.stream_position().map_err(io("shard"))? + len;
        f.set_len(end).map_err(io("shard"))?;
        f.seek(SeekFrom::Start(end)).map_err(io("shard"))?;
        Ok(())
    }

    fn end_shard(&mut self) -> MlResult<()> {
        let (part, fin, mut f) = self
            .open
            .take()
            .ok_or_else(|| MlError::Io("no open shard".into()))?;
        f.flush().map_err(io("shard"))?;
        drop(f);
        std::fs::rename(&part, &fin).map_err(io(&fin.display().to_string()))
    }
}

#[cfg(test)]
#[path = "ml_utils_io_tests.rs"]
mod ml_utils_io_tests;
