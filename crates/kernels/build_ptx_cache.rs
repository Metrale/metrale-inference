// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The opt-in content-hashed PTX cache: the I/O shell around
//! `metrale_closure::ptx_cache`, which owns the key, the sidecar check and the
//! eviction choice.
//!
//! Owner: metrale-kernels build.
//! Invariants:
//! - Off unless `METRALE_PTX_CACHE=1`. Off, nothing here runs and the build
//!   is the uncached one: no read, no write, no extra `cargo:` line beyond the
//!   `rerun-if-env-changed` set `emit_env_rerun` prints on every build.
//! - The key is derived from the very argv nvcc then runs with
//!   (`NvidiaTarget::argv`), so a key and its compile cannot disagree.
//! - Entries live in `<target-dir>/metrale-ptx-cache/v<schema>/<aa>/`, outside
//!   every `OUT_DIR`, so `build_stage::reset` cannot reach them.
//! - Writes are a same-directory tmp file plus a rename: the entry first, then
//!   its sidecar. A reader between the two sees a sidecar that does not vouch
//!   for the bytes, which is a miss.
//!
//! 2026-10-17: A cache I/O error is a hard error, reported with the other
//! compile errors once every job has run. The cache is a developer opt-in, so
//! a loud failure costs one rebuild with it unset, while a silent fall-through
//! would hide a cache that never hits, or one that keeps failing a check.

use std::collections::{BTreeSet, HashMap};
use std::ffi::OsString;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use metrale_closure::ptx_cache::{self, FileFact, KeyInputs};

use super::build_target::{ComputeTarget, NvidiaTarget};

/// 2026-10-17: `1` turns the cache on; unset or `0` leaves it off.
const CACHE_SWITCH: &str = "METRALE_PTX_CACHE";
/// 2026-10-17: The LRU cap in bytes. Required while the cache is on.
const CACHE_CAP: &str = "METRALE_PTX_CACHE_MAX_BYTES";
/// 2026-10-17: The `NVCC_*` variables nvcc documents. Each gets a
/// `rerun-if-env-changed` even while unset, so setting one reruns the build;
/// any other `NVCC_*` is watched only once it is present.
const NVCC_ENV_KNOWN: &[&str] = &["NVCC_APPEND_FLAGS", "NVCC_CCBIN", "NVCC_PREPEND_FLAGS"];
/// 2026-10-17: A tmp file or sidecar without its entry, older than this, is
/// debris from a killed build and is removed. A younger one may belong to a
/// concurrent build sharing the target directory, so it is left alone.
const ORPHAN_AGE: Duration = Duration::from_secs(60 * 60);
/// 2026-10-17: The per-run record of every cached compile, under `OUT_DIR`.
const LOG_NAME: &str = "ptx-cache.log";

/// 2026-10-17: Print the `rerun-if-env-changed` lines for the cache switch,
/// the cap and the `NVCC_*` variables. Called on every build, cache on or
/// off: nvcc reads `NVCC_*` either way.
pub(super) fn emit_env_rerun() {
    println!("cargo:rerun-if-env-changed={CACHE_SWITCH}");
    println!("cargo:rerun-if-env-changed={CACHE_CAP}");
    let names: BTreeSet<String> = NVCC_ENV_KNOWN
        .iter()
        .map(|n| n.to_string())
        .chain(nvcc_vars().map(|(name, _)| name))
        .collect();
    for name in names {
        println!("cargo:rerun-if-env-changed={name}");
    }
}

fn nvcc_vars() -> impl Iterator<Item = (String, OsString)> {
    std::env::vars_os().filter_map(|(name, value)| {
        let name = name.into_string().ok()?;
        name.starts_with("NVCC_").then_some((name, value))
    })
}

fn env_utf8(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(v) => Some(v),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(v)) => panic!("{name} is not UTF-8: {v:?}"),
    }
}

/// 2026-10-17: One build's cache: its settings, read once, and what the
/// compile workers record for `finish`.
pub(super) struct PtxCache<'a> {
    nvcc: &'a NvidiaTarget,
    root: PathBuf,
    cap: u64,
    compiler: String,
    nvcc_env: Vec<(String, String)>,
    out_dir: PathBuf,
    started: SystemTime,
    facts: Mutex<HashMap<PathBuf, FileFact>>,
    log: Mutex<Vec<String>>,
    watched: Mutex<BTreeSet<PathBuf>>,
    tmp_seq: AtomicU64,
}

impl<'a> PtxCache<'a> {
    /// 2026-10-17: `None` when `METRALE_PTX_CACHE` is unset or `0`. Panics on
    /// any other switch value, a missing or invalid cap, a compiler other than
    /// nvcc, an `OUT_DIR` with no cargo target directory above it, or an nvcc
    /// whose identity cannot be read.
    pub(super) fn from_env(out_dir: &Path, compute: &'a dyn ComputeTarget) -> Option<Self> {
        let on = ptx_cache::parse_switch(env_utf8(CACHE_SWITCH).as_deref())
            .unwrap_or_else(|e| panic!("{e}"));
        if !on {
            return None;
        }
        let cap =
            ptx_cache::parse_cap(env_utf8(CACHE_CAP).as_deref()).unwrap_or_else(|e| panic!("{e}"));
        let nvcc = compute.as_nvidia().unwrap_or_else(|| {
            panic!("{CACHE_SWITCH}=1 caches nvcc PTX only, and this target does not use nvcc")
        });
        // 2026-10-17: Cargo writes CACHEDIR.TAG at the root of the target
        // directory, the one `OUT_DIR` ancestor every profile shares.
        let target_dir = out_dir
            .ancestors()
            .find(|p| p.join("CACHEDIR.TAG").is_file())
            .unwrap_or_else(|| {
                panic!(
                    "{CACHE_SWITCH}=1: no CACHEDIR.TAG above {}",
                    out_dir.display()
                )
            });
        let compiler = nvcc
            .identity()
            .unwrap_or_else(|e| panic!("{CACHE_SWITCH}=1: {e}"));
        let nvcc_env = nvcc_vars()
            .map(|(name, value)| match value.into_string() {
                Ok(v) => (name, v),
                Err(v) => panic!("{name} is not UTF-8: {v:?}"),
            })
            .collect();
        let out_dir = out_dir
            .canonicalize()
            .unwrap_or_else(|e| panic!("resolving {}: {e}", out_dir.display()));
        Some(Self {
            nvcc,
            root: target_dir.join(ptx_cache::PTX_CACHE_DIR),
            cap,
            compiler,
            nvcc_env,
            out_dir,
            started: SystemTime::now(),
            facts: Mutex::new(HashMap::new()),
            log: Mutex::new(Vec::new()),
            watched: Mutex::new(BTreeSet::new()),
            tmp_seq: AtomicU64::new(0),
        })
    }

    pub(super) fn root(&self) -> &Path {
        &self.root
    }

    /// 2026-10-17: `NvidiaTarget::compile` through the cache: copy the entry
    /// to `output` on a hit; on a miss run nvcc with the keyed argv and store
    /// its output.
    pub(super) fn compile(
        &self,
        source: &Path,
        output: &Path,
        arch: &str,
        extra_flags: &[String],
    ) -> Result<(), String> {
        let argv = self.nvcc.argv(source, output, arch, extra_flags);
        let key = self.key(&argv, source)?;
        let entry = self.root.join(ptx_cache::entry_rel(&key));
        let outcome = match self.fetch(&entry, output)? {
            Fetch::Hit => "hit",
            found => {
                self.nvcc.run(&argv, source)?;
                self.store(&entry, output)?;
                if matches!(found, Fetch::Rejected) {
                    "stale"
                } else {
                    "miss"
                }
            }
        };
        self.log
            .lock()
            .unwrap()
            .push(format!("{outcome} {key} {}", source.display()));
        Ok(())
    }

    fn key(&self, argv: &[String], source: &Path) -> Result<String, String> {
        let search = ptx_cache::search_dirs(argv, &self.nvcc_env);
        let closure = metrale_closure::include_closure(source, &search)
            .map_err(|e| self.failure("walk the include closure of", source, &e))?;
        let files = closure
            .files
            .iter()
            .map(|path| self.fact(path))
            .collect::<Result<Vec<_>, _>>()?;
        let unresolved: Vec<String> = closure
            .unresolved
            .iter()
            .map(|(from, include)| format!("{} -> {include}", from.display()))
            .collect();
        // 2026-10-17: The staged copies under OUT_DIR are rewritten every run,
        // so watching them would rerun every build; their originals in
        // kernels/ are already watched through the layer directories.
        self.watched.lock().unwrap().extend(
            closure
                .files
                .iter()
                .filter(|p| !p.starts_with(&self.out_dir))
                .cloned(),
        );
        Ok(ptx_cache::cache_key(&KeyInputs {
            compiler: &self.compiler,
            argv,
            nvcc_env: &self.nvcc_env,
            files: &files,
            unresolved: &unresolved,
        }))
    }

    /// 2026-10-17: One read per file per build: targets share staged copies.
    fn fact(&self, path: &Path) -> Result<FileFact, String> {
        if let Some(known) = self.facts.lock().unwrap().get(path) {
            return Ok(known.clone());
        }
        let bytes = std::fs::read(path).map_err(|e| self.failure("read", path, &e))?;
        let fact = FileFact::of(path, &bytes);
        self.facts
            .lock()
            .unwrap()
            .insert(path.to_path_buf(), fact.clone());
        Ok(fact)
    }

    fn fetch(&self, entry: &Path, output: &Path) -> Result<Fetch, String> {
        let bytes = match std::fs::read(entry) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Fetch::Absent),
            Err(e) => return Err(self.failure("read", entry, &e)),
        };
        let sidecar_path = ptx_cache::sidecar_of(entry);
        let sidecar = match std::fs::read_to_string(&sidecar_path) {
            Ok(text) => text,
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Fetch::Rejected),
            // 2026-10-17: Non-UTF-8 sidecar bytes cannot name a digest: a
            // mismatch, not an I/O failure.
            Err(e) if e.kind() == ErrorKind::InvalidData => return Ok(Fetch::Rejected),
            Err(e) => return Err(self.failure("read", &sidecar_path, &e)),
        };
        if !ptx_cache::verify(&bytes, &sidecar) {
            return Ok(Fetch::Rejected);
        }
        std::fs::write(output, &bytes).map_err(|e| self.failure("write", output, &e))?;
        // 2026-10-17: The entry's mtime is the LRU clock; a hit moves it to now.
        std::fs::File::options()
            .write(true)
            .open(entry)
            .and_then(|f| f.set_modified(SystemTime::now()))
            .map_err(|e| self.failure("touch", entry, &e))?;
        Ok(Fetch::Hit)
    }

    fn store(&self, entry: &Path, output: &Path) -> Result<(), String> {
        let bytes = std::fs::read(output).map_err(|e| self.failure("read", output, &e))?;
        let dir = entry
            .parent()
            .expect("an entry path has a fan-out directory");
        std::fs::create_dir_all(dir).map_err(|e| self.failure("create", dir, &e))?;
        self.write_atomic(entry, &bytes)?;
        self.write_atomic(
            &ptx_cache::sidecar_of(entry),
            ptx_cache::sidecar_text(&bytes).as_bytes(),
        )
    }

    fn write_atomic(&self, dst: &Path, bytes: &[u8]) -> Result<(), String> {
        let seq = self.tmp_seq.fetch_add(1, Ordering::Relaxed);
        let mut name = dst
            .file_name()
            .expect("a cache path has a file name")
            .to_os_string();
        name.push(format!(".tmp-{}-{seq}", std::process::id()));
        let tmp = dst.with_file_name(name);
        std::fs::write(&tmp, bytes).map_err(|e| self.failure("write", &tmp, &e))?;
        std::fs::rename(&tmp, dst).map_err(|e| self.failure("rename", &tmp, &e))
    }

    fn failure(&self, what: &str, path: &Path, err: &dyn std::fmt::Display) -> String {
        format!(
            "PTX cache: cannot {what} {}: {err}. The cache is opt-in ({CACHE_SWITCH}=1): \
             rebuild with it unset, or remove {}",
            path.display(),
            self.root.display()
        )
    }
}

enum Fetch {
    Hit,
    Absent,
    /// 2026-10-17: Present, but the sidecar is missing or does not vouch for
    /// the bytes. Recompiled and rewritten, never served.
    Rejected,
}

#[path = "build_ptx_cache_gc.rs"]
mod gc;
