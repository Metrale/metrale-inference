// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The end of a cached build: the `rerun-if-changed` lines for
//! every include-closure file outside `OUT_DIR`, the per-run log, orphan
//! cleanup and LRU eviction down to the cap.
//!
//! Owner: metrale-kernels build.
//! Invariants:
//! - Runs only after every compile job succeeded.
//! - Removes only entries `metrale_closure::ptx_cache::evict` chose, their
//!   sidecars, and orphans older than `ORPHAN_AGE`. A file another build
//!   removed first is not an error.
//!
//! Included as a child of `build_ptx_cache` so it reads `PtxCache`'s fields.

use std::io::ErrorKind;
use std::path::Path;
use std::time::SystemTime;

use metrale_closure::ptx_cache::{self, CacheEntry};

use super::{LOG_NAME, ORPHAN_AGE, PtxCache};

impl PtxCache<'_> {
    /// 2026-10-17: Emit the closure reruns, write `OUT_DIR/ptx-cache.log`
    /// (`hit|miss|stale <key> <source>`, sorted), evict, then print one
    /// summary line. Panics on any cache I/O error.
    pub(crate) fn finish(&self) {
        // 2026-10-17: A header reached only through `-I` (the qwen
        // `w4a16_gemm.cu` the glm `dflash_fp8_gemm.cu` includes) is in no
        // layer directory, so without these lines editing it would not rerun
        // this script and the cache would never see the edit.
        for path in self.watched.lock().unwrap().iter() {
            println!("cargo:rerun-if-changed={}", path.display());
        }

        let mut log = self.log.lock().unwrap().clone();
        log.sort();
        let log_path = self.out_dir.join(LOG_NAME);
        let mut text = log.join("\n");
        if !text.is_empty() {
            text.push('\n');
        }
        std::fs::write(&log_path, text)
            .unwrap_or_else(|e| panic!("{}", self.failure("write", &log_path, &e)));
        for line in log.iter().filter(|l| l.starts_with("stale ")) {
            println!(
                "cargo:warning=metrale-kernels: PTX cache entry failed its sidecar check \
                 and was recompiled and rewritten: {line}"
            );
        }

        let (evicted, freed) = self.collect_garbage().unwrap_or_else(|e| panic!("{e}"));
        let count = |kind: &str| {
            log.iter()
                .filter(|l| l.split(' ').next() == Some(kind))
                .count()
        };
        println!(
            "cargo:warning=metrale-kernels: PTX cache {}: {} hits, {} misses, {} stale; \
             evicted {evicted} entries ({freed} bytes) to the {} byte cap",
            self.root.display(),
            count("hit"),
            count("miss"),
            count("stale"),
            self.cap,
        );
    }

    /// 2026-10-17: Remove old orphans, then the entries `evict` picks.
    /// Returns the number of entries evicted and the bytes they held.
    fn collect_garbage(&self) -> Result<(usize, u64), String> {
        let schema_dir = self.root.join(
            ptx_cache::entry_rel("00")
                .components()
                .next()
                .expect("an entry path starts with its schema directory"),
        );
        let now = SystemTime::now();
        let fans = match std::fs::read_dir(&schema_dir) {
            Ok(fans) => fans,
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok((0, 0)),
            Err(e) => return Err(self.failure("list", &schema_dir, &e)),
        };
        let mut entries: Vec<CacheEntry> = Vec::new();
        for fan in fans {
            let fan = fan
                .map_err(|e| self.failure("list", &schema_dir, &e))?
                .path();
            if !fan.is_dir() {
                continue;
            }
            for file in std::fs::read_dir(&fan).map_err(|e| self.failure("list", &fan, &e))? {
                let path = file.map_err(|e| self.failure("list", &fan, &e))?.path();
                let meta = match std::fs::metadata(&path) {
                    Ok(meta) => meta,
                    Err(e) if e.kind() == ErrorKind::NotFound => continue,
                    Err(e) => return Err(self.failure("stat", &path, &e)),
                };
                let modified = meta
                    .modified()
                    .map_err(|e| self.failure("stat", &path, &e))?;
                let orphan_age = now
                    .duration_since(modified)
                    .is_ok_and(|age| age > ORPHAN_AGE);
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default();
                if name.contains(".tmp-") {
                    if orphan_age {
                        self.remove(&path)?;
                    }
                } else if let Some(entry_name) = name.strip_suffix(ptx_cache::SIDECAR_SUFFIX) {
                    if orphan_age && !fan.join(entry_name).exists() {
                        self.remove(&path)?;
                    }
                } else if name.ends_with(".ptx") {
                    let sidecar_path = ptx_cache::sidecar_of(&path);
                    let sidecar = match std::fs::metadata(&sidecar_path) {
                        Ok(m) => m.len(),
                        Err(e) if e.kind() == ErrorKind::NotFound => 0,
                        Err(e) => return Err(self.failure("stat", &sidecar_path, &e)),
                    };
                    entries.push(CacheEntry {
                        path,
                        bytes: meta.len() + sidecar,
                        last_used: modified,
                    });
                }
            }
        }

        let victims = ptx_cache::evict(&entries, self.cap, self.started);
        let mut freed = 0;
        for victim in &victims {
            freed += entries
                .iter()
                .find(|e| &e.path == victim)
                .map_or(0, |e| e.bytes);
            self.remove(victim)?;
            self.remove(&ptx_cache::sidecar_of(victim))?;
        }
        Ok((victims.len(), freed))
    }

    /// 2026-10-17: Delete one cache file; already gone (a concurrent build
    /// evicted it) is success.
    fn remove(&self, path: &Path) -> Result<(), String> {
        match std::fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
            Err(e) => Err(self.failure("remove", path, &e)),
        }
    }
}
