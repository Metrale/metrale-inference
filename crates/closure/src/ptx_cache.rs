// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The pure half of the opt-in content-hashed PTX cache: the key,
//! the entry layout, the sidecar check and the LRU eviction choice. The kernels
//! build script owns every read, write and rename; this module takes values.
//!
//! Owner: metrale-closure (no CUDA dependency, so its unit tests run; the
//! kernels build script's own `#[cfg(test)]` code never does).
//! Invariants:
//! - The key covers the exact nvcc argv, the compiler identity, every `NVCC_*`
//!   variable and the path, length and SHA-256 of every file in the include
//!   closure. Anything else that changes the PTX and is not in this list is a
//!   false hit, so an input is added here before it is relied on.
//! - An entry is trusted only when its sidecar names the SHA-256 of its bytes.
//! - Eviction never picks an entry used by the running build.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use sha2::{Digest, Sha256};

use crate::hex_lower;

/// 2026-10-17: Version of the key definition and of the on-disk layout. It
/// names the `v<N>` directory, so a change starts an empty cache beside the
/// old one instead of reading entries keyed another way.
pub const PTX_CACHE_SCHEMA: u32 = 1;

/// 2026-10-17: The cache directory's name under the cargo target directory.
/// It sits outside every `OUT_DIR`, so `build_stage::reset` cannot reach it.
pub const PTX_CACHE_DIR: &str = "metrale-ptx-cache";

/// 2026-10-17: Extension of an entry's sidecar, appended to the entry name.
pub const SIDECAR_SUFFIX: &str = ".sha256";

/// 2026-10-17: One file of a compile's include closure, as the key sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileFact {
    /// 2026-10-17: The canonical path. It is hashed because the PTX of a
    /// device `assert` embeds the source path.
    pub path: String,
    pub len: u64,
    /// 2026-10-17: Lower-case hex SHA-256 of the bytes.
    pub sha256: String,
}

impl FileFact {
    /// 2026-10-17: The fact for `bytes` read from `path`.
    pub fn of(path: &Path, bytes: &[u8]) -> Self {
        Self {
            path: path.to_string_lossy().into_owned(),
            len: bytes.len() as u64,
            sha256: hex_lower(&Sha256::digest(bytes)),
        }
    }
}

/// 2026-10-17: Everything one cache key is derived from.
#[derive(Debug, Clone, Copy)]
pub struct KeyInputs<'a> {
    /// 2026-10-17: The nvcc path and its full `--version` output.
    pub compiler: &'a str,
    /// 2026-10-17: The argv nvcc is run with, program name excluded, in order.
    pub argv: &'a [String],
    /// 2026-10-17: Every `NVCC_*` variable as `(name, value)`, in any order.
    pub nvcc_env: &'a [(String, String)],
    /// 2026-10-17: The include closure, in any order.
    pub files: &'a [FileFact],
    /// 2026-10-17: Quoted includes no directory resolved, in any order.
    pub unresolved: &'a [String],
}

/// 2026-10-17: The lower-case hex SHA-256 key of one compile.
///
/// Framed as `hash_with_report` is: a domain tag, the schema, then a label
/// before each field, so equal bytes in different fields give different keys.
/// The argv is hashed in order (flag order is meaningful to nvcc); the env,
/// the files and the unresolved names are sorted first, so the caller's
/// iteration order cannot move the key.
pub fn cache_key(inputs: &KeyInputs<'_>) -> String {
    let mut digest = Sha256::new();
    digest.update(b"metrale-ptx-cache\x00");
    digest.update(PTX_CACHE_SCHEMA.to_le_bytes());
    digest.update(b"\x00compiler\x00");
    digest.update(inputs.compiler.as_bytes());
    digest.update(b"\x00argv\x00");
    for arg in inputs.argv {
        digest.update(arg.as_bytes());
        digest.update(b"\x1f");
    }
    digest.update(b"\x00nvccenv\x00");
    let mut env: Vec<&(String, String)> = inputs.nvcc_env.iter().collect();
    env.sort();
    for (name, value) in env {
        digest.update(name.as_bytes());
        digest.update(b"=");
        digest.update(value.as_bytes());
        digest.update(b"\x1f");
    }
    digest.update(b"\x00files\x00");
    let mut files: Vec<&FileFact> = inputs.files.iter().collect();
    files.sort_by(|a, b| a.path.cmp(&b.path));
    for file in files {
        digest.update(file.path.as_bytes());
        digest.update(b"\x1f");
        digest.update(file.len.to_le_bytes());
        digest.update(file.sha256.as_bytes());
        digest.update(b"\x1e");
    }
    digest.update(b"\x00unresolved\x00");
    let mut unresolved: Vec<&String> = inputs.unresolved.iter().collect();
    unresolved.sort();
    for entry in unresolved {
        digest.update(entry.as_bytes());
        digest.update(b"\x1f");
    }
    hex_lower(&digest.finalize())
}

/// 2026-10-17: The directories nvcc searches for a quoted include after the
/// including file's own, in order: `-I`/`--include-path` first, then
/// `-isystem`/`--system-include`. `NVCC_PREPEND_FLAGS` and `NVCC_APPEND_FLAGS`
/// wrap the argv exactly as nvcc applies them, so a `-I` there is followed
/// too.
pub fn search_dirs(argv: &[String], nvcc_env: &[(String, String)]) -> Vec<PathBuf> {
    let env = |name: &str| -> Vec<String> {
        nvcc_env
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.split_whitespace().map(str::to_string).collect())
            .unwrap_or_default()
    };
    let mut tokens: Vec<String> = env("NVCC_PREPEND_FLAGS");
    tokens.extend(argv.iter().cloned());
    tokens.extend(env("NVCC_APPEND_FLAGS"));

    let mut include = Vec::new();
    let mut system = Vec::new();
    let mut tokens = tokens.into_iter();
    while let Some(tok) = tokens.next() {
        let (list, value) = if let Some(v) = split_flag(&tok, &["-I", "--include-path"]) {
            (&mut include, v)
        } else if let Some(v) = split_flag(&tok, &["-isystem", "--system-include"]) {
            (&mut system, v)
        } else {
            continue;
        };
        let dir = match value {
            Some(dir) => dir,
            None => match tokens.next() {
                Some(dir) => dir,
                None => break,
            },
        };
        list.push(PathBuf::from(dir));
    }
    include.extend(system);
    include
}

/// 2026-10-17: `Some(Some(v))` for `<flag>v` or `<flag>=v`, `Some(None)` for
/// the bare flag (value in the next token), `None` for another token. The
/// match is case-sensitive, so `-I` never claims `-isystem`.
fn split_flag(tok: &str, flags: &[&str]) -> Option<Option<String>> {
    for flag in flags {
        let Some(rest) = tok.strip_prefix(flag) else {
            continue;
        };
        if rest.is_empty() {
            return Some(None);
        }
        if let Some(v) = rest.strip_prefix('=') {
            return Some(Some(v.to_string()));
        }
        // 2026-10-17: Only the short forms take a glued value; `--include-pathX`
        // is another option, not a search directory.
        if !flag.starts_with("--") {
            return Some(Some(rest.to_string()));
        }
    }
    None
}

/// 2026-10-17: An entry's path relative to the cache root:
/// `v<schema>/<first two key chars>/<key>.ptx`.
pub fn entry_rel(key: &str) -> PathBuf {
    let fan = key.get(..2).unwrap_or(key);
    PathBuf::from(format!("v{PTX_CACHE_SCHEMA}"))
        .join(fan)
        .join(format!("{key}.ptx"))
}

/// 2026-10-17: The sidecar path of the entry at `entry`.
pub fn sidecar_of(entry: &Path) -> PathBuf {
    let mut name = entry.as_os_str().to_os_string();
    name.push(SIDECAR_SUFFIX);
    PathBuf::from(name)
}

/// 2026-10-17: The sidecar text that vouches for `bytes`.
pub fn sidecar_text(bytes: &[u8]) -> String {
    format!("{}\n", hex_lower(&Sha256::digest(bytes)))
}

/// 2026-10-17: Whether `sidecar` vouches for `bytes`. Anything but the exact
/// text [`sidecar_text`] writes is a miss: a torn write, a truncated entry and
/// a hand-edited file all read as "not cached", never as PTX.
pub fn verify(bytes: &[u8], sidecar: &str) -> bool {
    sidecar == sidecar_text(bytes)
}

/// 2026-10-17: One entry as eviction sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEntry {
    pub path: PathBuf,
    /// 2026-10-17: Entry plus sidecar.
    pub bytes: u64,
    /// 2026-10-17: The entry's mtime; a hit sets it to now.
    pub last_used: SystemTime,
}

/// 2026-10-17: The entries to delete so the cache fits in `cap` bytes,
/// least recently used first, ties broken by path so the choice repeats.
///
/// An entry used at or after `protect_since` (the running build's start) is
/// never chosen: this build may still be copying it. If the protected entries
/// alone exceed `cap`, the result holds every unprotected entry and the cache
/// stays over the cap until a later build.
pub fn evict(entries: &[CacheEntry], cap: u64, protect_since: SystemTime) -> Vec<PathBuf> {
    let mut total: u64 = entries.iter().map(|e| e.bytes).sum();
    let mut order: Vec<&CacheEntry> = entries.iter().collect();
    order.sort_by(|a, b| {
        a.last_used
            .cmp(&b.last_used)
            .then_with(|| a.path.cmp(&b.path))
    });
    let mut gone = Vec::new();
    for entry in order {
        if total <= cap {
            break;
        }
        if entry.last_used >= protect_since {
            continue;
        }
        total -= entry.bytes;
        gone.push(entry.path.clone());
    }
    gone
}

/// 2026-10-17: `METRALE_PTX_CACHE`: unset or `0` is off, `1` is on. Any other
/// value is an error, so a typo cannot silently pick a mode.
///
/// # Errors
/// The offending value, for the build script to panic with.
pub fn parse_switch(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!(
            "METRALE_PTX_CACHE must be unset, 0 or 1, got {other:?}"
        )),
    }
}

/// 2026-10-17: `METRALE_PTX_CACHE_MAX_BYTES`, required while the cache is on:
/// there is no default cap (PCND). `scripts/dev-build.sh` names the value it
/// passes.
///
/// # Errors
/// A missing, non-decimal or zero value.
pub fn parse_cap(value: Option<&str>) -> Result<u64, String> {
    let raw = value.ok_or_else(|| {
        "METRALE_PTX_CACHE=1 requires METRALE_PTX_CACHE_MAX_BYTES (a byte count); \
         there is no default cap"
            .to_string()
    })?;
    match raw.parse::<u64>() {
        Ok(0) | Err(_) => Err(format!(
            "METRALE_PTX_CACHE_MAX_BYTES must be a positive byte count, got {raw:?}"
        )),
        Ok(cap) => Ok(cap),
    }
}

#[cfg(test)]
#[path = "ptx_cache_tests.rs"]
mod ptx_cache_tests;
