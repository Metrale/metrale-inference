// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The host half of a server's identity: reading this process's
//! arguments, executable and `METRALE_*` environment. The pure half (the
//! digests and the `GET /serve-config` shape) is
//! `metrale_serve_identity::identity`.
//!
//! 2026-10-17: Split out of metrale-bench (`serve_identity`, `serve_env`) so
//! `met` publishes its identity without the benchmark harness (OWAY, SBIO).
//!
//! Owner: host-probe (serve identity).
//! Invariants:
//! - Every digest this module computes is 64 lowercase hex characters;
//!   [`this_process`] reports `unavailable: …` as `binary_sha256` when the
//!   executable cannot be read.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use metrale_serve_identity::env::{fingerprint, levers};
use metrale_serve_identity::hex_lower;
use metrale_serve_identity::identity::{ServeIdentity, argv_fingerprint};
use sha2::{Digest, Sha256};

/// 2026-09-26: The SHA-256 of a file's bytes; used as a binary's identity.
pub fn file_sha256(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut h = Sha256::new();
    // 2026-09-30: Read in chunks; sha2 0.11's hasher is not an `io::Write`, so
    // `io::copy` into it no longer compiles.
    let mut buf = vec![0u8; 1 << 16];
    loop {
        match std::io::Read::read(&mut f, &mut buf) {
            Ok(0) => break,
            Ok(n) => h.update(&buf[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        }
    }
    Ok(hex_lower(&h.finalize()))
}

/// 2026-09-26: The levers of this process: what the server fingerprints, and
/// what the harness reconciles against the recipe before it serves.
///
/// Lossy on a non-UTF-8 name or value rather than skipping it: a skipped
/// variable would be a lever the fingerprint omits.
pub fn process_levers() -> BTreeMap<String, String> {
    levers(std::env::vars_os().map(|(k, v)| {
        (
            k.to_string_lossy().into_owned(),
            v.to_string_lossy().into_owned(),
        )
    }))
}

/// 2026-09-26: The identity of this process, computed once: the digests of
/// its executable, its arguments after the program name and its levers, and
/// its pid.
pub fn this_process() -> &'static ServeIdentity {
    static ID: std::sync::OnceLock<ServeIdentity> = std::sync::OnceLock::new();
    ID.get_or_init(|| {
        let args: Vec<String> = std::env::args().skip(1).collect();
        ServeIdentity {
            argv_sha256: argv_fingerprint(&args),
            binary_sha256: std::env::current_exe()
                .context("current_exe")
                .and_then(|p| file_sha256(&p))
                .unwrap_or_else(|e| format!("unavailable: {e:#}")),
            env_sha256: fingerprint(&process_levers()),
            pid: std::process::id(),
        }
    })
}

#[cfg(test)]
#[path = "serve_identity_tests.rs"]
mod tests;
