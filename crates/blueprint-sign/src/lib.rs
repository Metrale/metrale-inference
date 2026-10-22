// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Blueprint signing from seed files: the only place a signing secret is read.
//!
//! A key file is text, one field per line:
//!
//! ```text
//! kbp-key v1
//! kid kbp-blueprints-2026-10
//! suite mldsa65+ed25519
//! mldsa-seed <64 hex digits>
//! ed25519-seed <64 hex digits>
//! ```
//!
//! The math is `metrale_blueprint_trust` (deterministic, so no entropy is needed); this crate
//! reads the file, holds the seeds in `Zeroizing` buffers until the key is expanded, and writes
//! the public objects (anchors, certificates, revocation lists) as postcard files.
//!
//! Owner: blueprint.
//! Invariants:
//! - A key is only ever read from a file path, never from an argument or the environment.
//! - On unix a key file readable by group or others is refused.
//! - No seed is generated here: tests generate their own, and production keys are the
//!   owner's ceremony (DESIGN.md §11 Q1).

#![forbid(unsafe_code)]
#![deny(warnings)]
#![deny(clippy::all)]

use std::path::Path;

use metrale_blueprint::codec::DecodeMemory;
use metrale_blueprint_trust::{Kid, SignedCert, Signer, Suite};
use metrale_error::{Error, ErrorCode, error};
use serde::Serialize;
use serde::de::DeserializeOwned;
use zeroize::Zeroizing;

/// 2026-10-22: The first line of a key file.
pub const KEY_MAGIC: &str = "kbp-key v1";

fn io(path: &Path, e: &std::io::Error) -> Error {
    error!(ErrorCode::Io, format!("{}: {e}", path.display()))
}

fn bad(path: &Path, why: &str) -> Error {
    error!(
        ErrorCode::BlueprintKey,
        path.display().to_string(),
        why.to_string()
    )
}

fn seed(path: &Path, hex: &str) -> Result<Zeroizing<[u8; 32]>, Error> {
    let mut out = Zeroizing::new([0u8; 32]);
    let b = hex.as_bytes();
    if b.len() != 64 {
        return Err(bad(path, "a seed is 64 hex digits"));
    }
    for (i, o) in out.iter_mut().enumerate() {
        let pair =
            std::str::from_utf8(&b[2 * i..2 * i + 2]).map_err(|_| bad(path, "seed is not hex"))?;
        *o = u8::from_str_radix(pair, 16).map_err(|_| bad(path, "seed is not hex"))?;
    }
    Ok(out)
}

#[cfg(unix)]
fn check_mode(path: &Path) -> Result<(), Error> {
    use std::os::unix::fs::PermissionsExt;
    let mode = std::fs::metadata(path)
        .map_err(|e| io(path, &e))?
        .permissions()
        .mode();
    if mode & 0o077 != 0 {
        return Err(bad(
            path,
            "a key file must not be readable by group or others (chmod 600)",
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
fn check_mode(_: &Path) -> Result<(), Error> {
    Ok(())
}

/// 2026-10-22: The signer a key file holds.
pub fn read_key(path: &Path) -> Result<Signer, Error> {
    check_mode(path)?;
    let text = Zeroizing::new(std::fs::read_to_string(path).map_err(|e| io(path, &e))?);
    let mut lines = text.lines();
    if lines.next() != Some(KEY_MAGIC) {
        return Err(bad(
            path,
            "not a kbp key file (first line must be `kbp-key v1`)",
        ));
    }
    let (mut kid, mut suite, mut mldsa, mut ed) = (None, None, None, None);
    for line in lines.filter(|l| !l.trim().is_empty()) {
        let (k, v) = line
            .split_once(' ')
            .ok_or_else(|| bad(path, "a line is `field value`"))?;
        let slot_taken = match k {
            "kid" => kid.replace(Kid::new(v)?).is_some(),
            "suite" => suite
                .replace(Suite::parse(v).ok_or_else(|| bad(path, "unknown suite"))?)
                .is_some(),
            "mldsa-seed" => mldsa.replace(seed(path, v)?).is_some(),
            "ed25519-seed" => ed.replace(seed(path, v)?).is_some(),
            _ => return Err(bad(path, "unknown field")),
        };
        if slot_taken {
            return Err(bad(path, "a field is given twice"));
        }
    }
    match (kid, suite, mldsa, ed) {
        (Some(kid), Some(suite), Some(m), Some(e)) => Ok(Signer::from_seeds(kid, suite, &m, &e)),
        _ => Err(bad(
            path,
            "kid, suite, mldsa-seed and ed25519-seed are all required",
        )),
    }
}

/// 2026-10-22: Read a public object (anchor, certificate, revocation list) written by
/// [`write_public`].
pub fn read_public<T: DeserializeOwned>(path: &Path) -> Result<T, Error> {
    let bytes = std::fs::read(path).map_err(|e| io(path, &e))?;
    match postcard::take_from_bytes::<T>(&bytes) {
        Ok((v, [])) => Ok(v),
        _ => Err(error!(
            ErrorCode::BlueprintDecode,
            path.display().to_string(),
            "postcard"
        )),
    }
}

/// 2026-10-22: Write a public object as postcard.
pub fn write_public<T: Serialize>(path: &Path, v: &T) -> Result<(), Error> {
    let bytes = postcard::to_allocvec(v).map_err(|_| bad(path, "does not encode"))?;
    std::fs::write(path, bytes).map_err(|e| io(path, &e))
}

/// 2026-10-22: Sign a `.kbp` with the key at `key` and its certificate at `cert`; the body
/// decodes within the caller's `memory` budget.
pub fn sign_kbp(
    file: &[u8],
    key: &Path,
    cert: &Path,
    memory: DecodeMemory,
) -> Result<Vec<u8>, Error> {
    let signer = read_key(key)?;
    let cert: SignedCert = read_public(cert)?;
    if &cert.subject != signer.kid() {
        return Err(bad(key, "the certificate is for another kid"));
    }
    metrale_blueprint::frame::sign(file, &signer, vec![cert], memory)
}
