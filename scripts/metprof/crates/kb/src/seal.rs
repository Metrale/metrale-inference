// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: sealing the local export (DESIGN §3.1 step 4, §7.2): the
//! export `kkb.jsonl` sealed with the KKB1 envelope (the `metprof-envelope`
//! crate) to `kkb.jsonl.kkb` under the box's local export key, replaced
//! atomically. The key file holds the 32-byte key as 64 hex digits (and at
//! most one trailing newline); on Unix a key file readable by group or
//! others is refused. The kid is `local-` and the first 16 hex digits of the
//! key's SHA-256, so a reader knows which key a file needs without the key.

use std::fs;
use std::path::{Path, PathBuf};

use metprof_envelope::{Dek, Expected, OsEntropy, SealRequest, SystemClock};
use sha2::{Digest, Sha256};

use crate::export::EXPORT_FILE;
use crate::{Error, Result, atomic, bail};

pub const SEALED_FILE: &str = "kkb.jsonl.kkb";
/// The plaintext encoding label: the export as it is (no compression).
pub const ENCODING: &str = "lattice-jsonl/1";
pub const KEY_LEN: usize = 32;

fn envelope(what: &str, e: impl std::fmt::Display) -> Error {
    Error(format!("seal: {what}: {e}"))
}

/// The key from the key file's bytes.
pub fn parse_key(bytes: &[u8]) -> Result<[u8; KEY_LEN]> {
    let hex = bytes.strip_suffix(b"\n").unwrap_or(bytes);
    if hex.len() != 2 * KEY_LEN || !hex.iter().all(u8::is_ascii_hexdigit) {
        bail!(
            "seal: the key file must hold exactly {} hex digits",
            2 * KEY_LEN
        );
    }
    let mut key = [0u8; KEY_LEN];
    for (k, pair) in key.iter_mut().zip(hex.chunks_exact(2)) {
        let digit = |c: u8| (c as char).to_digit(16).unwrap_or(0) as u8;
        *k = digit(pair[0]) << 4 | digit(pair[1]);
    }
    Ok(key)
}

/// The local export key and the build that seals with it.
pub struct ExportSeal {
    dek: Dek,
    kid: String,
    publisher_build: String,
}

impl std::fmt::Debug for ExportSeal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ExportSeal({})", self.kid)
    }
}

impl ExportSeal {
    pub fn new(key: [u8; KEY_LEN], publisher_build: &str) -> ExportSeal {
        let digest: String = Sha256::digest(key)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        ExportSeal {
            dek: Dek::from_bytes(key),
            kid: format!("local-{}", &digest[..16]),
            publisher_build: publisher_build.to_string(),
        }
    }

    /// Read the key file at `path`.
    pub fn load(path: &Path, publisher_build: &str) -> Result<ExportSeal> {
        let what = |e: std::io::Error| Error(format!("seal: key file {}: {e}", path.display()));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(path).map_err(what)?.permissions().mode();
            if mode & 0o077 != 0 {
                bail!(
                    "seal: key file {} is mode {:o}; it must not be readable by group or others (0600)",
                    path.display(),
                    mode & 0o777
                );
            }
        }
        let bytes = fs::read(path).map_err(what)?;
        let key = parse_key(&bytes).map_err(|e| Error(format!("{} ({})", e.0, path.display())))?;
        Ok(ExportSeal::new(key, publisher_build))
    }

    pub fn kid(&self) -> &str {
        &self.kid
    }

    /// Seal `export_dir/kkb.jsonl` to `export_dir/kkb.jsonl.kkb`.
    pub fn seal(&self, export_dir: &Path) -> Result<PathBuf> {
        let plain = read(&export_dir.join(EXPORT_FILE))?;
        let request = SealRequest {
            kid: self.kid.clone(),
            path: EXPORT_FILE.to_string(),
            encoding: ENCODING.to_string(),
            publisher_build: self.publisher_build.clone(),
            wrapped_dek: None,
        };
        let sealed =
            metprof_envelope::seal(&self.dek, &request, &plain, &mut OsEntropy, &SystemClock)
                .map_err(|e| envelope("sealing", e))?;
        atomic::write(export_dir, SEALED_FILE, &sealed)
    }

    /// Open `kkb.jsonl.kkb` and require its plaintext to be `kkb.jsonl`;
    /// the number of bytes verified.
    pub fn verify(&self, export_dir: &Path) -> Result<usize> {
        let sealed_path = export_dir.join(SEALED_FILE);
        let opened = metprof_envelope::open(
            &self.dek,
            &Expected {
                kid: &self.kid,
                path: EXPORT_FILE,
            },
            &read(&sealed_path)?,
        )
        .map_err(|e| envelope(&format!("opening {}", sealed_path.display()), e))?;
        if opened.header.plaintext.encoding != ENCODING {
            bail!(
                "seal: {} has encoding {:?}, not {ENCODING:?}",
                sealed_path.display(),
                opened.header.plaintext.encoding
            );
        }
        if opened.plaintext != read(&export_dir.join(EXPORT_FILE))? {
            bail!(
                "seal: {} does not seal the current {EXPORT_FILE}; `metprof kb seal` re-seals it",
                sealed_path.display()
            );
        }
        Ok(opened.plaintext.len())
    }
}

fn read(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|e| Error(format!("seal: reading {}: {e}", path.display())))
}
