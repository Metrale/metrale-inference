// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: whole-file reads from the local filesystem, and the one write
//! the binary makes: a result file a command asked for.

use std::io::Read;

use sha2::{Digest, Sha256};

use crate::ports::Files;

pub struct FsFiles;

impl Files for FsFiles {
    fn read(&self, path: &str) -> std::io::Result<Vec<u8>> {
        std::fs::read(path)
    }

    /// Streamed, so a capture of many gigabytes is never held in memory.
    fn sha256(&self, path: &str) -> std::io::Result<String> {
        let mut f = std::fs::File::open(path)?;
        let mut h = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        loop {
            match f.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => h.update(&buf[..n]),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
    }
}

/// Write `bytes` to `path` whole, replacing any file there.
pub fn write(path: &str, bytes: &[u8]) -> std::io::Result<()> {
    std::fs::write(path, bytes)
}
