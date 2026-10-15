// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: whole-file reads from the local filesystem, and the one write
//! the binary makes: a result file a command asked for.

use crate::ports::Files;

pub struct FsFiles;

impl Files for FsFiles {
    fn read(&self, path: &str) -> std::io::Result<Vec<u8>> {
        std::fs::read(path)
    }
}

/// Write `bytes` to `path` whole, replacing any file there.
pub fn write(path: &str, bytes: &[u8]) -> std::io::Result<()> {
    std::fs::write(path, bytes)
}
