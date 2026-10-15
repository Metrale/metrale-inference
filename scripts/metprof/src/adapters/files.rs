// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: whole-file reads from the local filesystem.

use crate::ports::Files;

pub struct FsFiles;

impl Files for FsFiles {
    fn read(&self, path: &str) -> std::io::Result<Vec<u8>> {
        std::fs::read(path)
    }
}
