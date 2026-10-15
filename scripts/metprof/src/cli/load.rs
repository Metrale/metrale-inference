// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the reads every command shares, over the ports. Each error
//! names the file it came from exactly once.

use crate::cli::Io;
use crate::core::capture::device_kernels;
use crate::core::config::parse_toml;
use crate::core::error::{Error, Result};
use crate::core::families::{FamilyTable, table_from_toml};
use crate::core::model::Kernel;

fn io_message(e: &std::io::Error) -> String {
    let s = e.to_string();
    match s.find(" (os error") {
        Some(cut) => s[..cut].to_string(),
        None => s,
    }
}

pub fn read_toml(io: &Io, path: &str) -> Result<toml::Table> {
    let bytes = match io.files.read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(Error(format!("config not found: {path}")));
        }
        Err(e) => return Err(Error(format!("cannot read {path}: {}", io_message(&e)))),
    };
    parse_toml(&bytes).map_err(|e| e.context(path))
}

pub fn load_families(io: &Io, path: &str) -> Result<FamilyTable> {
    table_from_toml(&read_toml(io, path)?).map_err(|e| e.context(path))
}

/// One device's kernels, ordered by (start, end).
pub fn kernels(io: &Io, path: &str, device: i64) -> Result<Vec<Kernel>> {
    let trace = io.traces.open(path)?;
    device_kernels(path, trace.kernel_rows()?, device)
}
