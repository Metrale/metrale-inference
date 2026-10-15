// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the reads every command shares, over the ports. Each error
//! names the file it came from exactly once.

use crate::cli::Io;
use crate::core::capture::device_kernels;
use crate::core::config::parse_toml;
use crate::core::error::{Error, Result};
use crate::core::families::{FamilyTable, classify, coverage_gate, table_from_toml};
use crate::core::model::Kernel;
use crate::core::tables::UNCLASSIFIED;
use crate::core::union::union_ns;

fn io_message(e: &std::io::Error) -> String {
    let s = e.to_string();
    match s.find(" (os error") {
        Some(cut) => s[..cut].to_string(),
        None => s,
    }
}

pub fn read_bytes(io: &Io, path: &str) -> Result<Vec<u8>> {
    io.files
        .read(path)
        .map_err(|e| Error(format!("cannot read {path}: {}", io_message(&e))))
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

/// The device's kernels and their family labels, behind the coverage gate.
pub fn classified_window<'t>(
    io: &Io,
    path: &str,
    device: i64,
    table: &'t FamilyTable,
) -> Result<(Vec<Kernel>, Vec<&'t str>)> {
    let ks = kernels(io, path, device)?;
    let fams = classify(&ks, table);
    let classified = union_ns(
        ks.iter()
            .zip(&fams)
            .filter(|(_, f)| f.is_some())
            .map(|(k, _)| (k.start, k.end)),
    )?;
    let all = union_ns(ks.iter().map(|k| (k.start, k.end)))?;
    coverage_gate(classified as f64, all as f64, table)?;
    let labels = fams
        .into_iter()
        .map(|f| f.unwrap_or(UNCLASSIFIED))
        .collect();
    Ok((ks, labels))
}
