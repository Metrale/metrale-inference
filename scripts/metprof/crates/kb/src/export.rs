// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the local export (DESIGN §3.1 step 4): the collection as
//! deterministic JSONL, vectors omitted, replacing `<export_dir>/kkb.jsonl`
//! atomically. Sealing it to `kkb.jsonl.kkb` is `seal`.

use std::path::{Path, PathBuf};

use lattice_core::LatticeStorage;

use crate::Result;
use crate::atomic;
use crate::lattice::Store;

pub const EXPORT_FILE: &str = "kkb.jsonl";

/// The export bytes of `store`.
pub fn render<W: LatticeStorage, D: LatticeStorage>(store: &Store<W, D>) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    store.export(&mut out)?;
    Ok(out)
}

/// Replace `export_dir/kkb.jsonl` with the export of `store`.
pub fn write<W: LatticeStorage, D: LatticeStorage>(
    store: &Store<W, D>,
    export_dir: &Path,
) -> Result<PathBuf> {
    atomic::write(export_dir, EXPORT_FILE, &render(store)?)
}
