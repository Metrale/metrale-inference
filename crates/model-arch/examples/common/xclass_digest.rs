// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: One line per output buffer, `XCLASS-DIGEST <label> <sha256>`, so one microtest run
//! on two hardware classes answers "are these outputs byte-identical across the classes?" with a
//! diff of the two logs' digest lines (Tier 2 of the new-hardware bit-parity method,
//! `.claude/skills/new-hardware/references/bit-parity.md`).
//!
//! Owner: model-arch examples.
//! Invariants:
//! - The label names everything that selects the bytes (seed, shape, kernel, rows), so two lines
//!   with one label are the same computation.

use sha2::{Digest, Sha256};

/// 2026-10-05: Print the digest line for `bytes`.
pub fn print(label: &str, bytes: &[u8]) {
    let hex: String = Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    println!("XCLASS-DIGEST {label} {hex}");
}
