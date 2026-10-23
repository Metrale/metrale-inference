// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The kernel-math table of one plan: the explanation of every kernel family and
//! every tensor format it names, de-duplicated (one entry per family or format, nodes and edges
//! refer to it by id), and the per-token bytes of each edge. The blueprint graph export and
//! `met circuit plan --format explain` both build it here, so the math has one source
//! (KERNEL_FAMILIES.toml `explain` and [`crate::format_explain`]).
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - A family the plan names but the manifest lacks is an error, never a missing entry: the
//!   store shows math for every kernel node or refuses the export.
//! - Keys are sorted (`BTreeMap`), so the table is deterministic.

use crate::prelude::*;

use alloc::collections::BTreeMap;

use crate::format::Format;
use crate::format_explain::FormatExplain;
use crate::ir::{Circuit, Edge};
use crate::venn::Families;

/// 2026-10-22: One family's entry.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct FamilyMath {
    /// 2026-10-22: The manifest's one line.
    pub description: String,
    /// 2026-10-22: Prose.
    pub about: String,
    /// 2026-10-22: Precision prose.
    pub numerics: String,
    /// 2026-10-22: Display equations.
    pub math: Vec<String>,
}

/// 2026-10-22: The table.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize)]
pub struct KernelMath {
    /// 2026-10-22: By family id.
    pub families: BTreeMap<String, FamilyMath>,
    /// 2026-10-22: By format spelling.
    pub formats: BTreeMap<String, FormatExplain>,
}

/// 2026-10-22: The entries for `families` and `formats` (duplicates collapse).
pub fn kernel_math<'a>(
    manifest: &Families,
    families: impl IntoIterator<Item = &'a str>,
    formats: impl IntoIterator<Item = Format>,
) -> Result<KernelMath, String> {
    let mut out = KernelMath::default();
    for id in families {
        if out.families.contains_key(id) {
            continue;
        }
        let f = manifest
            .families
            .iter()
            .find(|f| f.id == id)
            .ok_or_else(|| {
                format!(
                    "family `{id}` is not in the {} KERNEL_FAMILIES.toml: no math to export",
                    manifest.hardware
                )
            })?;
        out.families.insert(
            id.to_string(),
            FamilyMath {
                description: f.description.clone(),
                about: f.explain.about.clone(),
                numerics: f.explain.numerics.clone(),
                math: f.explain.math.clone(),
            },
        );
    }
    for f in formats {
        out.formats.entry(f.name()).or_insert_with(|| f.explain());
    }
    Ok(out)
}

/// 2026-10-22: Rows of `e` per token (its row expression at `n = 1`); `None` when it reads a
/// dim the circuit does not bind.
pub fn rows_per_token(c: &Circuit, e: &Edge) -> Option<u64> {
    let mut dims = c.dims.clone();
    dims.insert("n".into(), 1);
    e.rows.eval(&dims).ok()
}

/// 2026-10-22: Bytes of `e` per token when stored as `format`, scales included
/// ([`Format::bytes`]); `None` when the rows are not derivable or the width is not a multiple
/// of the scale group.
pub fn bytes_per_token(c: &Circuit, e: &Edge, format: Format) -> Option<u64> {
    format.bytes(rows_per_token(c, e)?, e.dim_value)
}

/// 2026-10-22: The requantisation an edge stands for: `(from, to)` when the edge stores a
/// quantized format that differs from the format its producer read (the producer encoded it).
pub fn requant(producer_input: Option<Format>, stored: Format) -> Option<(Format, Format)> {
    let from = producer_input?;
    (!stored.is_plain() && from != stored).then_some((from, stored))
}

#[cfg(test)]
#[path = "explain_tests.rs"]
mod explain_tests;
