// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The `[[legacy_path]]` entries of KERNEL_FAMILIES.toml, split from
//! `families_file.rs` unchanged.
//!
//! Owner: metrale-circuit (venn).
//! Invariants: see [`super`].

use serde::Deserialize;

use super::{FamilyError, LegacyPath};
use crate::ir::LayerKind;
use crate::rules::Mode;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct LegacyFile {
    arch: String,
    layer_kind: String,
    per_sequence: Vec<String>,
    sites: Vec<String>,
    rows_above: u64,
    cite: String,
    holds: String,
    note: String,
}

pub(super) fn legacy(l: LegacyFile) -> Result<LegacyPath, FamilyError> {
    let bad = |detail: String| FamilyError::Parse(format!("legacy_path `{}`: {detail}", l.cite));
    let layer_kind = LayerKind::parse(&l.layer_kind)
        .ok_or_else(|| bad(format!("layer kind `{}`", l.layer_kind)))?;
    let per_sequence = l
        .per_sequence
        .iter()
        .map(|m| Mode::parse(m).ok_or_else(|| bad(format!("mode `{m}`"))))
        .collect::<Result<Vec<_>, _>>()?;
    if per_sequence.is_empty() || l.holds.trim().is_empty() {
        return Err(bad("needs modes and the text the cited line holds".into()));
    }
    Ok(LegacyPath {
        arch: l.arch.clone(),
        layer_kind,
        per_sequence,
        sites: l.sites.clone(),
        rows_above: l.rows_above,
        cite: l.cite.clone(),
        holds: l.holds.clone(),
        note: l.note,
    })
}
