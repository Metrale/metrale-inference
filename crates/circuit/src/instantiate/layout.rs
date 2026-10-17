// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The layout half of the instantiation: the arch shape's dims against the
//! circuit's, and the block sequence a model's layer kinds map to (prologue, layers, epilogue,
//! draft), split out of `instantiate.rs` unchanged apart from the `[layout.prefix]` rule.
//!
//! Owner: metrale-circuit.
//! Invariants: see [`crate::circuit_toml`]; every layout block named anywhere must exist, every
//! block the circuit file defines must be used, and the first failure is returned.

use std::collections::{BTreeMap, BTreeSet};

use crate::circuit_toml::{CircuitError, CircuitFile, LayoutRule, when_holds};
use crate::ir::{ArchShape, LayerKind, Section};

pub(super) fn check_dims(file: &CircuitFile, shape: &ArchShape) -> Result<(), CircuitError> {
    for d in &file.dims {
        if !shape.dims.contains_key(d) {
            return Err(CircuitError::ShapeMismatch(format!(
                "circuit `{}` needs dim `{d}`, which the arch shape does not give",
                file.arch
            )));
        }
    }
    Ok(())
}

/// 2026-09-30: One block to instantiate: template, layer, section, and a module that replaces
/// the draft module for its `{L}`.
pub(super) type Planned = (String, Option<usize>, Section, Option<String>);

pub(super) fn block_sequence(
    file: &CircuitFile,
    rule: &LayoutRule,
    kinds: &[LayerKind],
    dims: &BTreeMap<String, u64>,
) -> Result<Vec<Planned>, CircuitError> {
    if kinds.is_empty() {
        return Err(CircuitError::Layout("the arch shape has no layers".into()));
    }
    let mut used = BTreeSet::new();
    let mut out = Vec::new();
    let known = |name: &str, used: &mut BTreeSet<String>| {
        if !file.block.contains_key(name) {
            return Err(CircuitError::Layout(format!("no block template `{name}`")));
        }
        used.insert(name.to_string());
        Ok(())
    };
    for name in &file.prologue {
        known(name, &mut used)?;
        out.push((name.clone(), None, Section::Main, None));
    }
    // 2026-09-30: The layout blocks per kind, with the overrides of every switch that holds.
    let mut layout = file.layout.blocks.clone();
    for (w, over) in &file.layout.when {
        for names in over.values() {
            for n in names {
                known(n, &mut used)?;
            }
        }
        if when_holds(w, &file.dims, dims)? {
            layout.extend(over.clone());
        }
    }
    let prefix = match &file.layout.prefix {
        Some(p) => {
            for n in p.blocks.values().flatten() {
                known(n, &mut used)?;
            }
            Some((prefix_len(file, &p.count, dims)?, p))
        }
        None => None,
    };
    for (i, kind) in kinds.iter().enumerate() {
        if let LayoutRule::Interval { period } = rule {
            let want = if (i + 1) % period == 0 {
                LayerKind::FullAttention
            } else {
                LayerKind::LinearAttention
            };
            if *kind != want {
                return Err(CircuitError::Layout(format!(
                    "layer {i} is {} but the interval-{period} layout puts {} there",
                    kind.name(),
                    want.name()
                )));
            }
        }
        let blocks = match prefix {
            Some((len, p)) if i < len => p.blocks.get(kind.name()).ok_or_else(|| {
                CircuitError::Layout(format!(
                    "layer {i} is {}, which the `{}` prefix maps to no blocks",
                    kind.name(),
                    p.count
                ))
            })?,
            _ => layout.get(kind.name()).ok_or_else(|| {
                CircuitError::Layout(format!(
                    "layer {i} is {}, which the layout maps to no blocks",
                    kind.name()
                ))
            })?,
        };
        for name in blocks {
            known(name, &mut used)?;
            out.push((name.clone(), Some(i), Section::Main, None));
        }
    }
    for name in &file.epilogue {
        known(name, &mut used)?;
        out.push((name.clone(), None, Section::Main, None));
    }
    let on = match &file.draft_when {
        Some(w) => when_holds(w, &file.dims, dims)?,
        None => true,
    };
    let mut draft = &file.draft;
    for (w, list) in &file.draft_variant {
        for entry in list {
            known(entry.split('@').next().unwrap_or_default(), &mut used)?;
        }
        if when_holds(w, &file.dims, dims)? {
            draft = list;
        }
    }
    for entry in &file.draft {
        known(entry.split('@').next().unwrap_or_default(), &mut used)?;
    }
    if on {
        for entry in draft {
            let (name, module) = match entry.split_once('@') {
                Some((n, m)) => (n.to_string(), Some(m.to_string())),
                None => (entry.clone(), None),
            };
            out.push((name, None, Section::Draft, module));
        }
    }
    if !file.draft.is_empty() && file.draft_module.is_none() {
        return Err(CircuitError::Layout(
            "`draft` blocks need a `draft_module` for their bindings".into(),
        ));
    }
    if let Some(unused) = file.local_blocks.iter().find(|k| !used.contains(*k)) {
        return Err(CircuitError::Layout(format!(
            "block template `{unused}` is never used"
        )));
    }
    Ok(out)
}

/// 2026-10-08: The length of a `[layout.prefix]`: the dim `count`, which the circuit must
/// declare and the arch shape give.
fn prefix_len(
    file: &CircuitFile,
    count: &str,
    dims: &BTreeMap<String, u64>,
) -> Result<usize, CircuitError> {
    if !file.dims.iter().any(|d| d == count) {
        return Err(CircuitError::ShapeMismatch(format!(
            "the layout prefix counts `{count}`, which is not in the circuit's `dims` list"
        )));
    }
    let len = dims.get(count).copied().ok_or_else(|| {
        CircuitError::ShapeMismatch(format!("the layout prefix: the arch shape lacks `{count}`"))
    })?;
    usize::try_from(len)
        .map_err(|_| CircuitError::Layout(format!("the `{count}` prefix ({len}) overflows")))
}
