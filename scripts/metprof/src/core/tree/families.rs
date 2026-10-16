// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the kernel tree from the family table (pure), for a model with
//! no circuit.
//!
//! A classified launch sits at `<overlay node path>/<family>/<short name>` when
//! an overlay node claims its family, else at `<group>/<family>/<short name>`
//! (the table's own roll-up). An unclassified launch is listed as unattributed
//! under its full (demangled) name. Labels come from `classify`, which the
//! caller runs over the whole device window: a call-site rule needs the
//! launch's stream neighbours, which may lie outside the segment.

use indexmap::IndexMap;
use serde_json::{Map, Value};

use crate::core::error::Result;
use crate::core::families::FamilyTable;
use crate::core::model::Kernel;
use crate::core::tree::config::Resolved;
use crate::core::tree::{Loose, Node, Source, Tree, child};

/// The tree of `segment` (indices into `kernels`, any order; they are walked
/// in start order). `labels` is aligned with `kernels`.
pub fn families_tree(
    kernels: &[Kernel],
    labels: &[Option<&str>],
    segment: &[usize],
    table: &FamilyTable,
    overlay: &Resolved,
    provenance: Map<String, Value>,
) -> Result<Tree> {
    let mut order = segment.to_vec();
    order.sort_by_key(|&i| (kernels[i].start, kernels[i].end, i));
    let mut roots: IndexMap<String, Node> = IndexMap::new();
    let mut unattributed: IndexMap<String, Loose> = IndexMap::new();
    for i in order {
        let k = &kernels[i];
        let Some(family) = labels[i] else {
            unattributed
                .entry(k.demangled.clone())
                .or_default()
                .record(k);
            continue;
        };
        let path: Vec<String> = match overlay.claim.get(family) {
            Some(p) => p.clone(),
            None => vec![table.group(family)?.to_string()],
        };
        let mut map = &mut roots;
        for (depth, seg) in path.iter().enumerate() {
            let node = child(map, seg)?;
            if node.fuses.is_empty()
                && let Some(f) = overlay.node_fuses.get(&path[..=depth].join("/"))
            {
                node.fuses = f.clone();
            }
            map = &mut node.kernels;
        }
        let fam = child(map, family)?;
        if let Some(f) = overlay.family_fuses.get(family) {
            fam.fuses = f.clone();
        }
        child(&mut fam.kernels, &k.name)?.record(k);
    }
    let mut tree = Tree {
        provenance,
        source: Source::Families,
        kernels: roots,
        unattributed,
        baseline: None,
        slope: None,
    };
    tree.seal();
    Ok(tree)
}
