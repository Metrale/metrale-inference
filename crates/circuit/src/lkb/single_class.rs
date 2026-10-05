// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The single-class bucket of the LKB residual (owner decision, 2026-10-05): family
//! points whose atom bundle only one class realizes (crates/circuit/src/venn/atoms.rs). They are
//! declared points, not silent copies, but they converge nothing until a second class realizes
//! the bundle, so `met circuit lkb` counts them apart from the shared points and from the
//! class's own unregistered sources.
//!
//! Owner: metrale-circuit.
//! Invariants: a point is in the bucket exactly when a parameter of domain `atom_bundle` names a
//! bundle with one class.

use std::fmt::Write as _;

use crate::venn::families::Families;
use crate::venn::families::atoms::DOMAIN;

/// 2026-10-05: One point in the bucket.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SingleClassPoint {
    /// 2026-10-05: Family id.
    pub family: String,
    /// 2026-10-05: The point's values, `name=value` joined by commas.
    pub values: String,
    /// 2026-10-05: The bundle.
    pub bundle: String,
    /// 2026-10-05: The one class that realizes it.
    pub class: String,
}

/// 2026-10-05: The bucket of `fams` (a class's resolved families).
pub fn single_class_points(fams: &Families) -> Vec<SingleClassPoint> {
    let mut out = Vec::new();
    for f in &fams.families {
        let atoms: Vec<&str> = f
            .params
            .iter()
            .filter(|p| p.of.as_deref() == Some(DOMAIN))
            .map(|p| p.name.as_str())
            .collect();
        for p in &f.points {
            for a in &atoms {
                let Some(b) = p.values.get(*a).and_then(|id| fams.bundles.get(id)) else {
                    continue;
                };
                if b.single_class() {
                    out.push(SingleClassPoint {
                        family: f.id.clone(),
                        values: p
                            .values
                            .iter()
                            .map(|(k, v)| format!("{k}={v}"))
                            .collect::<Vec<_>>()
                            .join(","),
                        bundle: b.id.clone(),
                        class: b.classes[0].clone(),
                    });
                }
            }
        }
    }
    out
}

/// 2026-10-05: The bundles the class declares and the single-class bucket, as Markdown.
pub fn render(
    s: &mut String,
    bundles: &std::collections::BTreeMap<String, crate::venn::families::atoms::AtomBundle>,
    bucket: &[SingleClassPoint],
) {
    let _ = writeln!(s, "## Atom bundles\n");
    if bundles.is_empty() {
        let _ = writeln!(s, "None declared.\n");
    } else {
        let _ = writeln!(
            s,
            "| bundle | MMA | m x n x k | schedule | accumulator | classes |"
        );
        let _ = writeln!(s, "|---|---|---|---|---|---|");
        for b in bundles.values() {
            let _ = writeln!(
                s,
                "| {} | `{}` | {}x{}x{} | {} ({} stages) | {} | {} |",
                b.id,
                b.mma,
                b.shape.0,
                b.shape.1,
                b.shape.2,
                b.schedule,
                b.stages,
                b.accumulator,
                b.classes.join(", ")
            );
        }
        s.push('\n');
    }
    let _ = writeln!(
        s,
        "## LKB residual, single-class bucket: {} points\n\nDeclared points whose atom bundle \
         only one class realizes. They leave this bucket when a second class realizes the \
         bundle.\n",
        bucket.len()
    );
    if !bucket.is_empty() {
        let _ = writeln!(s, "| family | point | bundle | class |\n|---|---|---|---|");
        for p in bucket {
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} |",
                p.family, p.values, p.bundle, p.class
            );
        }
        s.push('\n');
    }
}
