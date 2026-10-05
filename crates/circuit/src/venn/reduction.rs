// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Reduction trees, the numerics of a kernel (book/src/appendix/lkb-math.md,
//! section 2). Floating-point addition is not associative, so a reduction's result depends on its
//! bracketing: the order partial sums are formed and rounded at each level (thread, warp, CTA,
//! split), the rounding points, FMA contraction and the MMA atom's own order. Two kernels with the
//! same tree, inputs, formats and build flags give the same bytes; with different trees they are
//! compared under a tolerance.
//!
//! `[[reduction]]` declares a tree once by id; a family's `reduction` table names the tree each of
//! its kernels runs; a `numerics` parameter's values are tree ids.
//!
//! Owner: metrale-circuit (venn).
//! Invariants:
//! - A tree is written down from the kernel source and its launch, never inferred; a kernel
//!   with no declaration is "undeclared", which the parity verdict reads as unknown.
//! - Nothing defaults: every field of a tree is stated.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::rules::KernelId;

#[cfg(test)]
#[path = "reduction_tests.rs"]
mod reduction_tests;

/// 2026-10-05: How partial sums are combined at one level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// 2026-10-05: One accumulator, terms added in index order.
    Sequential,
    /// 2026-10-05: An XOR-shuffle butterfly (offsets halving).
    Butterfly,
    /// 2026-10-05: Partials through shared memory, then reduced by one warp in slot order.
    SharedThenWarp,
    /// 2026-10-05: The MMA instruction's own accumulation (fixed by the atom).
    Mma,
    /// 2026-10-05: Atomic adds: the order depends on scheduling (not deterministic).
    Atomic,
}

impl Order {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "sequential" => Order::Sequential,
            "butterfly" => Order::Butterfly,
            "shared_then_warp" => Order::SharedThenWarp,
            "mma" => Order::Mma,
            "atomic" => Order::Atomic,
            _ => return None,
        })
    }
}

/// 2026-10-05: One level of a tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Level {
    /// 2026-10-05: `thread`, `warp`, `cta`, `split` or `mma`.
    pub level: String,
    /// 2026-10-05: Terms combined at this level, as written (an expression over the model's
    /// dims, e.g. `min(hidden, 1024)`).
    pub width: String,
    /// 2026-10-05: How they are combined.
    pub order: Order,
}

/// 2026-10-05: What a split count depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitSource {
    /// 2026-10-05: No split.
    None,
    /// 2026-10-05: The model's shape only: the same on every device.
    Shape,
    /// 2026-10-05: The device (SM count, occupancy): differs between devices.
    Device,
}

/// 2026-10-05: A declared bracketing tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reduction {
    /// 2026-10-05: Id.
    pub id: String,
    /// 2026-10-05: The reduced dimension.
    pub dim: String,
    /// 2026-10-05: Levels, innermost first.
    pub levels: Vec<Level>,
    /// 2026-10-05: What the split count depends on.
    pub split: SplitSource,
    /// 2026-10-05: Where values are rounded, in order (`f32_acc`, `bf16_out`, ...).
    pub round: Vec<String>,
    /// 2026-10-05: `a * b + c` contracted into one FMA.
    pub fma: bool,
    /// 2026-10-05: The MMA atom, or `none`.
    pub mma: String,
}

impl Reduction {
    /// 2026-10-05: The result does not depend on scheduling.
    pub fn deterministic(&self) -> bool {
        self.levels.iter().all(|l| l.order != Order::Atomic)
    }
}

/// 2026-10-05: `[[reduction]]` in KERNEL_FAMILIES.toml.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ReductionFile {
    id: String,
    dim: String,
    levels: Vec<LevelFile>,
    split: String,
    round: Vec<String>,
    fma: bool,
    mma: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LevelFile {
    level: String,
    width: String,
    order: String,
}

const LEVELS: [&str; 5] = ["thread", "warp", "cta", "split", "mma"];

/// 2026-10-05: Validate the manifest's trees, by id.
pub(super) fn reductions(files: Vec<ReductionFile>) -> Result<BTreeMap<String, Reduction>, String> {
    let mut out = BTreeMap::new();
    for f in files {
        let at = |d: String| format!("reduction `{}`: {d}", f.id);
        if f.levels.is_empty() || f.round.is_empty() {
            return Err(at("states no levels or no rounding points".into()));
        }
        let levels = f
            .levels
            .into_iter()
            .map(|l| {
                if !LEVELS.contains(&l.level.as_str()) {
                    return Err(at(format!("level `{}` ({})", l.level, LEVELS.join(" | "))));
                }
                let order = Order::parse(&l.order).ok_or_else(|| {
                    at(format!(
                        "order `{}` (sequential | butterfly | shared_then_warp | mma | atomic)",
                        l.order
                    ))
                })?;
                Ok(Level {
                    level: l.level,
                    width: l.width,
                    order,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let split = match f.split.as_str() {
            "none" => SplitSource::None,
            "shape" => SplitSource::Shape,
            "device" => SplitSource::Device,
            other => return Err(at(format!("split `{other}` (none | shape | device)"))),
        };
        let r = Reduction {
            id: f.id.clone(),
            dim: f.dim,
            levels,
            split,
            round: f.round,
            fma: f.fma,
            mma: f.mma,
        };
        if out.insert(f.id.clone(), r).is_some() {
            return Err(format!("reduction `{}` is declared twice", f.id));
        }
    }
    Ok(out)
}

/// 2026-10-05: A family's `reduction` table: kernel to tree id, every kernel its own.
pub(super) fn kernel_reductions(
    table: &BTreeMap<String, String>,
    kernels: &[KernelId],
) -> Result<BTreeMap<KernelId, String>, String> {
    table
        .iter()
        .map(|(name, tree)| {
            let k = kernels
                .iter()
                .find(|k| k.to_string() == *name)
                .ok_or_else(|| format!("reduction names `{name}`, which is not its kernel"))?;
            Ok((k.clone(), tree.clone()))
        })
        .collect()
}

/// 2026-10-05: The value naming no declared tree.
pub const UNDECLARED: &str = "undeclared";

/// 2026-10-05: What two classes running one kernel at one point may be held to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// 2026-10-05: Bytes: the tree is declared, deterministic and device-independent, so a
    /// second class compiling the same source with the same flags must match exactly.
    Bytes,
    /// 2026-10-05: A derived tolerance: the tree differs between devices or between runs.
    Tolerance(String),
    /// 2026-10-05: No tree is declared.
    Unknown,
}

/// 2026-10-05: The verdict for a kernel running `tree` (None: undeclared).
pub fn verdict(tree: Option<&Reduction>) -> Verdict {
    match tree {
        None => Verdict::Unknown,
        Some(t) if !t.deterministic() => Verdict::Tolerance("atomic combine".into()),
        Some(t) if t.split == SplitSource::Device => {
            Verdict::Tolerance("split count derived from the device".into())
        }
        Some(_) => Verdict::Bytes,
    }
}

/// 2026-10-05: Every tree a family's `reduction` table or a `numerics` point names is declared
/// (or is `undeclared` on a point).
pub(super) fn check_names(
    families: &[super::Family],
    trees: &BTreeMap<String, Reduction>,
) -> Result<(), String> {
    for f in families {
        for (k, id) in &f.reduction {
            if !trees.contains_key(id) {
                return Err(format!(
                    "family `{}`: `{k}` runs reduction `{id}`, which no [[reduction]] declares",
                    f.id
                ));
            }
        }
        let numerics: Vec<&str> = f
            .params
            .iter()
            .filter(|p| p.kind == super::ParamKind::Numerics)
            .map(|p| p.name.as_str())
            .collect();
        for p in &f.points {
            for n in &numerics {
                let v = p.values.get(*n).map(String::as_str).unwrap_or(UNDECLARED);
                if v != UNDECLARED && !trees.contains_key(v) {
                    return Err(format!(
                        "family `{}`: numerics parameter `{n}` = `{v}` names no [[reduction]]",
                        f.id
                    ));
                }
            }
        }
    }
    Ok(())
}
