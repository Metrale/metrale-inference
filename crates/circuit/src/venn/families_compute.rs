// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: A family's `compute`, `mma` and `kernel_compute` fields into
//! [`super::FamilyCompute`] ([`crate::venn::compute`]).
//!
//! Owner: metrale-circuit (venn).
//! Invariants:
//! - A `kernel_compute` key is one of the family's kernels, and differs from the family's unit
//!   unless the family's points run on different units, in which case every kernel is named.

use std::collections::BTreeMap;

use serde::Deserialize;

use super::{ComputeUnit, FamilyCompute, Point};
use crate::rules::KernelId;

/// 2026-10-02: A compute unit as a `kernel_compute` entry states it.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ComputeFile {
    compute: String,
    mma: Option<String>,
}

/// 2026-10-02: The family's units, or what is wrong with them.
pub(super) fn family_compute(
    compute: &str,
    mma: Option<&str>,
    overrides: &BTreeMap<String, ComputeFile>,
    kernels: &[KernelId],
    points: &[Point],
) -> Result<FamilyCompute, String> {
    let unit = ComputeUnit::parse(compute, mma)?;
    // 2026-10-02: Points on another unit: no kernel's unit may be inferred from the family's.
    let points_differ = points
        .iter()
        .filter_map(|p| p.compute.as_ref())
        .any(|u| *u != unit);
    let mut by_kernel = BTreeMap::new();
    for (name, c) in overrides {
        let k = kernels
            .iter()
            .find(|k| k.to_string() == *name)
            .ok_or_else(|| format!("kernel_compute names `{name}`, which is not its kernel"))?;
        let u = ComputeUnit::parse(&c.compute, c.mma.as_deref())
            .map_err(|e| format!("kernel_compute `{name}`: {e}"))?;
        if u == unit && !points_differ {
            return Err(format!(
                "kernel_compute `{name}` repeats the family's unit `{}`",
                unit.name()
            ));
        }
        by_kernel.insert(k.clone(), u);
    }
    if points_differ {
        let unnamed: Vec<String> = kernels
            .iter()
            .filter(|k| !by_kernel.contains_key(*k))
            .map(|k| k.to_string())
            .collect();
        if !unnamed.is_empty() {
            return Err(format!(
                "its points run on different units, so kernel_compute must name the unit of {}",
                unnamed.join(", ")
            ));
        }
    }
    Ok(FamilyCompute {
        unit,
        kernels: by_kernel,
    })
}
