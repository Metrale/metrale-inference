// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: A rule's dim predicates (`dims = { moe_inter = { mod = 128, rem = 64 } }`): the
//! shape a kernel takes, stated on the rule, so a TP rank whose slice takes another entry point
//! (the `_k64` twins at a 704-wide expert slice, the plain entries at 640) is planned by rule
//! rather than by a comment.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - A predicate states at least one test; every stated test must hold (`eq`, `min`, `max`,
//!   `mod` with `rem`). `rem` without `mod`, `mod = 0` or `rem >= mod` is a load error.
//! - A rule whose predicate names a dim the circuit does not define is refused by the fuser where
//!   the rule's pattern matches a node of the planned section ([`crate::fuser::FuseError::DimMissing`]),
//!   never read as false (PCND).

use crate::prelude::*;

use alloc::collections::BTreeMap;

use serde::Deserialize;

/// 2026-10-23: One dim's tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DimPred {
    /// 2026-10-23: The dim equals this.
    pub eq: Option<u64>,
    /// 2026-10-23: The dim is at least this.
    pub min: Option<u64>,
    /// 2026-10-23: The dim is at most this.
    pub max: Option<u64>,
    /// 2026-10-23: `(m, r)`: the dim modulo `m` is `r`.
    pub modulo: Option<(u64, u64)>,
}

/// 2026-10-23: The file form of a [`DimPred`].
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DimPredFile {
    eq: Option<u64>,
    min: Option<u64>,
    max: Option<u64>,
    #[serde(rename = "mod")]
    modulo: Option<u64>,
    rem: Option<u64>,
}

impl DimPred {
    /// 2026-10-23: Whether `v` passes every stated test.
    pub fn holds(&self, v: u64) -> bool {
        self.eq.is_none_or(|e| v == e)
            && self.min.is_none_or(|m| v >= m)
            && self.max.is_none_or(|m| v <= m)
            && self.modulo.is_none_or(|(m, r)| v % m == r)
    }

    /// 2026-10-23: The predicate as a plan or report prints it (`%128=64`, `=704`, `>=2`, `<=16`).
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        if let Some(e) = self.eq {
            parts.push(format!("={e}"));
        }
        if let Some(m) = self.min {
            parts.push(format!(">={m}"));
        }
        if let Some(m) = self.max {
            parts.push(format!("<={m}"));
        }
        if let Some((m, r)) = self.modulo {
            parts.push(format!("%{m}={r}"));
        }
        parts.join(",")
    }
}

/// 2026-10-23: Parse a rule's `dims` table; `Err` carries the detail of a malformed predicate.
pub(crate) fn parse(
    file: BTreeMap<String, DimPredFile>,
) -> Result<BTreeMap<String, DimPred>, String> {
    let mut out = BTreeMap::new();
    for (dim, f) in file {
        let p = parse_one(&format!("dims.{dim}"), &f)?;
        out.insert(dim, p);
    }
    Ok(out)
}

/// 2026-10-23: Parse one predicate named `what` in errors (`dims.<dim>`, a collective's `ranks`).
pub(crate) fn parse_one(what: &str, f: &DimPredFile) -> Result<DimPred, String> {
    let modulo = match (f.modulo, f.rem) {
        (None, None) => None,
        (Some(m), Some(r)) if m > 0 && r < m => Some((m, r)),
        (Some(m), Some(r)) => {
            return Err(format!("{what}: mod {m} rem {r} needs 0 <= rem < mod"));
        }
        _ => return Err(format!("{what}: `mod` and `rem` go together")),
    };
    let p = DimPred {
        eq: f.eq,
        min: f.min,
        max: f.max,
        modulo,
    };
    if p == DimPred::default() {
        return Err(format!("{what}: states no test"));
    }
    Ok(p)
}

/// 2026-10-23: `Ok(true)` when every predicate holds under `dims`, `Err(dim)` naming the first dim
/// the circuit does not define.
pub fn all_hold(
    preds: &BTreeMap<String, DimPred>,
    dims: &BTreeMap<String, u64>,
) -> Result<bool, String> {
    for (d, p) in preds {
        let v = dims.get(d).ok_or_else(|| d.clone())?;
        if !p.holds(*v) {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
#[path = "rules_dims_tests.rs"]
mod rules_dims_tests;
