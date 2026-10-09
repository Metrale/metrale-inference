// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The rounding plan of one contract at one point: the family's declared pipeline
//! for the op (kernel-level declaration, else the point's, else the family's — the manifest's
//! own precedence) plus what the contract adds (reduction depths, flush-to-zero, approximate
//! functions, scale folding). References read only this.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - The formats are never restated by a contract: a pipeline the manifest does not declare,
//!   or declares `uninstantiated`, is an error, not an assumption.

use std::collections::BTreeMap;

use metrale_circuit::pipeline::declare::{ByOp, Entry, resolve};
use metrale_circuit::pipeline::{NodePipeline, StepKind, Value};
use metrale_circuit::rules::KernelId;
use metrale_circuit::venn::families::{Family, Values};

use crate::contract::{Contract, Level, ScaleFold};

/// 2026-10-09: The rounding plan of one contract at one point and reduced length.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    /// 2026-10-09: The declared pipeline (inputs, steps, outputs).
    pub pipeline: NodePipeline,
    /// 2026-10-09: Tree depth of each reduced dimension at this point.
    pub depth: BTreeMap<String, u64>,
    /// 2026-10-09: Flush-to-zero.
    pub ftz: bool,
    /// 2026-10-09: Approximate-function relative errors.
    pub approx: BTreeMap<String, f64>,
    /// 2026-10-09: Where block scales fold.
    pub scale_fold: ScaleFold,
}

/// 2026-10-09: Terms a level combines, for reduced length `k`; `None` for a malformed width.
pub fn level_width(width: &str, k: u64) -> Option<u64> {
    if let Some(d) = width.strip_prefix("k/") {
        let d: u64 = d.parse().ok().filter(|d| *d > 0)?;
        return Some(k.div_ceil(d).max(1));
    }
    width.parse().ok().filter(|w: &u64| *w > 0)
}

/// 2026-10-09: The depth of a tree of `levels` over `k` terms: a sequential level adds its
/// width, a tree level `ceil(log2 width)`.
pub fn tree_depth(levels: &[Level], k: u64) -> Option<u64> {
    let mut d = 0u64;
    for l in levels {
        let w = level_width(&l.width, k)?;
        d += match l.order.as_str() {
            "sequential" => w,
            "tree" => u64::from(64 - (w - 1).leading_zeros()) * u64::from(w > 1),
            _ => return None,
        };
    }
    Some(d)
}

fn lookup<'a>(by: &'a ByOp, op: &str) -> Option<&'a Entry> {
    by.get(op)
}

/// 2026-10-09: The pipeline `family` declares for `op` run by `kernel` at `point`.
pub fn declared(
    family: &Family,
    kernel: &str,
    op: &str,
    point: &Values,
) -> Result<NodePipeline, String> {
    let kid = parse_kernel(kernel)?;
    let by_kernel = family
        .pipeline
        .kernels
        .get(&kid)
        .and_then(|by| lookup(by, op));
    let by_point = family
        .points
        .iter()
        .filter(|p| point.iter().all(|(k, v)| p.values.get(k) == Some(v)))
        .find_map(|p| lookup(&p.pipeline, op));
    let entry = by_kernel
        .or(by_point)
        .or_else(|| lookup(&family.pipeline.family, op))
        .ok_or_else(|| format!("family `{}` declares no pipeline for `{op}`", family.id))?;
    match entry {
        Entry::Pipeline(p) => resolve(p, point),
        Entry::Uninstantiated => Err(format!(
            "family `{}` lists `{op}` as uninstantiated",
            family.id
        )),
    }
}

/// 2026-10-09: `module::function` as a kernel id.
pub fn parse_kernel(k: &str) -> Result<KernelId, String> {
    let (module, func) = k
        .split_once("::")
        .ok_or_else(|| format!("kernel `{k}` is not module::function"))?;
    Ok(KernelId {
        module: module.to_string(),
        func: func.to_string(),
    })
}

/// 2026-10-09: The plan of contract `c` over `pipeline` (from [`declared`]), with reduced lengths
/// `lens` (`k` and any other dimension the contract's reduction names).
pub fn plan(
    c: &Contract,
    pipeline: NodePipeline,
    lens: &BTreeMap<String, u64>,
) -> Result<Plan, String> {
    let mut depth = BTreeMap::new();
    for (dim, levels) in &c.reduction {
        let k = *lens.get(dim).ok_or_else(|| {
            format!("the reference gives no length for reduced dimension `{dim}`")
        })?;
        let d = tree_depth(levels, k).ok_or_else(|| format!("reduction `{dim}` is malformed"))?;
        depth.insert(dim.clone(), d);
    }
    Ok(Plan {
        pipeline,
        depth,
        ftz: c.ftz,
        approx: c.approx.clone(),
        scale_fold: c.scale_fold,
    })
}

impl Plan {
    /// 2026-10-09: The value of step `k`, if the pipeline has it.
    pub fn step(&self, k: StepKind) -> Option<&Value> {
        self.pipeline
            .steps
            .iter()
            .find(|s| s.kind == k)
            .map(|s| &s.value)
    }

    /// 2026-10-09: Depth of reduced dimension `dim`; an error names a missing declaration.
    pub fn depth_of(&self, dim: &str) -> Result<u64, String> {
        self.depth
            .get(dim)
            .copied()
            .ok_or_else(|| format!("the contract declares no reduction for `{dim}`"))
    }

    /// 2026-10-09: The relative error of approximate function `f`; an error names a missing
    /// declaration (an approximate function is never assumed exact).
    pub fn approx_of(&self, f: &str) -> Result<f64, String> {
        self.approx
            .get(f)
            .copied()
            .ok_or_else(|| format!("the contract declares no error for `{f}`"))
    }
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
