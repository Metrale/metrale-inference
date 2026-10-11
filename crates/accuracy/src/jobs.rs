// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: From the sweep and the contracts to the checks to run, and the coverage report:
//! which swept points a contract covers, which have none (and why), and which contracted
//! kernels no described model runs.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - Nothing is skipped silently: every swept point is either planned or listed uncovered with
//!   its reason.
//! - `quick` keeps every distinct (contract, kernel, op, k, n) at its smallest and largest row
//!   count on the gaussian class (with the mutations), plus every adversarial class at the
//!   largest shape of each (contract, kernel); `full` is every point on every class.

use std::collections::{BTreeMap, BTreeSet};

use metrale_circuit::venn::families::{Families, Family, Values};

use crate::contract::{Contract, Contracts};
use crate::inputs::InputClass;
use crate::points::{Shape, Sweep};

/// 2026-10-09: How much of the sweep a run checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// 2026-10-09: The representative subset (seconds per family).
    Quick,
    /// 2026-10-09: Every point on every input class.
    Full,
}

impl Scope {
    /// 2026-10-09: Parse `quick` or `full`.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "quick" => Some(Scope::Quick),
            "full" => Some(Scope::Full),
            _ => None,
        }
    }
}

/// 2026-10-09: One planned check.
#[derive(Debug, Clone)]
pub struct Planned<'a> {
    /// 2026-10-09: The contract.
    pub contract: &'a Contract,
    /// 2026-10-09: Its family.
    pub family: &'a Family,
    /// 2026-10-09: The entry point.
    pub kernel: String,
    /// 2026-10-09: Compile-time and policy values.
    pub point: Values,
    /// 2026-10-09: Shape.
    pub shape: Shape,
    /// 2026-10-09: Input class.
    pub input: InputClass,
    /// 2026-10-09: Kernel targets that run the point (the runner needs one compiled).
    pub targets: BTreeSet<String>,
    /// 2026-10-09: Recipes that run it.
    pub users: BTreeSet<String>,
}

/// 2026-10-09: What the contracts cover.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Coverage {
    /// 2026-10-09: Swept (family, kernels, op) with no applicable contract, and why.
    pub uncovered: BTreeMap<(String, String, String), String>,
    /// 2026-10-09: Contracted (family, kernel) no swept point runs.
    pub unused: BTreeSet<(String, String)>,
    /// 2026-10-09: Swept points covered.
    pub covered_points: usize,
    /// 2026-10-09: Swept points in all.
    pub swept_points: usize,
}

fn op_matches(contract_op: &str, node_op: &str) -> bool {
    node_op == contract_op || node_op.split(':').next() == Some(contract_op)
}

fn group_kernels(kernels: &str) -> BTreeSet<String> {
    kernels.split(" + ").map(str::to_string).collect()
}

/// 2026-10-09: The checks of `scope` over `sweep`, restricted to `family` and to recipes or
/// checkpoints containing `model` when given.
pub fn plan<'a>(
    sweep: &Sweep,
    contracts: &'a Contracts,
    fams: &'a Families,
    scope: Scope,
    family: Option<&str>,
    model: Option<&str>,
) -> (Vec<Planned<'a>>, Coverage) {
    let mut cov = Coverage::default();
    let mut used: BTreeSet<(String, String)> = BTreeSet::new();
    let mut all: Vec<Planned<'a>> = Vec::new();
    for p in &sweep.points {
        if family.is_some_and(|f| f != p.family) {
            continue;
        }
        if model.is_some_and(|m| !p.users.iter().chain(&p.checkpoints).any(|u| u.contains(m))) {
            continue;
        }
        cov.swept_points += 1;
        let group = group_kernels(&p.kernels);
        let mine: Vec<&Contract> = contracts
            .contracts
            .iter()
            .filter(|c| c.family == p.family)
            .collect();
        let key = (p.family.clone(), p.kernels.clone(), p.shape.op.clone());
        if mine.is_empty() {
            cov.uncovered
                .insert(key, "no contract for the family".into());
            continue;
        }
        if p.kernels.starts_with('(') {
            cov.uncovered.insert(
                key,
                format!("no planned kernel {}: a plan-only instance", p.kernels),
            );
            continue;
        }
        let Some(f) = fams.families.iter().find(|f| f.id == p.family) else {
            cov.uncovered
                .insert(key, "family not in the manifest".into());
            continue;
        };
        let mut any = false;
        for c in mine.iter().filter(|c| op_matches(&c.op, &p.shape.op)) {
            for k in c.kernels.iter().filter(|k| group.contains(*k)) {
                any = true;
                used.insert((c.family.clone(), k.clone()));
                for input in &c.inputs {
                    all.push(Planned {
                        contract: c,
                        family: f,
                        kernel: k.clone(),
                        point: p.point.clone(),
                        shape: p.shape.clone(),
                        input: *input,
                        targets: p.targets.clone(),
                        users: p.users.clone(),
                    });
                }
            }
        }
        if any {
            cov.covered_points += 1;
        } else {
            cov.uncovered.insert(
                key,
                "the family's contracts name none of the group's kernels for this op".into(),
            );
        }
    }
    for c in contracts
        .contracts
        .iter()
        .filter(|c| family.is_none_or(|f| f == c.family))
    {
        for k in &c.kernels {
            if !used.contains(&(c.family.clone(), k.clone())) {
                cov.unused.insert((c.family.clone(), k.clone()));
            }
        }
    }
    let jobs = match scope {
        Scope::Full => all,
        Scope::Quick => quick(all),
    };
    (jobs, cov)
}

fn quick(all: Vec<Planned<'_>>) -> Vec<Planned<'_>> {
    type Shp = (usize, String, String, u64, u64);
    let ident = |p: &Planned<'_>| p.contract as *const Contract as usize;
    let mut rows: BTreeMap<Shp, (u64, u64)> = BTreeMap::new();
    let mut largest: BTreeMap<(usize, String), (u64, u64, u64)> = BTreeMap::new();
    for p in &all {
        let s = (
            ident(p),
            p.kernel.clone(),
            p.shape.op.clone(),
            p.shape.in_dim,
            p.shape.out_dim,
        );
        let e = rows.entry(s).or_insert((p.shape.rows, p.shape.rows));
        e.0 = e.0.min(p.shape.rows);
        e.1 = e.1.max(p.shape.rows);
        let size = p.shape.in_dim * p.shape.out_dim.max(1);
        let l = largest
            .entry((ident(p), p.kernel.clone()))
            .or_insert((0, 0, 0));
        if size > l.0 || (size == l.0 && p.shape.rows > l.2) {
            *l = (
                size,
                p.shape.in_dim * 1_000_000 + p.shape.out_dim,
                p.shape.rows,
            );
        }
    }
    let mut seen = BTreeSet::new();
    all.into_iter()
        .filter(|p| {
            let s = (
                ident(p),
                p.kernel.clone(),
                p.shape.op.clone(),
                p.shape.in_dim,
                p.shape.out_dim,
            );
            let (lo, hi) = rows[&s];
            if p.shape.rows != lo && p.shape.rows != hi {
                return false;
            }
            let keep = if p.input == InputClass::Gaussian {
                true
            } else {
                let l = largest[&(ident(p), p.kernel.clone())];
                p.shape.in_dim * 1_000_000 + p.shape.out_dim == l.1 && p.shape.rows == l.2
            };
            let dedupe = (s, p.shape.rows, p.input, format!("{:?}", p.point));
            keep && seen.insert(dedupe)
        })
        .collect()
}

/// 2026-10-09: Problems of `contracts` against `fams`: unknown families or kernels, a reference
/// that cannot stand for the op, an op the family declares no pipeline for. Empty when the
/// contracts are usable. (A bit_identical sibling may be any entry point the target compiles:
/// the base kernel a family's fast paths must equal is often not itself a planned point.)
pub fn validate(contracts: &Contracts, fams: &Families) -> Vec<String> {
    let mut out = Vec::new();
    for c in &contracts.contracts {
        let Some(f) = fams.families.iter().find(|f| f.id == c.family) else {
            out.push(format!("`{}`: no such family", c.family));
            continue;
        };
        let names: BTreeSet<String> = f.kernels.iter().map(|k| k.to_string()).collect();
        for k in &c.kernels {
            if !names.contains(k) {
                out.push(format!("`{}`: kernel `{k}` is not in the family", c.family));
            }
            let declared = f
                .points
                .iter()
                .any(|p| crate::plan::declared(f, k, &c.op, &p.values).is_ok());
            if !declared {
                out.push(format!(
                    "`{}`: no pipeline for `{}` run by `{k}` at any point",
                    c.family, c.op
                ));
            }
        }
        match crate::refs::Reference::parse(&c.reference) {
            None => out.push(format!("`{}`: no reference `{}`", c.family, c.reference)),
            Some(r) if !r.serves(&c.op) => out.push(format!(
                "`{}`: reference `{}` cannot stand for `{}`",
                c.family, c.reference, c.op
            )),
            Some(_) => {}
        }
    }
    out
}
