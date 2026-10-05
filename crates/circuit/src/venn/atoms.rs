// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Atom bundles: the hardware atoms a tensor-core kernel is built from, declared once
//! in KERNEL_FAMILIES.toml (`[[atom_bundle]]`) and chosen by a family point through a policy
//! parameter whose domain is `atom_bundle`. A bundle names its MMA instruction, how each operand
//! travels (global to register, to shared memory, by bulk tensor copy), the pipeline schedule,
//! the shared-memory operand layout, where the accumulator lives, and the classes that realize
//! it.
//!
//! A bundle realized by one class only (a cluster-multicast copy, a tensor-memory accumulator,
//! a block-scaled FP4 MMA) is a declared point, never a silent copy, but `met circuit lkb`
//! reports the points that use it in a separate single-class bucket of the LKB residual until a
//! second class realizes it (owner decision, 2026-10-05).
//!
//! Owner: metrale-circuit (venn).
//! Invariants:
//! - Nothing defaults: every field is stated; an unknown scope, path, schedule or accumulator
//!   is refused.
//! - `classes` lists every class that realizes the bundle; a class is listed only if it is the
//!   manifest's class or inherits it (checked where the class tree is known,
//!   `crates/circuit/tests`).

use std::collections::BTreeMap;

use serde::Deserialize;

#[cfg(test)]
#[path = "atoms_tests.rs"]
mod atoms_tests;

/// 2026-10-05: How an operand reaches the MMA.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Copy {
    /// 2026-10-05: Which operand (`weight`, `activation`, `a`, `b`).
    pub operand: String,
    /// 2026-10-05: `global_to_register`, `global_to_shared`, `shared_to_register` or
    /// `bulk_tensor` (an asynchronous tensor copy into shared memory).
    pub path: String,
    /// 2026-10-05: The instruction(s), as written in the source.
    pub inst: String,
    /// 2026-10-05: Bytes one lane moves per instruction (the box bytes for `bulk_tensor`).
    pub bytes: u32,
}

/// 2026-10-05: A declared bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomBundle {
    /// 2026-10-05: Id.
    pub id: String,
    /// 2026-10-05: The MMA instruction.
    pub mma: String,
    /// 2026-10-05: Its `m x n x k`.
    pub shape: (u32, u32, u32),
    /// 2026-10-05: `warp`, `warpgroup` or `cta`.
    pub scope: String,
    /// 2026-10-05: Operand paths.
    pub copies: Vec<Copy>,
    /// 2026-10-05: `multistage` or `warp_specialized`.
    pub schedule: String,
    /// 2026-10-05: Pipeline stages.
    pub stages: u32,
    /// 2026-10-05: Shared-memory swizzle per operand: `none`, or `b,m,s` (crates/layout).
    pub swizzle: BTreeMap<String, String>,
    /// 2026-10-05: `registers` or `tensor_memory`.
    pub accumulator: String,
    /// 2026-10-05: The classes that realize it.
    pub classes: Vec<String>,
}

impl AtomBundle {
    /// 2026-10-05: Realized by exactly one class.
    pub fn single_class(&self) -> bool {
        self.classes.len() == 1
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MmaFile {
    inst: String,
    m: u32,
    n: u32,
    k: u32,
    scope: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CopyFile {
    operand: String,
    path: String,
    inst: String,
    bytes: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScheduleFile {
    kind: String,
    stages: u32,
}

/// 2026-10-05: `[[atom_bundle]]` in KERNEL_FAMILIES.toml.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AtomBundleFile {
    id: String,
    mma: MmaFile,
    copy: Vec<CopyFile>,
    schedule: ScheduleFile,
    swizzle: BTreeMap<String, String>,
    accumulator: String,
    classes: Vec<String>,
}

fn one_of(what: &str, v: &str, allowed: &[&str]) -> Result<(), String> {
    if allowed.contains(&v) {
        Ok(())
    } else {
        Err(format!("{what} `{v}` ({})", allowed.join(" | ")))
    }
}

/// 2026-10-05: Validate the manifest's bundles, by id.
pub(super) fn bundles(files: Vec<AtomBundleFile>) -> Result<BTreeMap<String, AtomBundle>, String> {
    let mut out = BTreeMap::new();
    for f in files {
        let at = |e: String| format!("atom_bundle `{}`: {e}", f.id);
        one_of("scope", &f.mma.scope, &["warp", "warpgroup", "cta"]).map_err(at)?;
        one_of(
            "schedule",
            &f.schedule.kind,
            &["multistage", "warp_specialized"],
        )
        .map_err(at)?;
        one_of(
            "accumulator",
            &f.accumulator,
            &["registers", "tensor_memory"],
        )
        .map_err(at)?;
        if f.copy.is_empty() || f.classes.is_empty() || f.schedule.stages == 0 {
            return Err(at("states no copy, no class or zero stages".into()));
        }
        for c in &f.copy {
            one_of(
                "copy path",
                &c.path,
                &[
                    "global_to_register",
                    "global_to_shared",
                    "shared_to_register",
                    "bulk_tensor",
                ],
            )
            .map_err(at)?;
        }
        for (operand, s) in &f.swizzle {
            let ok = s == "none"
                || (s.split(',').count() == 3
                    && s.split(',').all(|x| x.trim().parse::<u32>().is_ok()));
            if !ok {
                return Err(at(format!("swizzle of `{operand}`: `{s}` (none | b,m,s)")));
            }
        }
        let b = AtomBundle {
            id: f.id.clone(),
            mma: f.mma.inst,
            shape: (f.mma.m, f.mma.n, f.mma.k),
            scope: f.mma.scope,
            copies: f
                .copy
                .into_iter()
                .map(|c| Copy {
                    operand: c.operand,
                    path: c.path,
                    inst: c.inst,
                    bytes: c.bytes,
                })
                .collect(),
            schedule: f.schedule.kind,
            stages: f.schedule.stages,
            swizzle: f.swizzle,
            accumulator: f.accumulator,
            classes: f.classes,
        };
        if out.insert(f.id.clone(), b).is_some() {
            return Err(format!("atom_bundle `{}` is declared twice", f.id));
        }
    }
    Ok(out)
}

/// 2026-10-05: The parameter domain naming bundles.
pub const DOMAIN: &str = "atom_bundle";

/// 2026-10-05: Every parameter with a domain names a known one, and every point's value of an
/// `atom_bundle` parameter names a declared bundle.
pub(super) fn check_points(
    families: &[super::Family],
    bundles: &BTreeMap<String, AtomBundle>,
) -> Result<(), String> {
    for f in families {
        for p in &f.params {
            let Some(of) = &p.of else { continue };
            if of != DOMAIN || p.kind != super::ParamKind::Policy {
                return Err(format!(
                    "family `{}`: parameter `{}` of `{of}`: only a policy parameter of `{DOMAIN}` has a domain",
                    f.id, p.name
                ));
            }
            for pt in &f.points {
                let v = pt
                    .values
                    .get(&p.name)
                    .map(String::as_str)
                    .unwrap_or_default();
                if !bundles.contains_key(v) {
                    return Err(format!(
                        "family `{}`: `{}` = `{v}` names no [[atom_bundle]]",
                        f.id, p.name
                    ));
                }
            }
        }
    }
    Ok(())
}
