// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: [`affected`] over a repository reached through [`Repo`]: every hardware's
//! KERNEL_FAMILIES.toml, and each INSTANCES.toml instance's family usage from its fused plans.
//!
//! Owner: metrale-circuit (venn).
//! Invariants:
//! - Business logic only: every byte arrives through [`Repo`] (SBIO).
//! - A golden instance's usage is the union over every checked-in plan (each mode and row
//!   count, each batched-verify table) and every runtime route's arm, fused exactly as the
//!   golden tests fuse them; a non-golden one's is [`op_usage`].
//! - Fail-closed: an instance that does not load or fuse, or whose hardware has no manifest,
//!   has an unknown usage (it runs every family on its hardware) with the reason recorded.

use super::affected::{Affected, InstanceUse, Usage, affected, op_usage, plan_usage};
use super::repo::load_instance;
use super::{Families, Repo, VennError, parse_families};
use crate::fuser::{AvailableKernels, fuse, fuse_table};
use crate::instances::{Instance, parse_instances};
use crate::rules::Mode;
use crate::runtime::route_arms;

/// 2026-10-15: The manifest path of one hardware directory.
fn manifest_path(hw: &str) -> String {
    format!("kernels/{hw}/common/KERNEL_FAMILIES.toml")
}

/// 2026-10-15: Every hardware's parsed manifest, found by listing `kernels/`.
pub fn manifests(repo: &dyn Repo) -> Result<Vec<Families>, VennError> {
    let io = VennError::Load;
    let mut out = Vec::new();
    for path in repo.list("kernels").map_err(io)? {
        let parts: Vec<&str> = path.split('/').collect();
        if let ["kernels", hw, "common", "KERNEL_FAMILIES.toml"] = parts.as_slice() {
            let fams = parse_families(&repo.read(&path).map_err(io)?)
                .map_err(|e| VennError::Load(format!("{path}: {e}")))?;
            if fams.hardware != *hw {
                return Err(VennError::Load(format!(
                    "{path} declares hardware `{}`",
                    fams.hardware
                )));
            }
            out.push(fams);
        }
    }
    Ok(out)
}

/// 2026-10-15: The families `inst` runs, given its hardware's manifest.
pub fn instance_usage(repo: &dyn Repo, inst: &Instance, fams: Option<&Families>) -> Usage {
    let Some(fams) = fams else {
        let hw = inst.target.split('/').next().unwrap_or_default();
        return Usage::Unknown(format!("no {} on this tree", manifest_path(hw)));
    };
    let l = match load_instance(repo, inst) {
        Ok(l) => l,
        Err(e) => return Usage::Unknown(format!("does not load: {e}")),
    };
    if !inst.golden {
        return op_usage(&l.circuit, fams);
    }
    let avail = AvailableKernels::all_named_by(&l.rules);
    let mut plans = Vec::new();
    let mut fail = None;
    for (&mode, rows) in inst.plans.iter().filter(|(m, _)| **m != Mode::VerifyBatch) {
        for &n in rows {
            match fuse(&l.circuit, &l.rules, &avail, &inst.policy, mode, n)
                .map_err(|e| e.to_string())
            {
                Ok(p) => plans.push(p),
                Err(e) => fail = Some(format!("{} n{n} does not fuse: {e}", mode.name())),
            }
        }
    }
    for t in &inst.verify_batch {
        match fuse_table(&l.circuit, &l.rules, &avail, &inst.policy, t).map_err(|e| e.to_string()) {
            Ok(p) => plans.push(p),
            Err(e) => fail = Some(format!("a verify_batch table does not fuse: {e}")),
        }
    }
    let set = (l.rules.as_slice(), l.runtime.as_slice());
    let mut arms = Vec::new();
    for p in &plans {
        match route_arms(&l.circuit, set, &avail, &inst.policy, p).map_err(|e| e.to_string()) {
            Ok(a) => arms.extend(a.into_iter().map(|(_, arm)| arm)),
            Err(e) => fail = Some(format!("a runtime route does not fuse: {e}")),
        }
    }
    if let Some(why) = fail {
        return Usage::Unknown(why);
    }
    plans
        .iter()
        .chain(&arms)
        .fold(Usage::Known(Default::default()), |acc, p| {
            acc.union(plan_usage(&l.circuit, p, fams))
        })
}

/// 2026-10-15: Every instance's usage on this tree, with the manifests it was derived from.
pub fn instance_uses(repo: &dyn Repo) -> Result<(Vec<Families>, Vec<InstanceUse>), VennError> {
    let io = VennError::Load;
    let all = parse_instances(&repo.read("kernels/circuits/INSTANCES.toml").map_err(io)?)
        .map_err(|e| VennError::Load(e.to_string()))?;
    let fams = manifests(repo)?;
    let uses = all
        .iter()
        .map(|inst| {
            let hw = inst.target.split('/').next().unwrap_or_default();
            InstanceUse {
                recipe: inst.recipe.clone(),
                checkpoint: inst.checkpoint.clone(),
                target: inst.target.clone(),
                usage: instance_usage(repo, inst, fams.iter().find(|f| f.hardware == hw)),
            }
        })
        .collect();
    Ok((fams, uses))
}

/// 2026-10-15: The models `changed` reaches on the tree `repo` reads.
pub fn affected_in(repo: &dyn Repo, changed: &[String]) -> Result<Affected, VennError> {
    let (fams, uses) = instance_uses(repo)?;
    Ok(affected(changed, &fams, &uses))
}
