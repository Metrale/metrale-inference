// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: `met circuit venn --hardware <device>`: the kernel Venn with every side planned
//! on a device's class. Each compared model is fused with the class's rules (its FUSIONS.toml
//! overlay) over the kernels the device can run, under the policy after the class's defaults;
//! nodes are costed with the device's roofline; and only the class's own microbench evidence
//! counts, so on a class without records everything shared is "shared, unmeasured".
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - Pure: every byte arrives through the [`KernelTree`] and the [`Registry`].
//! - The instances, manifest drift check and measurements are the offline Venn's
//!   ([`crate::venn::repo::prepare`]); only the planning, families and roofline differ.
//! - A compared op no rule of the class covers is a placeholder group on the device, not a
//!   usage: the compared model does not run it there.
//! - The target is matched to the class's families by op (its device plan may hold placeholder
//!   groups no family runs).
//! - The families are the union of every side's: a family is kept per side only where that
//!   side's kernel sources compile it, and each compared plan names its own side's kernels.

use std::collections::BTreeMap;

use super::model::{model_of, policy_on_class};
use super::plan::{NOVEL_EMITTER, fuse_on, resolve};
use super::{HwError, KernelTree, PrecisionChoice, Registry};
use crate::Loaded;
use crate::fuser::FusionPlan;
use crate::instances::Instance;
use crate::venn::report::{OnDevice, Side, VennInputs};
use crate::venn::{Families, VennArgs, build, render};

/// 2026-10-05: One side resolved on the device: its plans per run, its settings and families.
struct DeviceSide {
    plans: Vec<FusionPlan>,
    settings: BTreeMap<String, String>,
    families: Families,
}

fn on_device(
    tree: &dyn KernelTree,
    registry: &Registry,
    device: &str,
    inst: &Instance,
    args: &VennArgs,
) -> Result<DeviceSide, HwError> {
    let model = model_of(
        tree.as_repo(),
        inst,
        format!("recipe {}", inst.recipe),
        PrecisionChoice::Recipe,
    )?;
    let resolved = resolve(registry, device, tree, &model)?;
    let (policy, _) = policy_on_class(
        &model.policy,
        model.settings_class.as_deref(),
        &resolved.chain[0],
    )?;
    let runs = args.runs().map_err(|e| HwError::Plan(e.to_string()))?;
    let mut plans = Vec::with_capacity(runs.len());
    for run in runs {
        let mut plan = fuse_on(&resolved, &model.circuit, &policy, run)?.plan;
        plan.groups.retain(|g| g.emitter != NOVEL_EMITTER);
        plans.push(plan);
    }
    let mut families = resolved.families.clone();
    families.roofline = resolved.roofline.roofline;
    Ok(DeviceSide {
        plans,
        settings: policy.settings,
        families,
    })
}

fn side<'a>(instance: &'a Instance, loaded: &'a Loaded, d: &'a DeviceSide) -> Side<'a> {
    Side {
        instance,
        loaded,
        on_device: Some(OnDevice {
            plans: &d.plans,
            settings: &d.settings,
        }),
    }
}

/// 2026-10-05: The Venn report `args` asks for, every side planned on `args.hardware`.
/// `checkpoint` is as for [`crate::venn::report_text`].
///
/// # Errors
/// What the offline report refuses, an unknown device, a class that cannot be resolved for a
/// side, or a plan that cannot be built on the device.
pub fn venn_text(
    tree: &dyn KernelTree,
    registry: &Registry,
    args: &VennArgs,
    checkpoint: Option<(&str, Option<&str>)>,
) -> Result<String, HwError> {
    let Some(device) = args.hardware.as_deref() else {
        return Err(HwError::Plan(
            "venn_text plans on a device: --hardware is required".into(),
        ));
    };
    let venn = |e: crate::venn::VennError| HwError::Plan(e.to_string());
    let p = crate::venn::repo::prepare(tree.as_repo(), args, checkpoint).map_err(venn)?;
    let target = on_device(tree, registry, device, &p.target, args)?;
    let against = p
        .against
        .iter()
        .map(|a| on_device(tree, registry, device, a, args))
        .collect::<Result<Vec<_>, _>>()?;
    let class = &registry.device(device)?.class;
    let mut families = target.families.clone();
    for d in &against {
        for f in &d.families.families {
            if !families.families.iter().any(|g| g.id == f.id) {
                families.families.push(f.clone());
            }
        }
    }
    let inputs = VennInputs {
        target: side(&p.target, &p.target_loaded, &target),
        against: p
            .against
            .iter()
            .zip(&p.against_loaded)
            .zip(&against)
            .map(|((i, l), d)| side(i, l, d))
            .collect(),
        families: &families,
        measurements: &p.measurements,
        runs: args.runs().map_err(venn)?,
        command: args.command(),
        device: Some(format!(
            "`{device}` (class `{class}`: its rules, the kernels it can run, its roofline, its own evidence only)"
        )),
    };
    Ok(render(&build(&inputs).map_err(venn)?))
}
