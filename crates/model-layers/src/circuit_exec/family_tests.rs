// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The family emitter registry over a real plan (the dense circuit's decode plan):
//! every group reaches the emitter its plan names, a group with no registered emitter is an
//! error naming it, a duplicate id is refused, and a plan kernel resolves through the device.
//!
//! Owner: model-layers circuit executor.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use anyhow::Result;
use metrale_circuit::{AvailableKernels, KernelId, Mode};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::recording::RecordingGpu;

use super::*;
use crate::circuit_exec::sources;

/// 2026-10-23: Counts the groups it was handed, by emitter id.
struct Count(&'static str);

impl GroupEmitter<BTreeMap<String, usize>> for Count {
    fn id(&self) -> &'static str {
        self.0
    }
    fn emit(&self, g: &PlanGroup<'_>, site: &mut BTreeMap<String, usize>) -> Result<()> {
        *site.entry(g.group.emitter.clone()).or_default() += 1;
        Ok(())
    }
}

fn plan() -> (metrale_circuit::Circuit, metrale_circuit::FusionPlan) {
    let inst = sources::instance("qwen3.8/qwen3.8-27b-nvfp4-unsloth").unwrap();
    let loaded = metrale_circuit::load(&inst, sources::sources(&inst).unwrap()).unwrap();
    let avail = AvailableKernels::all_named_by(&loaded.rules);
    let plan = metrale_circuit::fuse(
        &loaded.circuit,
        &loaded.rules,
        &avail,
        &inst.policy,
        Mode::Decode,
        1,
    )
    .unwrap();
    (loaded.circuit, plan)
}

/// 2026-10-23: With one counting emitter per id the plan names, every group is emitted once
/// through its own id; dropping one id makes exactly its groups an error that names the id and
/// the rule. Without the lookup by the plan's `emitter` field the counts would not match.
#[test]
fn every_group_reaches_the_emitter_its_plan_names() {
    let (c, p) = plan();
    let mut want = BTreeMap::<String, usize>::new();
    for g in &p.groups {
        *want.entry(g.emitter.clone()).or_default() += 1;
    }
    let ids: Vec<&'static str> = want
        .keys()
        .map(|k| &*Box::leak(k.clone().into_boxed_str()))
        .collect();
    let emitters: Vec<&'static dyn GroupEmitter<BTreeMap<String, usize>>> = ids
        .iter()
        .map(|&id| &*Box::leak(Box::new(Count(id))) as &'static dyn GroupEmitter<_>)
        .collect();
    let reg = EmitterRegistry::new(&emitters).unwrap();
    let mut got = BTreeMap::new();
    for g in PlanGroup::all(&c, &p) {
        reg.emit(&g, &mut got).unwrap();
    }
    assert_eq!(got, want);

    let dropped = ids[0];
    let fewer = EmitterRegistry::new(&emitters[1..]).unwrap();
    let mut site = BTreeMap::new();
    let errors: Vec<String> = PlanGroup::all(&c, &p)
        .filter_map(|g| fewer.emit(&g, &mut site).err().map(|e| e.to_string()))
        .collect();
    assert_eq!(errors.len(), want[dropped]);
    assert!(
        errors
            .iter()
            .all(|e| e.contains(&format!("`{dropped}`")) && e.contains("rule `"))
    );
}

/// 2026-10-23: Two emitters for one id are refused.
#[test]
fn a_duplicate_emitter_id_is_refused() {
    static A: Count = Count("x");
    static B: Count = Count("x");
    let e = EmitterRegistry::<BTreeMap<String, usize>>::new(&[&A, &B])
        .err()
        .unwrap();
    assert!(e.to_string().contains("`x`"), "{e}");
    assert!(EmitterRegistry::<BTreeMap<String, usize>>::new(&[&A]).is_ok());
}

/// 2026-10-23: A plan kernel resolves to the device's handle for its `(module, func)`; an absent
/// one is an error naming it.
#[test]
fn a_plan_kernel_resolves_through_the_device_or_is_named() {
    let gpu = RecordingGpu::new(48);
    let k = KernelId {
        module: "m".into(),
        func: "f".into(),
    };
    assert_eq!(
        plan_kernel(&gpu, &k).unwrap().0,
        gpu.kernel("m", "f").unwrap().0
    );
    gpu.deny("m", "g");
    let g = KernelId {
        module: "m".into(),
        func: "g".into(),
    };
    let e = plan_kernel(&gpu, &g).err().unwrap();
    assert!(format!("{e:#}").contains("m::g"), "{e:#}");
}

/// 2026-10-23: A group's typed accessors: its site, locals and kernels; a kernel index past the
/// rule's list and a wrong local list are errors.
#[test]
fn group_accessors_refuse_what_the_group_does_not_have() {
    let (c, p) = plan();
    let g = PlanGroup::all(&c, &p)
        .find(|g| g.group.kernels.len() == 1)
        .unwrap();
    assert!(g.kernel(0).is_ok());
    assert!(g.kernel(1).is_err());
    assert!(g.expect_kernels("t", 1).is_ok());
    assert!(g.expect_kernels("t", 2).is_err());
    let locals = g.locals();
    assert!(g.expect_locals("t", &locals).is_ok());
    assert!(g.expect_locals("t", &["not_a_member"]).is_err());
    assert!(g.site().starts_with(g.block()));
    assert!(g.dim("hidden").is_ok());
    assert!(g.dim("no_such_dim").is_err());
}
