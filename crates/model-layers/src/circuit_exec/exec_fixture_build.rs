// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The executor test fixture's builders (split from `exec_fixture.rs`, which keeps
//! the synthetic bindings): fuse a checked-in instance at a mode and row count, compile it over
//! bindings, on the recording mock backend.
//!
//! Owner: model-layers circuit executor.
//! Invariants: none beyond the types.

use std::collections::BTreeSet;

use metrale_circuit::planner::plan_buffers_with;
use metrale_circuit::{AvailableKernels, Circuit, Mode, Numerics};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

use super::bindings::*;
use super::compile::{self, Inputs};
use super::exec_fixture::*;
use super::kernels::KernelTable;
use super::{Fusions, sources};
use crate::weight_map::DenseWeight;

pub(super) fn build(
    fusions: Fusions,
    edit: impl Fn(&mut Vec<CircuitLayer>),
) -> anyhow::Result<Fixture> {
    build_at(fusions, Mode::Decode, 1, edit, |_| {})
}

/// 2026-09-28: Compile the `mode` plan at `rows` rows over the synthetic bindings, after `edit`
/// and `edit_head`.
pub(super) fn build_at(
    fusions: Fusions,
    mode: Mode,
    rows: u64,
    edit: impl Fn(&mut Vec<CircuitLayer>),
    edit_head: impl Fn(&mut HeadBinding),
) -> anyhow::Result<Fixture> {
    let bind = |c: &Circuit, i: usize, attn: usize| (layer_binding(c, i, attn), Vec::new());
    build_for(
        RECIPE,
        &bind,
        fusions,
        (mode, rows, Arm::Primary),
        edit,
        edit_head,
    )
}

/// 2026-09-30: Which arm of a plan a fixture compiles.
#[derive(Debug, Clone, Copy)]
pub(super) enum Arm {
    /// 2026-09-30: The plan the policy selects.
    Primary,
    /// 2026-09-30: The arm of the runtime route with this id; an error when it does not apply.
    Route(&'static str),
    /// 2026-09-30: The plan under the instance's policy with one setting changed.
    Setting(&'static str, &'static str),
    /// 2026-09-30: The batched-verify plan of this row table (`mode` is `VerifyBatch`).
    Table(&'static str),
}

/// 2026-09-30: The carried-verify buffers every fixture GDN layer binds.
pub(super) const CARRY: crate::layer::GdnCarryBinding = crate::layer::GdnCarryBinding {
    flag: DevicePtr(0xE000_0000),
    stash: DevicePtr(0xE100_0000),
    pend: DevicePtr(0xE200_0000),
    slot_tab: DevicePtr(0xE300_0000),
    seq_floats: 4096,
    conv_stash: DevicePtr(0xE400_0000),
    conv_seq_elems: 1024,
    conv_tab: DevicePtr(0xE500_0000),
};

/// 2026-09-30: [`build_at`] for the batched-verify plan of `table`.
pub(super) fn build_table(table: &'static str) -> anyhow::Result<Fixture> {
    let bind = |c: &Circuit, i: usize, attn: usize| (layer_binding(c, i, attn), Vec::new());
    let rows = metrale_circuit::RowTable::parse(table)?.rows();
    build_for(
        RECIPE,
        &bind,
        Fusions::All,
        (Mode::VerifyBatch, rows, Arm::Table(table)),
        |_| {},
        |_| {},
    )
}

/// 2026-09-30: [`build_at`] for the arm of runtime route `route`.
pub(super) fn build_route_at(
    route: &'static str,
    mode: Mode,
    rows: u64,
) -> anyhow::Result<Fixture> {
    let bind = |c: &Circuit, i: usize, attn: usize| (layer_binding(c, i, attn), Vec::new());
    build_for(
        RECIPE,
        &bind,
        Fusions::All,
        (mode, rows, Arm::Route(route)),
        |_| {},
        |_| {},
    )
}

/// 2026-09-30: [`build_at`] for the instance `recipe`, its layers bound by `bind`. The W4A4
/// kernels are prepared on the build's backend, as the model build does.
pub(super) fn build_for(
    recipe: &str,
    bind: &Bind,
    fusions: Fusions,
    plan: (Mode, u64, Arm),
    edit: impl Fn(&mut Vec<CircuitLayer>),
    edit_head: impl Fn(&mut HeadBinding),
) -> anyhow::Result<Fixture> {
    let gpu = MockGpuBackend::new();
    build_on(
        (&gpu, &config()),
        recipe,
        bind,
        fusions,
        plan,
        edit,
        edit_head,
    )
}

/// 2026-10-03: [`build_for`] on `gpu` under `cfg`, for a test that denies a kernel or edits the
/// model config before the compile.
pub(super) fn build_on(
    gpu_cfg: (&MockGpuBackend, &metrale_config::ModelConfig),
    recipe: &str,
    bind: &Bind,
    fusions: Fusions,
    plan: (Mode, u64, Arm),
    edit: impl Fn(&mut Vec<CircuitLayer>),
    edit_head: impl Fn(&mut HeadBinding),
) -> anyhow::Result<Fixture> {
    build_drafting(
        gpu_cfg,
        recipe,
        bind,
        fusions,
        plan,
        (edit, edit_head, |_| {}),
    )
}

/// 2026-10-05: [`build_on`] with the draft head's binding edited too (a MoE drafter).
pub(super) fn build_drafting(
    (gpu, cfg): (&MockGpuBackend, &metrale_config::ModelConfig),
    recipe: &str,
    bind: &Bind,
    fusions: Fusions,
    (mode, rows, arm): (Mode, u64, Arm),
    (edit, edit_head, edit_draft): (
        impl Fn(&mut Vec<CircuitLayer>),
        impl Fn(&mut HeadBinding),
        impl Fn(&mut CircuitLayer),
    ),
) -> anyhow::Result<Fixture> {
    let inst = sources::instance(recipe)?;
    let loaded = metrale_circuit::load(&inst, sources::sources(&inst)?)?;
    let mut avail = AvailableKernels::all_named_by(&loaded.rules);
    if fusions == Fusions::ReferenceOnly {
        for r in loaded
            .rules
            .iter()
            .filter(|r| matches!(r.numerics, Numerics::BitIdentical { .. }))
        {
            for k in &r.kernels {
                avail.kernels.remove(k);
            }
        }
    }
    let mut policy = inst.policy.clone();
    if let Arm::Setting(k, v) = arm {
        policy.settings.insert(k.into(), v.into());
    }
    let mut plan = match arm {
        Arm::Table(t) => metrale_circuit::fuse_table(
            &loaded.circuit,
            &loaded.rules,
            &avail,
            &policy,
            &metrale_circuit::RowTable::parse(t)?,
        )?,
        _ => metrale_circuit::fuse(&loaded.circuit, &loaded.rules, &avail, &policy, mode, rows)?,
    };
    if let Arm::Route(id) = arm {
        let set = (loaded.rules.as_slice(), loaded.runtime.as_slice());
        let arms = metrale_circuit::runtime::route_arms(
            &loaded.circuit,
            set,
            &avail,
            &inst.policy,
            &plan,
        )?;
        plan = arms
            .into_iter()
            .find(|(r, _)| r.id == id)
            .map(|(_, p)| p)
            .ok_or_else(|| anyhow::anyhow!("route `{id}` does not apply at {mode:?} {rows}"))?;
    }
    let layout = compile::layout(&loaded.circuit, &plan)?;
    let buffers = plan_buffers_with(&loaded.circuit, &plan, rows, &layout)?;
    let mut attn = 0;
    let mut w8a8_ptrs = BTreeSet::new();
    let mut layers: Vec<CircuitLayer> = (0..loaded.circuit.layer_kinds.len())
        .map(|i| {
            let (l, ptrs) = bind(&loaded.circuit, i, attn);
            attn += usize::from(matches!(l.mixer, MixerFacts::Attention(_)));
            w8a8_ptrs.extend(ptrs);
            l
        })
        .collect();
    edit(&mut layers);
    let mut head = HeadBinding {
        embed: DenseWeight {
            weight: ptr(0x8f00_0000),
        },
        final_norm: DenseWeight {
            weight: ptr(0x9000_0000),
        },
        lm_head: dense(0x9100_0000),
        unmodelled: Vec::new(),
        batchm_max_rows: 8,
        nvfp4_twin: None,
        nvfp4_rows: false,
    };
    edit_head(&mut head);
    let draft = (mode == Mode::Draft).then(|| {
        let mut d = draft_binding(&loaded.circuit);
        edit_draft(&mut d);
        d
    });
    let fixed = fixed(attn);
    crate::layers::ops::w4a4_proj::prepare(gpu)?;
    let table = KernelTable::resolve(gpu, &AvailableKernels::all_named_by(&loaded.rules));
    let program = compile::compile(
        &loaded.circuit,
        &plan,
        &layout,
        &buffers,
        ptr(WORKSPACE),
        &Inputs {
            gpu,
            config: cfg,
            kernels: &table,
            fixed: &fixed,
            layers: &layers,
            head: &head,
            draft: draft.as_ref(),
            arena: None,
            lane: None,
        },
    )?;
    Ok(Fixture {
        draft,
        circuit: loaded.circuit,
        plan,
        program,
        arena: buffers.arena_bytes,
        fixed,
        layers,
        head,
        w8a8_ptrs,
    })
}

// 2026-09-30: The run helpers live in exec_fixture_run.rs (split for the file-size cap).
