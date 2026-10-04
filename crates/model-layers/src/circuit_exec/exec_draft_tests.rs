// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: The executor's MTP draft program (one row of the draft head), compiled from the
//! real dense circuit over a synthetic draft binding and run on the recording mock backend:
//! launch counts, the buffers it reads (the draft head's own weights, cache and metadata), and
//! the refusal of a draft head the circuit does not model.
//!
//! Owner: model-layers circuit executor.
//! Invariants: none beyond the types.

use metrale_circuit::{LinearRole, Mode};
use metrale_gpu_runtime::gpu::mock::MockArg;

use super::Fusions;
use super::bindings::*;
use super::exec_fixture::*;

fn draft() -> Fixture {
    build_at(Fusions::All, Mode::Draft, 1, |_| {}, |_| {}).unwrap_or_else(|e| panic!("{e:#}"))
}

#[test]
fn the_draft_step_launches_its_plan_on_the_heads_own_weights_and_cache() {
    let f = draft();
    assert_eq!(f.program.launches.len() as u64, f.plan.launches());
    let launched = run(&f, &[], 3);
    assert_eq!(launched.len() as u64, f.plan.launches());
    assert_pointers_known(&f, &[], &launched);
    let d = f.fixed.draft.as_ref().unwrap();
    let reads = |p: u64| {
        launched
            .iter()
            .any(|l| l.args.contains(&MockArg::Buffer(ptr(p))))
    };
    assert!(
        reads(d.k_pool.0) && reads(d.v_pool.0),
        "the draft cache is written and read"
    );
    assert!(
        reads(d.meta.block_table.0),
        "the draft attention reads the draft metadata"
    );
    assert!(
        !reads(f.fixed.k_pools[0].0),
        "the target cache is untouched"
    );
    let lm_head = match f.draft.as_ref().unwrap().weights[&WeightSlot::LmHead] {
        BoundWeight::Nvfp4(q) => q.weight.0,
        _ => unreachable!(),
    };
    assert!(reads(lm_head), "the draft lm_head is the head's own");
    // 2026-09-29: `forward_one` leaves the draft logits in the model's logits buffer, where the
    // host reads them; an arena edge would leave that buffer stale.
    let head = launched
        .iter()
        .find(|l| l.args.contains(&MockArg::Buffer(ptr(lm_head))))
        .unwrap();
    assert!(
        head.args.contains(&MockArg::Buffer(f.fixed.logits)),
        "the draft logits land in the model's logits buffer"
    );
    let scored = MockArg::Bytes(d.vocab.to_le_bytes().to_vec());
    let head_and_argmax = launched.iter().filter(|l| l.args.contains(&scored)).count();
    assert_eq!(
        head_and_argmax, 2,
        "the lm_head and the argmax cover the draft's rows"
    );
    assert!(!reads(0x9100_0000), "the target lm_head is untouched");
}

#[test]
fn a_draft_binding_the_plan_does_not_describe_is_refused() {
    let f = draft();
    let mut d = f.draft.clone().unwrap();
    d.weights
        .insert(WeightSlot::Linear(LinearRole::Q), nvfp4(tag(200, 40)));
    let err = format!("{:#}", compile_draft(d).unwrap_err());
    assert!(err.contains("resolved") || err.contains("dense"), "{err}");
}

/// 2026-09-29: The draft plan compiled over `d` in place of the fixture's binding.
fn compile_draft(d: CircuitLayer) -> anyhow::Result<()> {
    let f = draft();
    let gpu = metrale_gpu_runtime::gpu::mock::MockGpuBackend::new();
    let inst = super::sources::instance(RECIPE)?;
    let loaded = metrale_circuit::load(&inst, super::sources::sources(&inst)?)?;
    let table = super::kernels::KernelTable::resolve(
        &gpu,
        &metrale_circuit::AvailableKernels::all_named_by(&loaded.rules),
    );
    let layout = super::compile::layout(&f.circuit, &f.plan)?;
    let buffers = metrale_circuit::planner::plan_buffers_with(&f.circuit, &f.plan, 1, &layout)?;
    let cfg = config();
    super::compile::compile(
        &f.circuit,
        &f.plan,
        &layout,
        &buffers,
        ptr(WORKSPACE),
        &super::compile::Inputs {
            gpu: &gpu,
            config: &cfg,
            kernels: &table,
            fixed: &f.fixed,
            layers: &f.layers,
            head: &f.head,
            draft: Some(&d),
            arena: None,
            lane: None,
        },
    )
    .map(|_| ())
}
