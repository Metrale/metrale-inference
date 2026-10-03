// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The executor's n-row MTP draft programs (the batched propose), compiled from the
//! real dense circuit over the synthetic draft binding and run on the recording mock backend:
//! the launches at each tier edge, the metadata and buffers they read, and the refusals that
//! keep a width the plan does not describe from running.
//!
//! Owner: model-layers circuit executor.
//! Invariants: none beyond the types.

use metrale_circuit::Mode;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::gpu::mock::MockArg;

use super::Fusions;
use super::exec_fixture::*;
use super::program::{DraftPrograms, DraftRunner};

/// 2026-09-30: The widths at the edges of the tensor-core tiers and the LM head's batch32 tier.
const WIDTHS: [u64; 7] = [2, 5, 8, 9, 16, 17, 32];

fn draft_rows(n: u64) -> Fixture {
    build_at(Fusions::All, Mode::Draft, n, |_| {}, |_| {})
        .unwrap_or_else(|e| panic!("{n} rows: {e:#}"))
}

#[test]
fn each_width_launches_its_plan_on_the_batched_metadata_and_the_heads_cache() {
    for n in WIDTHS {
        let f = draft_rows(n);
        assert_eq!(
            f.program.launches.len() as u64,
            f.plan.launches(),
            "{n} rows"
        );
        let launched = run(&f, &[], 3);
        assert_eq!(launched.len() as u64, f.plan.launches(), "{n} rows");
        assert_pointers_known(&f, &[], &launched);
        let d = f.fixed.draft.as_ref().unwrap();
        let reads = |p: DevicePtr| {
            launched
                .iter()
                .any(|l| l.args.contains(&MockArg::Buffer(p)))
        };
        let m = d.meta_rows(n).unwrap();
        for (what, p) in [
            ("positions", m.positions),
            ("slots", m.slot),
            ("sequence lengths", m.seq_len),
            ("block tables", m.block_table),
        ] {
            assert!(reads(p), "{n} rows: the batched {what} are read");
        }
        assert!(
            !reads(d.meta.block_table),
            "{n} rows: the single-row metadata is not read"
        );
        assert!(
            reads(d.k_pool) && reads(d.v_pool),
            "{n} rows: the draft cache"
        );
        assert!(
            !reads(f.fixed.k_pools[0]),
            "{n} rows: the target cache is untouched"
        );
        let lp = f.fixed.tokens.offset(d.rows.as_ref().unwrap().lp_offset);
        let argmax = launched
            .iter()
            .find(|l| l.args.contains(&MockArg::Buffer(lp)));
        assert!(
            argmax.is_some_and(|l| l.args.contains(&MockArg::Buffer(f.fixed.tokens))),
            "{n} rows: one launch writes the ids at scratch and the confidences at the offset"
        );
        let concats = f
            .program
            .launches
            .iter()
            .filter(|l| l.kernel.ends_with("::bf16_concat"))
            .count();
        assert_eq!(
            concats as u64, n,
            "{n} rows: one concat per row, as the head runs it"
        );
    }
}

#[test]
fn the_projection_and_lm_head_kernels_follow_the_heads_tiers() {
    let tier = |n: u64, func: &str| {
        draft_rows(n)
            .program
            .launches
            .iter()
            .filter(|l| l.kernel.ends_with(&format!("::{func}")))
            .count()
    };
    // 2026-09-30: fc, q, k, v, o, gate, up, down: eight tensor-core GEMVs of the tier covering n.
    for (n, func) in [
        (2, "dense_gemv_bf16_tc8"),
        (8, "dense_gemv_bf16_tc8"),
        (9, "dense_gemv_bf16_tc16"),
        (16, "dense_gemv_bf16_tc16"),
        (17, "dense_gemv_bf16_tc32"),
        (32, "dense_gemv_bf16_tc32"),
    ] {
        assert_eq!(tier(n, func), 8, "{n} rows on {func}");
    }
    for (n, func) in [
        (5, "w4a16_gemv_tc8"),
        (9, "w4a16_gemv_tc16"),
        (17, "w4a16_gemv_batch32"),
    ] {
        assert_eq!(tier(n, func), 1, "the {n}-row LM head on {func}");
    }
}

#[test]
fn a_head_outside_the_batched_propose_gets_no_n_row_program() {
    let err = build_for_draft(|f| f.rows = None).unwrap_err();
    assert!(format!("{err:#}").contains("batched propose"), "{err:#}");
}

/// 2026-10-01: The fixture head scores 100 000 of the vocabulary's rows (`mtp_vocab_size`, as
/// the 27B checkpoint's head does): the LM head writes them packed that far apart and the
/// argmax reads them so, never at the logits edge's full width.
#[test]
fn a_head_with_a_smaller_vocabulary_packs_its_rows_as_the_head_does() {
    let f = draft_rows(4);
    let d = f.fixed.draft.as_ref().unwrap();
    let full = f.circuit.dims["vocab"] as u32;
    assert!(d.vocab < full);
    let launched = run(&f, &[], 3);
    let lp = f.fixed.tokens.offset(d.rows.as_ref().unwrap().lp_offset);
    let scored = MockArg::Bytes(d.vocab.to_le_bytes().to_vec());
    let wide = MockArg::Bytes(full.to_le_bytes().to_vec());
    let argmax = launched
        .iter()
        .find(|l| l.args.contains(&MockArg::Buffer(lp)))
        .unwrap();
    assert_eq!(
        argmax.args.iter().filter(|a| **a == scored).count(),
        2,
        "the argmax reads the scored rows at their packed stride"
    );
    assert!(
        launched.iter().all(|l| !l.args.contains(&wide)),
        "nothing spans the full width"
    );
    let head = launched
        .iter()
        .find(|l| l.args.contains(&MockArg::Buffer(f.fixed.logits)) && l.args.contains(&scored));
    assert!(
        head.is_some(),
        "the LM head writes the scored rows into the logits buffer"
    );
}

#[test]
fn a_head_that_takes_the_tile_twin_is_refused_where_the_plan_states_the_gemv() {
    // 2026-09-30: 17 rows: no tensor-core LM head, so a ready twin is what the head would run.
    let err = build_for_draft_at(17, |f| f.rows.as_mut().unwrap().lm_head_twin = true).unwrap_err();
    assert!(format!("{err:#}").contains("twin"), "{err:#}");
}

#[test]
fn the_runner_serves_exactly_its_compiled_widths() {
    let programs = DraftPrograms {
        programs: [1, 4]
            .map(|n| {
                let f = build_at(Fusions::All, Mode::Draft, n, |_| {}, |_| {}).unwrap();
                (f.program, f.plan)
            })
            .into(),
    };
    assert!(programs.serves(1) && programs.serves(4));
    assert!(!programs.serves(2) && !programs.serves(5));
    let gpu = metrale_gpu_runtime::gpu::mock::MockGpuBackend::new();
    gpu.set_kernel_n_tile(metrale_gpu_runtime::gpu::KernelHandle(0xDEAD), 128);
    let err = programs.run_draft(&gpu, 7, 3, 1).unwrap_err();
    assert!(format!("{err:#}").contains("3 rows"), "{err:#}");
    programs.run_draft(&gpu, 7, 4, 1).unwrap();
    assert!(!gpu.launches_snapshot().is_empty());
}

/// 2026-09-30: The 4-row draft compiled with the draft head's fixed buffers edited by `edit`.
fn build_for_draft(edit: impl Fn(&mut super::DraftFixed)) -> anyhow::Result<()> {
    build_for_draft_at(4, edit)
}

fn build_for_draft_at(n: u64, edit: impl Fn(&mut super::DraftFixed)) -> anyhow::Result<()> {
    let f = draft_rows(n);
    let mut fixed = f.fixed.clone();
    edit(fixed.draft.as_mut().unwrap());
    let gpu = metrale_gpu_runtime::gpu::mock::MockGpuBackend::new();
    let inst = super::sources::instance(RECIPE)?;
    let loaded = metrale_circuit::load(&inst, super::sources::sources(&inst)?)?;
    let table = super::kernels::KernelTable::resolve(
        &gpu,
        &metrale_circuit::AvailableKernels::all_named_by(&loaded.rules),
    );
    let layout = super::compile::layout(&f.circuit, &f.plan)?;
    let buffers = metrale_circuit::planner::plan_buffers_with(&f.circuit, &f.plan, n, &layout)?;
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
            fixed: &fixed,
            layers: &f.layers,
            head: &f.head,
            draft: f.draft.as_ref(),
            arena: None,
        },
    )
    .map(|_| ())
}
