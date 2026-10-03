// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The executor's batched-verify programs, compiled from the real dense circuit for
//! row tables over synthetic bindings and run on the recording mock backend: launch counts,
//! the pointers read, each run's carried launches on its own table slice and state, the
//! fragmented and uncarried runs' per-sequence launches, and the refusal of out-of-place slots.
//!
//! Owner: model-layers circuit executor.
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};

use super::exec_fixture::*;
use super::program::{GdnState, StepEnv};

/// 2026-09-30: The golden tables of the dense instance (INSTANCES.toml `verify_batch`).
const TABLES: [&str; 14] = [
    "2x2",
    "4x2",
    "3x3",
    "4x4",
    "2x8",
    "3x8",
    "4x8",
    "4x3 3x2 2x3",
    "4x2! 2x2",
    "4x8 2x24",
    "uncarried: 4x8! 2x8!",
    "uncarried: 4x8! 2x24!",
    "2x48",
    "4x32",
];

fn seqs(table: &str) -> usize {
    metrale_circuit::RowTable::parse(table).unwrap().seqs() as usize
}

fn launched_kernels<'a>(
    f: &'a Fixture,
    name: &'a str,
) -> impl Iterator<Item = (usize, usize)> + 'a {
    kernel_launches(f)
        .enumerate()
        .filter(move |(_, l)| l.kernel.ends_with(name))
        .map(|(j, l)| {
            (
                j,
                f.circuit.nodes[f.plan.groups[l.group].nodes[0]]
                    .layer
                    .unwrap(),
            )
        })
}

/// 2026-09-30: `n` sequences' states per GDN layer, as [`states_rows`] lays them out, with each
/// layer's conv windows one allocation on `gpu` (`STATE_PITCH` apart) and their rollback slots
/// allocated too, so the per-row snapshots can copy them.
fn states_batch_on(gpu: &MockGpuBackend, f: &Fixture, n: usize) -> Vec<Vec<GdnState>> {
    let pitch = STATE_PITCH as usize;
    states_rows(f, 0xD000_0000, n)
        .into_iter()
        .map(|l| {
            if l.is_empty() {
                return l;
            }
            let base = gpu.alloc(n * pitch).unwrap();
            l.into_iter()
                .enumerate()
                .map(|(i, s)| GdnState {
                    conv: base.offset(i * pitch),
                    conv_steps: std::array::from_fn(|_| gpu.alloc(pitch).unwrap()),
                    ..s
                })
                .collect()
        })
        .collect()
}

fn buffer(p: metrale_gpu_runtime::gpu::DevicePtr) -> MockArg {
    MockArg::Buffer(p)
}

// 2026-09-30: Every golden table compiles to exactly its plan's launches and copies, and each
// launch reads only bound, placed, fixed or state memory. Mutation: an emitter that launches
// one run's schedule for every run, or drops the fold, fails the count.
#[test]
fn every_golden_table_launches_what_its_plan_counts_and_reads_only_known_buffers() {
    for table in TABLES {
        let f = build_table(table).unwrap_or_else(|e| panic!("`{table}`: {e:#}"));
        let copies = f
            .program
            .launches
            .iter()
            .filter(|l| l.kind == super::program::LaunchKind::Copy)
            .count() as u64;
        assert_eq!(
            (f.program.launches.len() as u64 - copies, copies),
            (f.plan.launches(), f.plan.copies()),
            "`{table}`"
        );
        let gpu = MockGpuBackend::new();
        gpu.set_kernel_n_tile(metrale_gpu_runtime::gpu::KernelHandle(0xDEAD), 128);
        let gdn = states_batch_on(&gpu, &f, seqs(table));
        let launched = run_on(&gpu, &f, &gdn, 9);
        assert_eq!(launched.len() as u64, f.plan.launches(), "`{table}`");
        assert_pointers_known(&f, &gdn, &launched);
    }
}

// 2026-09-30: A contiguous run's carried conv reads its first sequence's conv state and its
// slice of the slot table, and its WY its slice of the layer's WY tables and engaged words; the
// lazy kernel from 8 sequences. Mutation: the batch's first sequence in place of the run's, or
// a slice at the row rather than the sequence, fails.
#[test]
fn a_contiguous_run_launches_on_its_own_slices_and_state() {
    let f = build_table("4x2 2x8").unwrap();
    let gdn = states_rows(&f, 0xD000_0000, 10);
    let launched = run(&f, &gdn, 9);
    let conv: Vec<(usize, usize)> = launched_kernels(&f, "::gdn_carry_conv").collect();
    let gdn_layers = gdn.iter().filter(|l| !l.is_empty()).count();
    assert_eq!(
        conv.len(),
        2 * gdn_layers,
        "one carried conv per run per GDN layer"
    );
    for (i, &(j, layer)) in conv.iter().enumerate() {
        let first = if i % 2 == 0 { 0usize } else { 2 };
        let args = &launched[j].args;
        assert!(
            args.contains(&buffer(gdn[layer][first].conv)),
            "run {i}: conv state"
        );
        assert!(
            args.contains(&buffer(CARRY.slot_tab.offset(first * 4))),
            "run {i}: slots"
        );
    }
    let eager = launched_kernels(&f, "::gdn_carry_wy4").count();
    let lazy = launched_kernels(&f, "::gdn_carry_wy2_lazy").count();
    assert_eq!((eager, lazy), (gdn_layers, gdn_layers));
    for (j, _) in launched_kernels(&f, "::gdn_carry_wy2_lazy") {
        let args = &launched[j].args;
        assert!(
            args.contains(&buffer(CARRY.flag.offset(2 * 4))),
            "the run's engaged words"
        );
    }
}

// 2026-09-30: A fragmented carried run folds its pending rows, then runs each sequence alone on
// its own state; an uncarried table has no fold. Mutation: reading the run's first state for
// every sequence, or folding an uncarried run, fails.
#[test]
fn a_fragmented_run_folds_then_runs_each_sequence_on_its_own_state() {
    for (table, folds) in [("4x2! 2x2", 1), ("uncarried: 4x2! 2x2!", 0)] {
        let f = build_table(table).unwrap();
        let gpu = MockGpuBackend::new();
        gpu.set_kernel_n_tile(metrale_gpu_runtime::gpu::KernelHandle(0xDEAD), 128);
        let gdn = states_batch_on(&gpu, &f, 4);
        let launched = run_on(&gpu, &f, &gdn, 9);
        let gdn_layers = gdn.iter().filter(|l| !l.is_empty()).count();
        assert_eq!(
            launched_kernels(&f, "::gdn_carry_flush").count(),
            folds * gdn_layers,
            "`{table}`"
        );
        let mut seen = std::collections::BTreeMap::<usize, usize>::new();
        for (j, layer) in launched_kernels(&f, "::causal_conv1d_update_l2norm") {
            let row = seen.entry(layer).or_default();
            let seq = if *row < 8 {
                *row / 4
            } else {
                2 + (*row - 8) / 2
            };
            assert!(
                launched[j].args.contains(&buffer(gdn[layer][seq].conv)),
                "`{table}` layer {layer} conv row {row} reads sequence {seq}'s window"
            );
            *row += 1;
        }
        assert!(
            seen.values().all(|&r| r == if folds == 1 { 8 } else { 12 }),
            "{seen:?}"
        );
    }
}

// 2026-09-30: A contiguous run whose slots are not consecutive at run time fails before it
// launches. Mutation: dropping the conv-slot check reads another sequence's window.
#[test]
fn a_contiguous_run_refuses_out_of_place_slots() {
    let f = build_table("2x4").unwrap();
    let mut gdn = states_rows(&f, 0xD000_0000, 4);
    for layer in gdn.iter_mut().filter(|l| !l.is_empty()) {
        layer[3] = GdnState {
            conv: layer[3].conv.offset(STATE_PITCH as usize),
            ..layer[3]
        };
    }
    let gpu = MockGpuBackend::new();
    gpu.set_kernel_n_tile(metrale_gpu_runtime::gpu::KernelHandle(0xDEAD), 128);
    let e = f
        .program
        .run(&StepEnv {
            gpu: &gpu,
            stream: 7,
            gdn: &gdn,
            max_blocks_per_seq: 9,
            prefill: None,
        })
        .unwrap_err();
    assert!(format!("{e:#}").contains("conv slots on"), "{e:#}");
}

/// 2026-10-01: The boot sizes the workspace over `sizing_tables` (`CircuitExec::build`): every
/// table must parse with exactly its row count and fuse for the dense instance, at every width
/// the serve admits. A run of one sequence left unmarked failed the boot at 5 rows.
#[test]
fn every_sizing_table_fuses_at_its_row_count() {
    let inst = super::sources::instance(RECIPE).unwrap();
    let loaded = metrale_circuit::load(&inst, super::sources::sources(&inst).unwrap()).unwrap();
    let avail = metrale_circuit::AvailableKernels::all_named_by(&loaded.rules);
    let tables = super::verify_batch::sizing_tables(128);
    assert_eq!(tables.len(), 125);
    for (i, t) in tables.iter().enumerate() {
        assert_eq!(t.rows(), i as u64 + 4, "`{t}`");
        super::verify_batch::arena_bytes(&loaded.circuit, &loaded.rules, &avail, &inst.policy, t)
            .unwrap_or_else(|e| panic!("`{t}`: {e:#}"));
    }
}
