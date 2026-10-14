// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Host tests of `decode_seqs_tokens` on the mock backend: the three launches per
//! 16 runs with each run's state, first row, row count and snapshot slots in its argument
//! position (zeros after), no per-row stateful launch, and the refusals before any launch. The
//! numerics are checked on a GPU by `examples/glm5next_kda_seqs_gate.rs`.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};

use super::*;
use crate::glm5next_kda::decode_rows_tests::{
    CONV_SEQS, CONV_TOKENS, CONV_WINDOW, RECUR_SEQ, RECUR_SEQS, WINDOW_SEQS, cfg, layer, seq_state,
    since,
};

fn u64a(v: u64) -> MockArg {
    MockArg::Bytes(v.to_le_bytes().to_vec())
}

fn u32a(v: u32) -> MockArg {
    MockArg::Bytes(v.to_le_bytes().to_vec())
}

/// 2026-10-14: The `KDA_SEQS_MAX` slots of a per-run argument starting at `at`.
fn slots(l: &MockLaunch, at: usize) -> &[MockArg] {
    &l.args[at..at + KDA_SEQS_MAX]
}

fn padded64(v: &[u64]) -> Vec<MockArg> {
    (0..KDA_SEQS_MAX)
        .map(|r| u64a(v.get(r).copied().unwrap_or(0)))
        .collect()
}

fn padded32(v: &[u32]) -> Vec<MockArg> {
    (0..KDA_SEQS_MAX)
        .map(|r| u32a(v.get(r).copied().unwrap_or(0)))
        .collect()
}

fn stateful(l: &[MockLaunch], f: u64) -> Vec<MockLaunch> {
    l.iter().filter(|x| x.func == f).cloned().collect()
}

/// 2026-10-14: Three runs of 2, 1 and 3 rows, the first snapshotted: one launch of each seqs
/// kernel, grid z (y for the window) the run count, grid y of the conv the longest run; each
/// run's conv state, recurrent state, snapshot slots, first row (0, 2, 3) and rows in its
/// slot, zeros after; the recurrent strides as the per-sequence kernel's.
#[test]
fn one_launch_each_with_every_runs_arguments_in_its_slot() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 16).unwrap();
    let states: Vec<KdaSeqState> = (0..3).map(|_| seq_state(&gpu)).collect();
    let snap = (gpu.alloc(64).unwrap(), gpu.alloc(64).unwrap());
    let runs: Vec<KdaSeqRun> = states
        .iter()
        .zip([2usize, 1, 3])
        .enumerate()
        .map(|(i, (&state, rows))| KdaSeqRun {
            state,
            rows,
            snapshot: (i == 0).then_some(snap),
        })
        .collect();
    let hidden = gpu.alloc(6 * 256 * 2).unwrap();
    let from = gpu.launch_count();
    l.decode_seqs_tokens(&gpu, hidden, &runs, &ws, 7).unwrap();
    let launches = since(&gpu, from);
    for f in [CONV_TOKENS, CONV_WINDOW, RECUR_SEQ] {
        assert!(
            stateful(&launches, f).is_empty(),
            "per-sequence kernel {f:#x}"
        );
    }
    let (conv, window, recur) = (
        stateful(&launches, CONV_SEQS),
        stateful(&launches, WINDOW_SEQS),
        stateful(&launches, RECUR_SEQS),
    );
    assert_eq!((conv.len(), window.len(), recur.len()), (1, 1, 1));
    let (conv, window, recur) = (&conv[0], &window[0], &recur[0]);
    let c = cfg();
    let cd = c.conv_dim() as u32;
    assert_eq!(conv.grid, [cd.div_ceil(256), 3, 3]);
    assert_eq!(window.grid, [cd.div_ceil(256), 3, 1]);
    assert_eq!(recur.grid, [c.heads as u32, 4, 3]);
    assert_eq!((recur.block, recur.shared_mem), ([32, 1, 1], 3 * 128 * 4));
    let conv_ptrs: Vec<u64> = states.iter().map(|s| s.conv.0).collect();
    let rec_ptrs: Vec<u64> = states.iter().map(|s| s.recurrent.0).collect();
    let (w, n) = (padded32(&[0, 2, 3]), padded32(&[2, 1, 3]));
    assert_eq!(conv.args[0], MockArg::Buffer(ws.qkv_proj));
    assert_eq!(conv.args[3], MockArg::Buffer(ws.conv_out));
    assert_eq!(slots(conv, 9), padded64(&conv_ptrs));
    assert_eq!((slots(conv, 25), slots(conv, 41)), (&w[..], &n[..]));
    assert_eq!(window.args[0], MockArg::Buffer(ws.qkv_proj));
    assert_eq!(slots(window, 3), padded64(&conv_ptrs));
    assert_eq!(slots(window, 19), padded64(&[snap.1.0]));
    assert_eq!((slots(window, 35), slots(window, 51)), (&w[..], &n[..]));
    assert_eq!(recur.args[5], MockArg::Buffer(ws.core));
    let u = |v: usize| u32a(v as u32);
    assert_eq!(
        recur.args[8..12],
        [u(c.conv_dim()), u(c.qkv_dim()), u(c.heads), u(c.qkv_dim())]
    );
    assert_eq!(slots(recur, 12), padded64(&rec_ptrs));
    assert_eq!(slots(recur, 28), padded64(&[snap.0.0]));
    assert_eq!((slots(recur, 44), slots(recur, 60)), (&w[..], &n[..]));
    assert_eq!(recur.args.len(), 76);
}

/// 2026-10-14: 17 one-row-and-more runs split 16 + 1: two launches of each kernel, the
/// second's first row the 17th run's absolute workspace row.
#[test]
fn seventeen_runs_take_two_launches_with_absolute_rows() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 40).unwrap();
    let runs: Vec<KdaSeqRun> = (0..17)
        .map(|i| KdaSeqRun {
            state: seq_state(&gpu),
            rows: 1 + i % 2,
            snapshot: None,
        })
        .collect();
    let total: usize = runs.iter().map(|r| r.rows).sum();
    let hidden = gpu.alloc(total * 256 * 2).unwrap();
    let from = gpu.launch_count();
    l.decode_seqs_tokens(&gpu, hidden, &runs, &ws, 7).unwrap();
    let launches = since(&gpu, from);
    let recur = stateful(&launches, RECUR_SEQS);
    assert_eq!(recur.len(), 2);
    assert_eq!((recur[0].grid[2], recur[1].grid[2]), (16, 1));
    let first16: usize = runs[..16].iter().map(|r| r.rows).sum();
    assert_eq!(slots(&recur[1], 44), padded32(&[first16 as u32]));
    assert_eq!(
        slots(&recur[1], 12),
        padded64(&[runs[16].state.recurrent.0])
    );
    assert_eq!(stateful(&launches, CONV_SEQS).len(), 2);
    assert_eq!(stateful(&launches, WINDOW_SEQS).len(), 2);
}

/// 2026-10-14: Runs it cannot step correctly are refused before any launch: none, an empty
/// run, two runs on one recurrent or one conv state, an unset state or snapshot pointer.
#[test]
fn bad_runs_are_refused_before_any_launch() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 16).unwrap();
    let hidden = gpu.alloc(8 * 256 * 2).unwrap();
    let (a, b) = (seq_state(&gpu), seq_state(&gpu));
    let run = |state, rows, snapshot| KdaSeqRun {
        state,
        rows,
        snapshot,
    };
    let half = KdaSeqState {
        conv: a.conv,
        recurrent: b.recurrent,
    };
    let unset = KdaSeqState {
        conv: a.conv,
        recurrent: DevicePtr::NULL,
    };
    let cases: Vec<Vec<KdaSeqRun>> = vec![
        vec![],
        vec![run(a, 0, None)],
        vec![run(a, 2, None), run(a, 2, None)],
        vec![run(a, 2, None), run(half, 2, None)],
        vec![run(unset, 2, None)],
        vec![run(a, 2, Some((DevicePtr::NULL, b.conv)))],
    ];
    for (i, runs) in cases.iter().enumerate() {
        let from = gpu.launch_count();
        assert!(
            l.decode_seqs_tokens(&gpu, hidden, runs, &ws, 7).is_err(),
            "case {i} accepted"
        );
        assert_eq!(gpu.launch_count(), from, "case {i} launched");
    }
}
