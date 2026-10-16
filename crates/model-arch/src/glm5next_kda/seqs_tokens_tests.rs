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
    CONV_SEQS, CONV_TOKENS, CONV_WINDOW, RECUR_SEQ, RECUR_SEQS, SNAPZ_RESOLVE, WINDOW_SEQS, cfg,
    layer, seq_state, since,
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
/// 2026-10-16: The third run keeps two snapshots: block t of the snapshot arguments holds each
/// run's snapshot t (run 0's one in block 0, run 2's two in blocks 0 and 1), zeros elsewhere.
#[test]
fn one_launch_each_with_every_runs_arguments_in_its_slot() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 16).unwrap();
    let states: Vec<KdaSeqState> = (0..3).map(|_| seq_state(&gpu)).collect();
    let snap = (gpu.alloc(64).unwrap(), gpu.alloc(64).unwrap());
    let two = [
        (gpu.alloc(64).unwrap(), gpu.alloc(64).unwrap()),
        (gpu.alloc(64).unwrap(), gpu.alloc(64).unwrap()),
    ];
    let defer = KdaDefer {
        flag: gpu.alloc(4).unwrap(),
        h: gpu.alloc(64).unwrap(),
        conv: gpu.alloc(64).unwrap(),
    };
    let runs: Vec<KdaSeqRun> = states
        .iter()
        .zip([2usize, 1, 3])
        .enumerate()
        .map(|(i, (&state, rows))| KdaSeqRun {
            state,
            rows,
            snapshots: match i {
                0 => KdaSnaps::new(&[snap]).unwrap(),
                2 => KdaSnaps::new(&two).unwrap(),
                _ => KdaSnaps::NONE,
            },
            defer: (i == 1).then_some(defer),
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
    let blocks = |l: &MockLaunch, at: usize, half: fn(&(DevicePtr, DevicePtr)) -> u64| {
        let want = [
            vec![half(&snap), 0, half(&two[0])],
            vec![0, 0, half(&two[1])],
        ];
        for t in 0..KDA_SEQS_SNAP_MAX {
            let v = want.get(t).cloned().unwrap_or_default();
            assert_eq!(slots(l, at + t * KDA_SEQS_MAX), padded64(&v), "block {t}");
        }
    };
    blocks(window, 19, |p| p.1.0);
    assert_eq!((slots(window, 131), slots(window, 147)), (&w[..], &n[..]));
    assert_eq!(recur.args[5], MockArg::Buffer(ws.core));
    let u = |v: usize| u32a(v as u32);
    assert_eq!(
        recur.args[8..12],
        [u(c.conv_dim()), u(c.qkv_dim()), u(c.heads), u(c.qkv_dim())]
    );
    assert_eq!(slots(recur, 12), padded64(&rec_ptrs));
    blocks(recur, 28, |p| p.0.0);
    assert_eq!((slots(recur, 140), slots(recur, 156)), (&w[..], &n[..]));
    // 2026-10-16: SNAPZ: the second run's flag in every kernel's flag slots, its deferred conv
    // snapshot as the conv and window source, its deferred recurrent snapshot as the
    // recurrent source; zeros elsewhere (a zero flag reads the live state).
    let flags = padded64(&[0, defer.flag.0]);
    let (zc, zh) = (padded64(&[0, defer.conv.0]), padded64(&[0, defer.h.0]));
    assert_eq!((slots(conv, 57), slots(conv, 73)), (&flags[..], &zc[..]));
    assert_eq!(
        (slots(window, 163), slots(window, 179)),
        (&flags[..], &zc[..])
    );
    assert_eq!(
        (slots(recur, 172), slots(recur, 188)),
        (&flags[..], &zh[..])
    );
    let lens = (conv.args.len(), window.args.len(), recur.args.len());
    assert_eq!(lens, (89, 195, 204));
}

/// 2026-10-16: SNAPZ resolve: one launch per state half (recurrent, then conv), each with the
/// flag, the deferred snapshot as source, the live state as destination and its 16-byte
/// count; refused before any launch without its kernel.
#[test]
fn snapz_resolve_copies_each_half_under_the_flag() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let state = seq_state(&gpu);
    let defer = KdaDefer {
        flag: gpu.alloc(4).unwrap(),
        h: gpu.alloc(64).unwrap(),
        conv: gpu.alloc(64).unwrap(),
    };
    let from = gpu.launch_count();
    l.snapz_resolve(&gpu, &state, &defer, 7).unwrap();
    let launches = stateful(&since(&gpu, from), SNAPZ_RESOLVE);
    assert_eq!(launches.len(), 2);
    let c = cfg();
    let halves = [
        (defer.h, state.recurrent, c.recurrent_state_elems() * 4),
        (defer.conv, state.conv, c.conv_state_elems() * 4),
    ];
    for (launch, (src, dst, bytes)) in launches.iter().zip(halves) {
        let n16 = (bytes / 16) as u64;
        assert_eq!(
            launch.args,
            [
                MockArg::Buffer(defer.flag),
                MockArg::Buffer(src),
                MockArg::Buffer(dst),
                u64a(n16)
            ]
        );
        assert_eq!(launch.grid, [n16.div_ceil(256).min(1024) as u32, 1, 1]);
        assert_eq!(launch.block, [256, 1, 1]);
    }
    let mut bare = layer(&gpu);
    bare.kernels.seq.snapz_resolve = KernelHandle(0);
    let from = gpu.launch_count();
    assert!(bare.snapz_resolve(&gpu, &state, &defer, 7).is_err());
    assert_eq!(gpu.launch_count(), from);
}

/// 2026-10-14: 17 one-row-and-more runs split 16 + 1: two launches of each kernel, the
/// second's first row the 17th run's absolute workspace row. On `seqs_chunks`, the stateful
/// half: the front end's projections at more than 16 rows take a wide route the mock does
/// not run.
#[test]
fn seventeen_runs_take_two_launches_with_absolute_rows() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 40).unwrap();
    let runs: Vec<KdaSeqRun> = (0..17)
        .map(|i| KdaSeqRun {
            state: seq_state(&gpu),
            rows: 1 + i % 2,
            snapshots: KdaSnaps::NONE,
            defer: None,
        })
        .collect();
    let from = gpu.launch_count();
    l.seqs_chunks(&gpu, &runs, &ws, 7).unwrap();
    let launches = since(&gpu, from);
    let recur = stateful(&launches, RECUR_SEQS);
    assert_eq!(recur.len(), 2);
    assert_eq!((recur[0].grid[2], recur[1].grid[2]), (16, 1));
    let first16: usize = runs[..16].iter().map(|r| r.rows).sum();
    assert_eq!(slots(&recur[1], 140), padded32(&[first16 as u32]));
    assert_eq!(
        slots(&recur[1], 12),
        padded64(&[runs[16].state.recurrent.0])
    );
    assert_eq!(stateful(&launches, CONV_SEQS).len(), 2);
    assert_eq!(stateful(&launches, WINDOW_SEQS).len(), 2);
}

/// 2026-10-14: Runs it cannot step correctly are refused before any launch: none, an empty
/// run, two runs on one recurrent or one conv state, an unset state or snapshot pointer.
/// 2026-10-16: An unset second snapshot half, and more snapshots than rows.
#[test]
fn bad_runs_are_refused_before_any_launch() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 16).unwrap();
    let hidden = gpu.alloc(8 * 256 * 2).unwrap();
    let (a, b) = (seq_state(&gpu), seq_state(&gpu));
    let run = |state, rows, snapshots| KdaSeqRun {
        state,
        rows,
        snapshots,
        defer: None,
    };
    let sn = |l: &[(DevicePtr, DevicePtr)]| KdaSnaps::new(l).unwrap();
    let deferred = |d: KdaDefer| KdaSeqRun {
        defer: Some(d),
        ..run(a, 2, KdaSnaps::NONE)
    };
    let p = gpu.alloc(64).unwrap();
    let d = KdaDefer {
        flag: p,
        h: p,
        conv: p,
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
        vec![run(a, 0, KdaSnaps::NONE)],
        vec![run(a, 2, KdaSnaps::NONE), run(a, 2, KdaSnaps::NONE)],
        vec![run(a, 2, KdaSnaps::NONE), run(half, 2, KdaSnaps::NONE)],
        vec![run(unset, 2, KdaSnaps::NONE)],
        vec![run(a, 2, sn(&[(DevicePtr::NULL, b.conv)]))],
        vec![run(a, 3, sn(&[(p, p), (p, DevicePtr::NULL)]))],
        vec![run(a, 1, sn(&[(p, p), (p, p)]))],
        // 2026-10-16: SNAPZ: a defer with any pointer unset.
        vec![deferred(KdaDefer {
            flag: DevicePtr::NULL,
            ..d
        })],
        vec![deferred(KdaDefer {
            h: DevicePtr::NULL,
            ..d
        })],
        vec![deferred(KdaDefer {
            conv: DevicePtr::NULL,
            ..d
        })],
    ];
    // 2026-10-16: The control: the same run with every defer pointer set is accepted, so each
    // refusal above is its unset pointer's.
    let from = gpu.launch_count();
    l.decode_seqs_tokens(&gpu, hidden, &[deferred(d)], &ws, 7)
        .unwrap();
    assert!(gpu.launch_count() > from);
    let from = gpu.launch_count();
    l.decode_seqs_tokens(&gpu, hidden, &[run(a, 2, sn(&[(p, p), (p, p)]))], &ws, 7)
        .unwrap();
    assert!(gpu.launch_count() > from);
    for (i, runs) in cases.iter().enumerate() {
        let from = gpu.launch_count();
        assert!(
            l.decode_seqs_tokens(&gpu, hidden, runs, &ws, 7).is_err(),
            "case {i} accepted"
        );
        assert_eq!(gpu.launch_count(), from, "case {i} launched");
    }
}

/// 2026-10-16: `KdaSnaps` keeps up to `KDA_SEQS_SNAP_MAX` slots in order and refuses one more
/// (a refused span walks in `kda_seqs_runs`, never truncates).
#[test]
fn snaps_keep_up_to_the_cap_in_order_and_refuse_more() {
    let list: Vec<(DevicePtr, DevicePtr)> = (1..=KDA_SEQS_SNAP_MAX as u64 + 1)
        .map(|i| (DevicePtr(i), DevicePtr(0x100 + i)))
        .collect();
    let s = KdaSnaps::new(&list[..KDA_SEQS_SNAP_MAX]).unwrap();
    assert_eq!(s.as_slice(), &list[..KDA_SEQS_SNAP_MAX]);
    assert!(KdaSnaps::new(&list).is_err());
    assert!(KdaSnaps::NONE.as_slice().is_empty());
}

/// 2026-10-16: The run and snapshot caps are the kernels' `#define`s: a mismatch would shift
/// every argument after the snapshot blocks.
#[test]
fn caps_match_the_kernel_defines() {
    let cu = include_str!("../../../../kernels/gb10/common/kda_seqs_tokens.cu");
    for (name, v) in [
        ("KDA_SEQS_MAX", KDA_SEQS_MAX),
        ("KDA_SEQS_SNAP_MAX", KDA_SEQS_SNAP_MAX),
    ] {
        let line = format!("#define {name} {v}\n");
        assert!(cu.contains(&line), "{line:?} not in kda_seqs_tokens.cu");
    }
}
