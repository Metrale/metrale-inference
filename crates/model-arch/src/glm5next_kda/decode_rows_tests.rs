// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Host tests of `Glm5NextKdaLayer::decode_rows` on the mock backend: row `r`'s
//! conv and recurrent step take sequence `r`'s state and workspace row `r`, one row launches
//! exactly what `decode` launches, `decode_k` over one state launches what `decode_rows`
//! launches over that state repeated, and the row-count refusals come before any launch. The
//! numerics are checked on a GPU by `examples/glm5next_multi_seq_decode_gate.rs`.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};
use metrale_model_layers::weight_map::DenseWeight;

use super::*;
use crate::glm5next_kda::KDA_ROWS_MAX;

const BATCHM: u64 = 0x305;
const CONV: u64 = 0x306;
const RECUR: u64 = 0x30B;
const RECUR_SMEM: u64 = 0x30C;
const RECUR_ROWS: u64 = 0x313;
const CONV_ROWS: u64 = 0x316;
const CONV_TOKENS: u64 = 0x317;
const CONV_WINDOW: u64 = 0x318;
const RECUR_SEQ: u64 = 0x319;

fn cfg() -> Glm5NextKdaConfig {
    Glm5NextKdaConfig {
        hidden: 256,
        heads: 2,
        head_dim: 128,
        conv_kernel: 4,
        gate_lower_bound: -5.0,
        rms_norm_eps: 1e-6,
        l2_eps: 1e-6,
        chunk: 32,
    }
}

fn layer(gpu: &MockGpuBackend) -> Glm5NextKdaLayer {
    let k = KernelHandle;
    let w = || DenseWeight {
        weight: gpu.alloc(64).unwrap(),
    };
    let weights = Glm5NextKdaWeights {
        q_proj: w(),
        k_proj: w(),
        v_proj: w(),
        conv: w(),
        f_a: w(),
        f_b: w(),
        dt_bias: gpu.alloc(64).unwrap(),
        a_log: gpu.alloc(64).unwrap(),
        b_proj: w(),
        g_a: w(),
        g_b: w(),
        o_norm: w(),
        o_proj: w(),
    };
    let kernels = Glm5NextKdaKernels {
        gemm: k(0x301),
        gemv: k(0x302),
        gemv_batchm: k(BATCHM),
        gemv_batchm_wide: k(0x314),
        conv_decode: k(CONV),
        conv_decode_rows: k(CONV_ROWS),
        conv_prefill: k(0x307),
        l2: k(0x308),
        gate: k(0x309),
        chunk_prepare: k(0x30A),
        chunk_scan: k(0x30D),
        recurrent: k(RECUR),
        recurrent_smem: k(RECUR_SMEM),
        recurrent_smem_rows: k(RECUR_ROWS),
        recurrent_rows_reg: k(0x315),
        seq: crate::glm5next_kda::KdaSeqKernels {
            conv_tokens: k(CONV_TOKENS),
            conv_window: k(CONV_WINDOW),
            recurrent: k(RECUR_SEQ),
        },
        o_norm: k(0x30E),
        split_widen: k(0x30F),
        sigmoid: k(0x310),
        fill: k(0x311),
        pack: k(0x312),
    };
    Glm5NextKdaLayer::new(3, cfg(), weights, kernels).unwrap()
}

fn seq_state(gpu: &MockGpuBackend) -> KdaSeqState {
    let c = cfg();
    KdaSeqState {
        conv: gpu.alloc(c.conv_state_elems() * 4).unwrap(),
        recurrent: gpu.alloc(c.recurrent_state_elems() * 4).unwrap(),
    }
}

fn since(gpu: &MockGpuBackend, from: usize) -> Vec<MockLaunch> {
    gpu.launches_snapshot()[from..].to_vec()
}

/// 2026-10-08: The recurrent kernel this target launches (the 1R+1W one, unless the
/// environment selects the other).
fn recurrent_launches(l: &[MockLaunch]) -> Vec<MockLaunch> {
    l.iter()
        .filter(|x| x.func == RECUR || x.func == RECUR_SMEM)
        .cloned()
        .collect()
}

fn key(l: &MockLaunch) -> (u64, [u32; 3], [u32; 3], u32, &Vec<MockArg>) {
    (l.func, l.grid, l.block, l.shared_mem, &l.args)
}

/// 2026-10-08: Row `r`'s conv reads workspace row `r` and updates sequence `r`'s conv state,
/// and its recurrent step updates sequence `r`'s recurrent state, in row order. The
/// projections run once at M = 3.
#[test]
fn each_row_steps_only_its_own_sequence_state() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 16).unwrap();
    let states: Vec<KdaSeqState> = (0..3).map(|_| seq_state(&gpu)).collect();
    let hidden = gpu.alloc(3 * 256 * 2).unwrap();
    let from = gpu.launch_count();
    let seqs: Vec<(KdaSeqState, usize)> = states.iter().map(|s| (*s, 1)).collect();
    l.decode_seq_rows_with(&gpu, hidden, &seqs, &ws, 7, |_| Ok(()), true)
        .unwrap();
    let launches = since(&gpu, from);

    let u64a = |v: u64| MockArg::Bytes(v.to_le_bytes().to_vec());
    let u32a = |v: u32| MockArg::Bytes(v.to_le_bytes().to_vec());
    // 2026-10-09: One conv launch for the three rows: grid y is the row; the window arguments
    // are the rows' conv states in order, then zeros; the row arguments 0, 1, 2, then zeros.
    assert!(
        launches.iter().all(|x| x.func != CONV),
        "no per-row conv launch"
    );
    let conv: Vec<_> = launches.iter().filter(|x| x.func == CONV_ROWS).collect();
    assert_eq!(conv.len(), 1);
    assert_eq!(conv[0].grid[1], 3);
    for r in 0..KDA_ROWS_MAX {
        let want = states.get(r).map_or(0, |s| s.conv.0);
        assert_eq!(conv[0].args[9 + r], u64a(want), "conv state argument {r}");
        let row = if r < 3 { r as u32 } else { 0 };
        assert_eq!(
            conv[0].args[9 + KDA_ROWS_MAX + r],
            u32a(row),
            "conv row {r}"
        );
    }
    // 2026-10-09: One recurrent launch for the three rows: grid z is the row, and the state
    // arguments are the rows' states in order, then zeros, then the rows.
    assert!(
        recurrent_launches(&launches).is_empty(),
        "no per-row recurrent launch"
    );
    let rows: Vec<_> = launches.iter().filter(|x| x.func == RECUR_ROWS).collect();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].grid[2], 3);
    for r in 0..KDA_ROWS_MAX {
        let want = states.get(r).map_or(0, |s| s.recurrent.0);
        assert_eq!(rows[0].args[14 + r], u64a(want), "state argument {r}");
        let row = if r < 3 { r as u32 } else { 0 };
        assert_eq!(rows[0].args[14 + KDA_ROWS_MAX + r], u32a(row), "row {r}");
    }
    // 2026-10-09: The row strides: q/k/v by `conv_dim`, gate and out by `qkv`, beta by heads.
    let u = |v: usize| MockArg::Bytes((v as u32).to_le_bytes().to_vec());
    let c = cfg();
    assert_eq!(
        rows[0].args[10..14],
        [u(c.conv_dim()), u(c.qkv_dim()), u(c.heads), u(c.qkv_dim())]
    );
    let proj: Vec<_> = launches.iter().filter(|x| x.func == BATCHM).collect();
    assert!(!proj.is_empty());
    for p in proj {
        assert_eq!(p.args[3], MockArg::Bytes(3u32.to_le_bytes().to_vec()));
    }
}

/// 2026-10-08: At one row, `decode_rows` launches exactly what `decode` launches.
#[test]
fn one_row_launches_what_decode_launches() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 16).unwrap();
    let st = seq_state(&gpu);
    let hidden = gpu.alloc(256 * 2).unwrap();
    let from = gpu.launch_count();
    l.decode(&gpu, hidden, &st, &ws, 7).unwrap();
    let single = since(&gpu, from);
    let from = gpu.launch_count();
    l.decode_rows(&gpu, hidden, &[st], &ws, 7).unwrap();
    let rows = since(&gpu, from);
    assert_eq!(rows.len(), single.len());
    for (a, b) in rows.iter().zip(&single) {
        assert_eq!(key(a), key(b));
    }
}

/// 2026-10-08: `decode_k` over one sequence's `k` rows, without snapshots, launches what
/// `decode_rows` launches with that sequence's state on every row when the rows kernel is
/// absent (2026-10-09): the two share
/// `rows_with`, and `decode_k` adds only the snapshot copies.
#[test]
fn decode_k_is_decode_rows_over_one_repeated_state() {
    let gpu = MockGpuBackend::new();
    let mut l = layer(&gpu);
    l.kernels.recurrent_smem_rows = KernelHandle(0);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 16).unwrap();
    let st = seq_state(&gpu);
    let hidden = gpu.alloc(4 * 256 * 2).unwrap();
    let from = gpu.launch_count();
    l.decode_k(&gpu, hidden, 4, &st, &ws, &[], 7).unwrap();
    let k_rows = since(&gpu, from);
    let from = gpu.launch_count();
    l.decode_rows(&gpu, hidden, &[st; 4], &ws, 7).unwrap();
    let rows = since(&gpu, from);
    assert_eq!(rows.len(), k_rows.len());
    for (a, b) in rows.iter().zip(&k_rows) {
        assert_eq!(key(a), key(b));
    }
}

/// 2026-10-08: No rows, or more rows than the workspace holds, are refused before any launch.
#[test]
fn row_counts_outside_the_workspace_are_refused_before_any_launch() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 16).unwrap();
    let hidden = gpu.alloc(17 * 256 * 2).unwrap();
    let st = seq_state(&gpu);
    let from = gpu.launch_count();
    assert!(l.decode_rows(&gpu, hidden, &[], &ws, 7).is_err());
    assert!(l.decode_rows(&gpu, hidden, &[st; 17], &ws, 7).is_err());
    assert_eq!(gpu.launch_count(), from);
    // 2026-10-09: Sixteen distinct sequences: a rows launch refuses two rows on one state.
    let distinct: Vec<KdaSeqState> = (0..16).map(|_| seq_state(&gpu)).collect();
    l.decode_rows(&gpu, hidden, &distinct, &ws, 7).unwrap();
    assert!(gpu.launch_count() > from, "16 rows fit a 16-row workspace");
}

/// 2026-10-09: A verify of sequences with 3, 1 and 2 rows steps row 0 of all three, then row 1
/// of the first and third, then row 2 of the first: each launch holds one row per sequence, a
/// sequence's rows in order, and `after_row` runs for a launch's rows before the next launch.
#[test]
fn verify_rows_step_t_major_one_row_per_sequence_per_launch() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 16).unwrap();
    let s: Vec<KdaSeqState> = (0..3).map(|_| seq_state(&gpu)).collect();
    let hidden = gpu.alloc(6 * 256 * 2).unwrap();
    let from = gpu.launch_count();
    let mut after = Vec::new();
    let seqs = [(s[0], 3usize), (s[1], 1), (s[2], 2)];
    l.decode_seq_rows_with(
        &gpu,
        hidden,
        &seqs,
        &ws,
        7,
        |row| {
            after.push((row, gpu.launch_count()));
            Ok(())
        },
        true,
    )
    .unwrap();
    let launches = since(&gpu, from);
    let rows: Vec<_> = launches.iter().filter(|x| x.func == RECUR_ROWS).collect();
    let u32a = |v: u32| MockArg::Bytes(v.to_le_bytes().to_vec());
    let u64a = |v: u64| MockArg::Bytes(v.to_le_bytes().to_vec());
    let want: [&[(usize, usize)]; 3] = [&[(0, 0), (3, 1), (4, 2)], &[(1, 0), (5, 2)], &[(2, 0)]];
    assert_eq!(rows.len(), 3);
    for (launch, w) in rows.iter().zip(want) {
        assert_eq!(launch.grid[2] as usize, w.len());
        for (z, &(row, seq)) in w.iter().enumerate() {
            assert_eq!(launch.args[14 + z], u64a(s[seq].recurrent.0));
            assert_eq!(launch.args[14 + KDA_ROWS_MAX + z], u32a(row as u32));
        }
    }
    let order: Vec<usize> = after.iter().map(|&(r, _)| r).collect();
    assert_eq!(order, vec![0, 3, 4, 1, 5, 2], "after_row in launch order");
    // 2026-10-09: Each after_row sees its launch done and the next not yet issued.
    let recur_at: Vec<usize> = gpu
        .launches_snapshot()
        .iter()
        .enumerate()
        .filter(|(_, x)| x.func == RECUR_ROWS)
        .map(|(i, _)| i + 1)
        .collect();
    for (k, &(_, at)) in after.iter().enumerate() {
        let launch = if k < 3 {
            0
        } else if k < 5 {
            1
        } else {
            2
        };
        assert!(at >= recur_at[launch], "after_row {k} before its launch");
        if launch < 2 {
            assert!(
                at < recur_at[launch + 1],
                "after_row {k} after the next launch"
            );
        }
    }
}

/// 2026-10-09: Twenty one-row sequences take two launches, 16 rows then 4.
#[test]
fn rows_launches_hold_at_most_sixteen_rows() {
    let gpu = MockGpuBackend::new();
    let st: Vec<(KdaSeqState, usize)> = (0..20).map(|_| (seq_state(&gpu), 1)).collect();
    let launches = super::rows::t_major_launches(&st);
    assert_eq!(
        launches.iter().map(Vec::len).collect::<Vec<_>>(),
        vec![16, 4]
    );
    assert_eq!(launches[1][0].row, 16);
}

/// 2026-10-09: Padding rows of a batched decode share the pool's dummy state. They run one per
/// launch after the real rows, never two in a launch (which the rows kernels would race on and
/// `check_launch` refuses), and a real state shared by two entries steps them in entry order.
#[test]
fn entries_sharing_a_state_never_share_a_launch() {
    let gpu = MockGpuBackend::new();
    let (a, b, dummy) = (seq_state(&gpu), seq_state(&gpu), seq_state(&gpu));
    let seqs = [(a, 1), (dummy, 1), (b, 1), (dummy, 1), (dummy, 1)];
    let launches = super::rows::t_major_launches(&seqs);
    let rows: Vec<Vec<usize>> = launches
        .iter()
        .map(|l| l.iter().map(|r| r.row).collect())
        .collect();
    assert_eq!(rows, vec![vec![0, 1, 2], vec![3], vec![4]]);
    for l in &launches {
        let mut seen: Vec<u64> = l.iter().map(|r| r.state.recurrent.0).collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), l.len(), "two rows of one state in a launch");
    }
    // 2026-10-09: Two two-row entries on one state: all of the first, then all of the second.
    let launches = super::rows::t_major_launches(&[(a, 2), (b, 1), (a, 2)]);
    let rows: Vec<Vec<usize>> = launches
        .iter()
        .map(|l| l.iter().map(|r| r.row).collect())
        .collect();
    assert_eq!(rows, vec![vec![0, 2], vec![1], vec![3], vec![4]]);
}

/// 2026-10-09: A padded batched decode (16 real rows and the dummy state on 4 padding rows)
/// runs to completion through `decode_rows` instead of being refused.
#[test]
fn a_padded_decode_group_is_not_refused() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 20).unwrap();
    let dummy = seq_state(&gpu);
    let mut states: Vec<KdaSeqState> = (0..12).map(|_| seq_state(&gpu)).collect();
    states.extend([dummy; 4]);
    let hidden = gpu.alloc(16 * 256 * 2).unwrap();
    let seqs: Vec<(KdaSeqState, usize)> = states.iter().map(|s| (*s, 1)).collect();
    l.decode_seq_rows_with(&gpu, hidden, &seqs, &ws, 7, |_| Ok(()), true)
        .unwrap();
}

/// 2026-10-09: Without the opt-in the batched decode steps row by row: one single-row conv
/// and one single-row recurrent launch per row, no rows launch.
#[test]
fn the_default_steps_row_by_row() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 16).unwrap();
    let states: Vec<KdaSeqState> = (0..3).map(|_| seq_state(&gpu)).collect();
    let seqs: Vec<(KdaSeqState, usize)> = states.iter().map(|s| (*s, 1)).collect();
    let hidden = gpu.alloc(3 * 256 * 2).unwrap();
    let from = gpu.launch_count();
    l.decode_seq_rows_with(&gpu, hidden, &seqs, &ws, 7, |_| Ok(()), false)
        .unwrap();
    let launches = since(&gpu, from);
    assert_eq!(launches.iter().filter(|x| x.func == CONV).count(), 3);
    assert_eq!(recurrent_launches(&launches).len(), 3);
    assert!(
        launches
            .iter()
            .all(|x| x.func != CONV_ROWS && x.func != RECUR_ROWS)
    );
}

/// 2026-10-09: With the token kernels, `k` rows of one sequence without snapshots step in one
/// conv launch (grid y = k, reading workspace row 0 on and the sequence's conv state), one
/// window advance (k as its token count) and one recurrence (k tokens, the sequence's state),
/// and no per-row conv or recurrence; with snapshots, or with one row, they step row by row.
#[test]
fn token_kernels_step_a_sequence_in_three_launches_without_snapshots() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 16).unwrap();
    let st = seq_state(&gpu);
    let hidden = gpu.alloc(16 * cfg().hidden * 2).unwrap();
    let from = gpu.launch_count();
    l.decode_k_with(&gpu, hidden, 7, &st, &ws, &[], 0, true)
        .unwrap();
    let got = since(&gpu, from);
    let of = |f: u64| {
        got.iter()
            .filter(|x| x.func == f)
            .cloned()
            .collect::<Vec<_>>()
    };
    let (conv, win, rec) = (of(CONV_TOKENS), of(CONV_WINDOW), of(RECUR_SEQ));
    assert_eq!((conv.len(), win.len(), rec.len()), (1, 1, 1));
    assert_eq!(conv[0].grid[1], 7);
    assert_eq!(conv[0].args[0], MockArg::Buffer(st.conv));
    assert_eq!(conv[0].args[1], MockArg::Buffer(ws.qkv_proj));
    assert_eq!(win[0].args[0], MockArg::Buffer(st.conv));
    assert_eq!(win[0].args[4], MockArg::Bytes(7u32.to_le_bytes().to_vec()));
    assert_eq!(rec[0].args[5], MockArg::Buffer(st.recurrent));
    assert_eq!(rec[0].args[9], MockArg::Bytes(7u32.to_le_bytes().to_vec()));
    assert!(of(CONV).is_empty() && recurrent_launches(&got).is_empty());

    let c = cfg();
    let snap = (
        gpu.alloc(c.recurrent_state_elems() * 4).unwrap(),
        gpu.alloc(c.conv_state_elems() * 4).unwrap(),
    );
    for (k, snaps) in [(7usize, vec![snap]), (1, vec![])] {
        let from = gpu.launch_count();
        l.decode_k_with(&gpu, hidden, k, &st, &ws, &snaps, 0, true)
            .unwrap();
        let got = since(&gpu, from);
        assert!(
            got.iter()
                .all(|x| ![CONV_TOKENS, CONV_WINDOW, RECUR_SEQ].contains(&x.func))
        );
        assert_eq!(recurrent_launches(&got).len(), k);
    }
}
