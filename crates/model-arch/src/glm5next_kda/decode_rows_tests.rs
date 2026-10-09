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
        conv_decode: k(CONV),
        conv_prefill: k(0x307),
        l2: k(0x308),
        gate: k(0x309),
        chunk_prepare: k(0x30A),
        chunk_scan: k(0x30D),
        recurrent: k(RECUR),
        recurrent_smem: k(RECUR_SMEM),
        recurrent_smem_rows: k(RECUR_ROWS),
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
    l.decode_rows(&gpu, hidden, &states, &ws, 7).unwrap();
    let launches = since(&gpu, from);
    let cd = cfg().conv_dim();

    let conv: Vec<_> = launches.iter().filter(|x| x.func == CONV).collect();
    assert_eq!(conv.len(), 3);
    for (r, c) in conv.iter().enumerate() {
        assert_eq!(
            c.args[0],
            MockArg::Buffer(states[r].conv),
            "row {r}'s conv state"
        );
        assert_eq!(c.args[1], MockArg::Buffer(ws.qkv_proj.offset(r * cd * 2)));
    }
    // 2026-10-09: One recurrent launch for the three rows: grid z is the row, and the state
    // arguments are the rows' states in order, then zeros.
    assert!(
        recurrent_launches(&launches).is_empty(),
        "no per-row recurrent launch"
    );
    let rows: Vec<_> = launches.iter().filter(|x| x.func == RECUR_ROWS).collect();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].grid[2], 3);
    let ptrs = &rows[0].args[14..];
    assert_eq!(ptrs.len(), KDA_ROWS_MAX);
    for (r, a) in ptrs.iter().enumerate() {
        let want = states.get(r).map_or(0, |s| s.recurrent.0);
        assert_eq!(
            *a,
            MockArg::Bytes(want.to_le_bytes().to_vec()),
            "state argument {r}"
        );
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
    l.decode_rows(&gpu, hidden, &[st; 16], &ws, 7).unwrap();
    assert!(gpu.launch_count() > from, "16 rows fit a 16-row workspace");
}
