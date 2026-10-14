// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The launch plan of [`Glm5NextKdaLayer::prefill_cut`] on the mock backend, on the
//! per-sequence token kernels (no 64-token floor is set here): one conv over all rows with the
//! live window advanced past all of them, a conv window advance per cut with a destination, and one recurrence per part, each reading and
//! writing from its own first row with its own row count, with the state copies a destination
//! asks for. The numerics are the split segments' by the module docs; a GPU compares them.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};

use super::super::decode_rows_tests::{
    CONV_TOKENS, CONV_WINDOW, RECUR_SEQ, cfg, layer, seq_state, since,
};
use super::super::{Glm5NextKdaWorkspace, KdaSeqState};
use super::KdaCut;

fn ptr(l: &MockLaunch, i: usize) -> DevicePtr {
    match &l.args[i] {
        MockArg::Buffer(p) => *p,
        other => panic!("arg {i} is {other:?}, not a pointer"),
    }
}

fn u32a(l: &MockLaunch, i: usize) -> u32 {
    match &l.args[i] {
        MockArg::Bytes(b) => u32::from_le_bytes(b[..4].try_into().unwrap()),
        other => panic!("arg {i} is {other:?}, not 4 bytes"),
    }
}

/// 2026-10-14: `prefill_cut` of `k` rows at `cuts` (a destination where `with_dst`): its
/// stateful launches and the copies it made.
fn plan(
    k: usize,
    cuts: &[(usize, bool)],
) -> (
    Vec<MockLaunch>,
    usize,
    Glm5NextKdaWorkspace,
    Vec<KdaCut>,
    KdaSeqState,
) {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 16).unwrap();
    let st = seq_state(&gpu);
    let c = cfg();
    let hidden = gpu.alloc(k * 256 * 2).unwrap();
    let cuts: Vec<KdaCut> = cuts
        .iter()
        .map(|&(row, with_dst)| KdaCut {
            row,
            dst: with_dst.then(|| {
                (
                    gpu.alloc(c.recurrent_state_elems() * 4).unwrap(),
                    gpu.alloc(c.conv_state_elems() * 4).unwrap(),
                )
            }),
        })
        .collect();
    let (from, copies) = (gpu.launch_count(), gpu.d2d_count());
    l.prefill_cut(&gpu, hidden, k, &st, &ws, &cuts, 7).unwrap();
    let kinds = [CONV_TOKENS, CONV_WINDOW, RECUR_SEQ];
    let launches = since(&gpu, from)
        .into_iter()
        .filter(|x| kinds.contains(&x.func))
        .collect();
    (launches, gpu.d2d_count() - copies, ws, cuts, st)
}

/// 2026-10-14: 10 rows cut at 3 (kept) and 7 (not kept): the conv once over all 10 rows and
/// the live window advanced past them, one window advance to row 3 into the kept cut's conv slot, then three recurrences of 3, 4 and 3
/// rows from rows 0, 3 and 7 on the live state; two copies (the kept cut's conv window and its
/// recurrent state), none for the cut without a slot.
#[test]
fn each_part_runs_from_its_own_row_with_its_own_count() {
    let (l, copies, ws, cuts, st) = plan(10, &[(3, true), (7, false)]);
    let funcs: Vec<u64> = l.iter().map(|x| x.func).collect();
    assert_eq!(
        funcs,
        vec![
            CONV_TOKENS,
            CONV_WINDOW,
            CONV_WINDOW,
            RECUR_SEQ,
            RECUR_SEQ,
            RECUR_SEQ
        ]
    );
    let c = cfg();
    let (cd, qkv) = (c.conv_dim(), c.qkv_dim());
    let (_, conv_dst) = cuts[0].dst.unwrap();
    assert_eq!((ptr(&l[1], 0), u32a(&l[1], 4)), (st.conv, 10));
    assert_eq!(ptr(&l[2], 0), conv_dst);
    assert_eq!(ptr(&l[2], 1), ws.qkv_proj);
    assert_eq!(u32a(&l[2], 4), 3, "the window advances past the cut's rows");
    for (launch, (row0, n)) in l[3..].iter().zip([(0, 3), (3, 4), (7, 3)]) {
        let q = ws.conv_out.offset(row0 * cd * 2);
        assert_eq!(ptr(launch, 0), q, "q from row {row0}");
        assert_eq!(ptr(launch, 1), q.offset(qkv * 2), "k from row {row0}");
        assert_eq!(ptr(launch, 2), q.offset(qkv * 4), "v from row {row0}");
        assert_eq!(ptr(launch, 3), ws.gate.offset(row0 * qkv * 4));
        assert_eq!(ptr(launch, 4), ws.beta.offset(row0 * c.heads * 4));
        assert_eq!(
            ptr(launch, 5),
            st.recurrent,
            "every part steps the live state"
        );
        assert_eq!(ptr(launch, 6), ws.core.offset(row0 * qkv * 4));
        assert_eq!(u32a(launch, 9), n as u32, "{n} rows from row {row0}");
    }
    assert_eq!(copies, 2);
}

/// 2026-10-14: A cut at the last row ends the only part there: one recurrence of all rows, its
/// state copied after it; no slot anywhere is the same launches with no copy.
#[test]
fn a_cut_at_the_end_is_one_part() {
    let (l, copies, ..) = plan(6, &[(6, true)]);
    let recur: Vec<&MockLaunch> = l.iter().filter(|x| x.func == RECUR_SEQ).collect();
    assert_eq!(recur.len(), 1);
    assert_eq!(u32a(recur[0], 9), 6);
    assert_eq!(copies, 2);
    let (l, copies, ..) = plan(6, &[(2, false), (6, false)]);
    let funcs: Vec<u64> = l.iter().map(|x| x.func).collect();
    assert_eq!(funcs, vec![CONV_TOKENS, CONV_WINDOW, RECUR_SEQ, RECUR_SEQ]);
    assert_eq!(copies, 0);
}

/// 2026-10-14: A kept cut's slots receive the live state's whole conv window and recurrent
/// state (the mock moves real bytes and runs no kernel, so this pins each copy's source and size;
/// that the conv copy lands before any window launch, and the bytes the launches then write,
/// are the GPU gate's): two copies per kept cut, each of the full state.
#[test]
fn a_kept_cut_copies_the_whole_live_state() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 16).unwrap();
    let st = seq_state(&gpu);
    let c = cfg();
    let (h_bytes, conv_bytes) = (c.recurrent_state_elems() * 4, c.conv_state_elems() * 4);
    let h_pat: Vec<u8> = (0..h_bytes).map(|i| (i * 7 + 1) as u8).collect();
    let conv_pat: Vec<u8> = (0..conv_bytes).map(|i| (i * 13 + 5) as u8).collect();
    gpu.copy_h2d(&h_pat, st.recurrent).unwrap();
    gpu.copy_h2d(&conv_pat, st.conv).unwrap();
    let dst = (gpu.alloc(h_bytes).unwrap(), gpu.alloc(conv_bytes).unwrap());
    let hidden = gpu.alloc(10 * 256 * 2).unwrap();
    let cuts = [
        KdaCut {
            row: 4,
            dst: Some(dst),
        },
        KdaCut { row: 8, dst: None },
    ];
    let copies = gpu.d2d_count();
    l.prefill_cut(&gpu, hidden, 10, &st, &ws, &cuts, 7).unwrap();
    assert_eq!(gpu.d2d_count() - copies, 2);
    assert_eq!(gpu.read_alloc(dst.0).unwrap(), h_pat);
    assert_eq!(gpu.read_alloc(dst.1).unwrap(), conv_pat);
}

/// 2026-10-14: The shared recurrence refuses rows past the workspace before any launch.
#[test]
fn the_token_recurrence_refuses_rows_past_the_workspace() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 16).unwrap();
    let st = seq_state(&gpu);
    let from = gpu.launch_count();
    assert!(l.recurrence_tokens_rows(&gpu, 10, 7, &st, &ws, 7).is_err());
    assert!(l.recurrence_tokens_rows(&gpu, 0, 0, &st, &ws, 7).is_err());
    assert_eq!(gpu.launch_count(), from);
    l.recurrence_tokens_rows(&gpu, 9, 7, &st, &ws, 7).unwrap();
    assert_eq!(gpu.launch_count(), from + 1);
}
