// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: Tensor-parallel attention on the mock. Under a world above one, `wo_b` writes f32
//! partials into the reduction's buffer and the cross-rank combine (no expert rows) writes the
//! layer output once, gathering exactly the call's rows; on one device `wo_b` writes the output
//! directly and nothing is gathered. A missing or mismatched communicator is refused, and no
//! tensor-parallel layer is offered for graph capture.
//!
//! Owner: model-arch, DeepSeek-V4.1.
//! Invariants: none beyond the types.

use super::*;
use crate::deepseek_v41_layer::levers::tests::at_defaults;
use crate::tp_reduce_v41::tests::RecordingComm;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use std::sync::Mutex;

const DIM: usize = 256;
const WO_B: DevicePtr = DevicePtr(0x7_0000);

pub(super) fn cfg(tp_world: usize) -> AttnV41Cfg {
    AttnV41Cfg {
        dim: DIM,
        n_heads: 4,
        head_dim: 32,
        rope_dim: 16,
        q_rank: 64,
        o_rank: 16,
        groups: 2,
        window: 8,
        eps: 1e-6,
        index_heads: 2,
        index_hd: 32,
        index_topk: 4,
        cand_topk_blocks: 1,
        cand_block: 1,
        max_seq: 32,
        max_tokens: 4,
        rope_theta: 10000.0,
        compress_rope_theta: 160000.0,
        rope_factor: 16.0,
        orig_seq: 16,
        beta_fast: 32.0,
        beta_slow: 1.0,
        fp8_stage: Fp8StageSize::NONE,
        tp_world,
        kv: kv_codec::declared().unwrap(),
    }
}

/// 2026-10-19: A ratio-0 layer (window attention only, no host work mid-forward). The mock runs
/// no kernel, so the weights are bare addresses.
fn layer() -> AttnV41LayerWeights {
    let mat = |a: u64| AttnMat::Bf16(DevicePtr(a));
    AttnV41LayerWeights {
        role: LayerRole::default(),
        sink: DevicePtr(0x1_0000),
        wq_a: mat(0x2_0000),
        q_norm: DevicePtr(0x3_0000),
        wq_b: mat(0x4_0000),
        wkv: mat(0x5_0000),
        kv_norm: DevicePtr(0x6_0000),
        wo_a: mat(0x8_0000),
        wo_b: AttnMat::Bf16(WO_B),
        comp: None,
        idx: None,
    }
}

fn comm(world: usize) -> RecordingComm {
    RecordingComm {
        world,
        gathers: Mutex::new(Vec::new()),
    }
}

fn ptr(a: &MockArg) -> DevicePtr {
    match a {
        MockArg::Buffer(p) => *p,
        other => panic!("{other:?}"),
    }
}

fn word(a: &MockArg) -> u64 {
    match a {
        MockArg::Bytes(b) if b.len() == 4 => u32::from_le_bytes(b[..].try_into().unwrap()) as u64,
        MockArg::Bytes(b) => u64::from_le_bytes(b[..8].try_into().unwrap()),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_tp_forward_projects_wo_b_into_the_partial_and_sums_it_into_the_output() {
    let g = MockGpuBackend::new();
    let attn = AttnV41::new(&g, cfg(3), at_defaults()).unwrap();
    let w = layer();
    let mut st = AttnV41LayerState::new(&g, &attn.cfg, w.role).unwrap();
    let mut shared = SharedV41::default();
    let x = g.alloc(2 * DIM * 2).unwrap();
    let c = comm(3);
    let base = g.launches_snapshot().len();
    let run = attn
        .forward(&g, &w, &mut st, &mut shared, x, 2, 0, Some(&c), 0)
        .unwrap();
    assert_eq!(run.out, attn.out_ptr());
    let l = &g.launches_snapshot()[base..];
    let (wo_b, combine) = (&l[l.len() - 2], &l[l.len() - 1]);
    let partial = attn.tp.as_ref().unwrap().reduce.partial();
    assert_eq!(ptr(&wo_b.args[1]), WO_B);
    assert_eq!(ptr(&wo_b.args[2]), partial, "wo_b writes the f32 partial");
    assert_eq!(
        [3, 4, 5].map(|i| word(&wo_b.args[i])),
        [2, DIM as u64, 2 * 16],
        "m, dim, this rank's groups x o_rank"
    );
    assert_eq!(combine.grid, [1, 2, 1]);
    assert_eq!(word(&combine.args[1]), (2 * DIM) as u64, "slot elements");
    assert_eq!(word(&combine.args[2]), 3, "world");
    assert_eq!(
        (word(&combine.args[4]), word(&combine.args[5])),
        (0, 0),
        "no expert rows: each token's one row is its own"
    );
    assert_eq!(ptr(&combine.args[6]), attn.out_ptr());
    let gathered = ptr(&combine.args[0]);
    assert_ne!(gathered, partial);
    assert_eq!(
        *c.gathers.lock().unwrap(),
        [(partial.0, gathered.0, 2 * DIM * 4)],
        "one gather of exactly this call's rows"
    );
}

#[test]
fn one_device_projects_wo_b_straight_into_the_output() {
    let g = MockGpuBackend::new();
    let attn = AttnV41::new(&g, cfg(1), at_defaults()).unwrap();
    assert!(attn.tp.is_none());
    let w = layer();
    let mut st = AttnV41LayerState::new(&g, &attn.cfg, w.role).unwrap();
    let mut shared = SharedV41::default();
    let x = g.alloc(2 * DIM * 2).unwrap();
    let run = attn
        .forward(&g, &w, &mut st, &mut shared, x, 2, 0, None, 0)
        .unwrap();
    let last = g.launches_snapshot().pop().unwrap();
    assert_eq!(ptr(&last.args[1]), WO_B);
    assert_eq!(ptr(&last.args[2]), run.out);
    assert!(attn.decode_capturable(&w));
}

#[test]
fn a_missing_or_mismatched_communicator_is_refused_and_no_tp_layer_is_captured() {
    let g = MockGpuBackend::new();
    let attn = AttnV41::new(&g, cfg(3), at_defaults()).unwrap();
    let w = layer();
    let mut st = AttnV41LayerState::new(&g, &attn.cfg, w.role).unwrap();
    let mut shared = SharedV41::default();
    let x = g.alloc(DIM * 2).unwrap();
    let e = attn
        .forward(&g, &w, &mut st, &mut shared, x, 1, 0, None, 0)
        .err()
        .expect("refused");
    assert!(e.to_string().contains("without a communicator"), "{e}");
    let e = attn
        .forward(&g, &w, &mut st, &mut shared, x, 1, 0, Some(&comm(2)), 0)
        .err()
        .expect("refused");
    assert!(e.to_string().contains("spans 2 ranks"), "{e}");
    assert!(!attn.decode_capturable(&w));
    let e = attn.decode_body(&g, &w, &st, x, None, 0).unwrap_err();
    assert!(e.to_string().contains("cannot be captured"), "{e}");
    assert!(AttnV41::new(&g, cfg(0), at_defaults()).is_err());
}
