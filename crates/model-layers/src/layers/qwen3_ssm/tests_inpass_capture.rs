// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Mock-GPU tests of the in-pass SSM capture split in
//! `prefill_gdn_recurrence`: a captured chunk issues exactly the launches of two
//! separate recurrences over `[0, cap)` and `[cap, k)`, at pointers derived here from
//! the row layout; under `replay_tail` the second is the exact-replay recurrence a
//! request restoring the captured state runs; and the snapshot slot receives h_state's
//! bytes.
//!
//! Owner: model-layers (Qwen3 SSM layer).
//! Invariants: none beyond the types.

use super::tests::native_fp8_gdn_layer;
use super::*;
use crate::layer::MidchunkCapture;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};

/// 2026-10-01: The launch fields a recurrence's arithmetic depends on.
fn shape(l: &MockLaunch) -> (u64, [u32; 3], [u32; 3], u32, Vec<MockArg>) {
    (l.func, l.grid, l.block, l.shared_mem, l.args.clone())
}

/// 2026-10-01: The per-chunk device buffers of one recurrence call.
struct Bufs {
    h: DevicePtr,
    qkv: DevicePtr,
    gates: DevicePtr,
    out: DevicePtr,
}

/// 2026-10-01: Run `prefill_gdn_recurrence` on `buffers` over `k` tokens of `b`, with a
/// capture at `cap` into `h_dst` when given (`replay_tail` as passed), as an exact replay
/// when `exact`, and return the launches it issued.
#[allow(clippy::too_many_arguments)]
fn run(
    gpu: &MockGpuBackend,
    config: &ModelConfig,
    buffers: &BufferArena,
    layer: &Qwen3SsmLayer,
    b: &Bufs,
    q_off: usize,
    k: u32,
    capture: Option<(usize, DevicePtr)>,
    replay_tail: bool,
    exact: bool,
) -> Vec<MockLaunch> {
    let dispatch = crate::layers::ops::GemmDispatch::defaults();
    let derived = crate::layers::ops::DerivedWeights::new();
    let levers = crate::layers::ops::ModelLevers::defaults();
    let stats = crate::layers::ops::ModelStats::new();
    let counter = std::sync::atomic::AtomicUsize::new(0);
    let h_dsts: Vec<DevicePtr> = capture.iter().map(|&(_, d)| d).collect();
    let conv_dsts = vec![DevicePtr::NULL; h_dsts.len()];
    let midchunk_capture = capture.map(|(cap, _)| MidchunkCapture {
        cap_local: cap,
        h_dsts: &h_dsts,
        conv_dsts: &conv_dsts,
        h_bytes: layer.h_state_bytes,
        conv_bytes: layer.conv_state_bytes,
        ssm_layer_counter: &counter,
        cap_local_early: None,
        h_dsts_early: &[],
        conv_dsts_early: &[],
        replay_tail,
    });
    let ctx = ForwardContext {
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        buffers,
        hc_row_offset: 0,
        gpu,
        config,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        decode_step: false,
        gdn_exact_replay: exact,
        gdn_write_on_accept: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture,
        moe_lora_route: crate::layer::MoeLoraRoute::Fold,
    };
    let nk = config.linear_num_key_heads;
    let nv = config.linear_num_value_heads;
    let kd = config.linear_key_head_dim;
    let vd = config.linear_value_head_dim;
    let key_dim = nk * kd;
    let conv_dim = key_dim * 2 + nv * vd;
    let q = b.qkv.offset(q_off);
    let before = gpu.launches_snapshot().len();
    layer
        .prefill_gdn_recurrence(
            b.h,
            q,
            q.offset(key_dim * 2),
            q.offset(key_dim * 2 * 2),
            b.gates,
            b.out,
            k,
            nk,
            nv,
            kd,
            vd,
            conv_dim,
            capture.map(|_| 0),
            &ctx,
            0,
        )
        .unwrap();
    gpu.launches_snapshot()[before..].to_vec()
}

/// 2026-10-01: The launches of a captured recurrence over `k` tokens split at `cap`, and
/// of the two recurrences it must equal: `[0, cap)`, then `[cap, k)` as an exact replay
/// when `replay_tail`.
fn captured_and_reference(
    replay_tail: bool,
) -> (Vec<MockLaunch>, Vec<MockLaunch>, Vec<MockLaunch>) {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = native_fp8_gdn_layer(&gpu, &config, true, true);
    // 2026-10-01: One arena for every call, so each run sees the same FLA scratch.
    let buffers = BufferArena::new(&config, 512, 4096, 16, 32, &gpu).unwrap();
    let (k, cap) = (299usize, 272usize);
    let nv = config.linear_num_value_heads;
    let vd = config.linear_value_head_dim;
    let conv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2 + nv * vd;
    let b = Bufs {
        h: gpu.alloc(layer.h_state_bytes).unwrap(),
        qkv: gpu.alloc(k * conv_dim * 2).unwrap(),
        gates: gpu.alloc(k * nv * 2 * 4).unwrap(),
        out: gpu.alloc(k * nv * vd * 2).unwrap(),
    };
    let h_dst = gpu.alloc(layer.h_state_bytes).unwrap();
    let marker: Vec<u8> = (0..layer.h_state_bytes)
        .map(|i| (i * 7 % 251) as u8)
        .collect();
    gpu.copy_h2d(&marker, b.h).unwrap();

    let captured = run(
        &gpu,
        &config,
        &buffers,
        &layer,
        &b,
        0,
        k as u32,
        Some((cap, h_dst)),
        replay_tail,
        false,
    );
    assert_eq!(
        gpu.read_alloc(h_dst).unwrap(),
        marker,
        "the snapshot slot must receive h_state's bytes"
    );

    // 2026-10-01: The reference: a recurrence over [0, cap), then one over [cap, k) whose
    // rows start `cap` rows into each buffer (conv rows `conv_dim` BF16, gate rows
    // `2 * nv` FP32, output rows `nv * vd` BF16).
    let head = run(
        &gpu, &config, &buffers, &layer, &b, 0, cap as u32, None, false, false,
    );
    let tail = Bufs {
        h: b.h,
        qkv: b.qkv,
        gates: b.gates.offset(cap * nv * 2 * 4),
        out: b.out.offset(cap * nv * vd * 2),
    };
    let tail_run = |exact| {
        run(
            &gpu,
            &config,
            &buffers,
            &layer,
            &tail,
            cap * conv_dim * 2,
            (k - cap) as u32,
            None,
            false,
            exact,
        )
    };
    let mut reference = head.clone();
    reference.extend(tail_run(replay_tail));
    let mut other = head;
    other.extend(tail_run(!replay_tail));
    (captured, reference, other)
}

#[test]
fn a_replay_tail_capture_runs_the_tail_as_the_exact_replay_a_warm_request_runs() {
    let (captured, reference, fla_tail) = captured_and_reference(true);
    assert!(!reference.is_empty(), "the recurrence launched nothing");
    assert_ne!(
        reference.iter().map(shape).collect::<Vec<_>>(),
        fla_tail.iter().map(shape).collect::<Vec<_>>(),
        "control: the exact-replay tail and the FLA tail must launch differently, or this \
         test cannot tell them apart"
    );
    assert_eq!(
        captured.iter().map(shape).collect::<Vec<_>>(),
        reference.iter().map(shape).collect::<Vec<_>>(),
        "a replay-tail capture must run [0, cap), then the exact replay of [cap, k)"
    );
}

#[test]
fn a_tail_capture_without_replay_tail_keeps_the_pass_arm_after_the_split() {
    let (captured, reference, _) = captured_and_reference(false);
    assert_eq!(
        captured.iter().map(shape).collect::<Vec<_>>(),
        reference.iter().map(shape).collect::<Vec<_>>(),
        "without replay_tail the chunk must run the two recurrences a pass ending at the \
         capture point and a pass starting there would"
    );
}

#[test]
fn without_a_capture_inside_the_chunk_the_recurrence_runs_once() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = native_fp8_gdn_layer(&gpu, &config, true, true);
    // 2026-10-01: One arena for every call, so each run sees the same FLA scratch.
    let buffers = BufferArena::new(&config, 512, 4096, 16, 32, &gpu).unwrap();
    let k = 120usize;
    let nv = config.linear_num_value_heads;
    let vd = config.linear_value_head_dim;
    let conv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2 + nv * vd;
    let b = Bufs {
        h: gpu.alloc(layer.h_state_bytes).unwrap(),
        qkv: gpu.alloc(k * conv_dim * 2).unwrap(),
        gates: gpu.alloc(k * nv * 2 * 4).unwrap(),
        out: gpu.alloc(k * nv * vd * 2).unwrap(),
    };
    let h_dst = gpu.alloc(layer.h_state_bytes).unwrap();
    let whole = run(
        &gpu, &config, &buffers, &layer, &b, 0, k as u32, None, true, false,
    );
    // 2026-10-01: A capture point at the chunk's end (or its start) is not inside it, so
    // nothing splits and nothing is copied.
    let at_end = run(
        &gpu,
        &config,
        &buffers,
        &layer,
        &b,
        0,
        k as u32,
        Some((k, h_dst)),
        true,
        false,
    );
    let at_start = run(
        &gpu,
        &config,
        &buffers,
        &layer,
        &b,
        0,
        k as u32,
        Some((0, h_dst)),
        true,
        false,
    );
    let shapes = |v: &[MockLaunch]| v.iter().map(shape).collect::<Vec<_>>();
    assert_eq!(shapes(&at_end), shapes(&whole));
    assert_eq!(shapes(&at_start), shapes(&whole));
    assert!(
        gpu.read_alloc(h_dst).unwrap().iter().all(|&b| b == 0),
        "no capture inside the chunk must copy nothing"
    );
}
