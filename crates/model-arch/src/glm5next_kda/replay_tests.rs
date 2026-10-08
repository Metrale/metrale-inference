// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Host-only tests of the KDA verify record and replay commit.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants: none beyond the types.
//!
//! `MockGpuBackend` moves real bytes for copies and records every launch with its arguments
//! but runs no kernel. So these tests prove the plumbing the bit-exactness argument rests on:
//! the commit restores the checkpoint, and each replayed row launches what the verify row
//! launched, on pointers whose bytes are the bytes that verify row read. The kernels' own
//! arithmetic is covered by the GPU test `glm_dflash_replay_matches_sequential_decode`.

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::weight_map::DenseWeight;

use super::*;

const K: usize = 4;

fn cfg() -> Glm5NextKdaConfig {
    Glm5NextKdaConfig {
        hidden: 256,
        heads: 2,
        head_dim: 128,
        conv_kernel: 4,
        gate_lower_bound: -5.0,
        rms_norm_eps: 1e-6,
        l2_eps: 1e-6,
        chunk: 16,
    }
}

struct Fixture {
    gpu: MockGpuBackend,
    layer: Glm5NextKdaLayer,
    ws: Glm5NextKdaWorkspace,
    state: KdaSeqState,
    ckpt: KdaSeqState,
}

/// 2026-10-08: Bytes with no short period, so two rows of any width differ.
fn fill(gpu: &MockGpuBackend, p: DevicePtr, bytes: usize, salt: u8) {
    let v: Vec<u8> = (0..bytes as u32)
        .map(|i| ((i ^ ((salt as u32) << 24)).wrapping_mul(0x9E37_79B1) >> 24) as u8)
        .collect();
    gpu.copy_h2d(&v, p).unwrap();
}

fn read(gpu: &MockGpuBackend, p: DevicePtr, bytes: usize) -> Vec<u8> {
    let mut v = vec![0u8; bytes];
    gpu.copy_d2h(p, &mut v).unwrap();
    v
}

fn fixture() -> Fixture {
    let gpu = MockGpuBackend::new();
    let c = cfg();
    let w = |gpu: &MockGpuBackend| DenseWeight {
        weight: gpu.alloc(64).unwrap(),
    };
    let weights = Glm5NextKdaWeights {
        q_proj: w(&gpu),
        k_proj: w(&gpu),
        v_proj: w(&gpu),
        conv: w(&gpu),
        f_a: w(&gpu),
        f_b: w(&gpu),
        dt_bias: gpu.alloc(64).unwrap(),
        a_log: gpu.alloc(64).unwrap(),
        b_proj: w(&gpu),
        g_a: w(&gpu),
        g_b: w(&gpu),
        o_norm: w(&gpu),
        o_proj: w(&gpu),
    };
    let kernels = Glm5NextKdaKernels::resolve(&gpu).unwrap();
    let layer = Glm5NextKdaLayer::new(7, c, weights, kernels).unwrap();
    let ws = Glm5NextKdaWorkspace::new(&gpu, &c, K).unwrap();
    let st = |gpu: &MockGpuBackend| KdaSeqState {
        conv: gpu.alloc(c.conv_state_elems() * 4).unwrap(),
        recurrent: gpu.alloc(c.recurrent_state_elems() * 4).unwrap(),
    };
    let state = st(&gpu);
    let ckpt = st(&gpu);
    Fixture {
        gpu,
        layer,
        ws,
        state,
        ckpt,
    }
}

/// 2026-10-08: The launches of `stateful_row`: the ones that name the state's conv or
/// recurrent buffer.
fn row_launches(launches: &[MockLaunch], state: &KdaSeqState) -> Vec<MockLaunch> {
    launches
        .iter()
        .filter(|l| {
            l.args.iter().any(|a| {
                *a == MockArg::Buffer(state.conv) || *a == MockArg::Buffer(state.recurrent)
            })
        })
        .cloned()
        .collect()
}

/// 2026-10-08: Byte length of the row a workspace pointer addresses, by the array it lies in.
fn ws_row_len(f: &Fixture, p: DevicePtr) -> usize {
    let c = cfg();
    let within = |base: DevicePtr, bytes: usize| p.0 >= base.0 && p.0 < base.0 + bytes as u64;
    if within(f.ws.qkv_proj, K * c.conv_dim() * 2) {
        c.conv_dim() * 2
    } else if within(f.ws.gate, f.ws.t_pad() * c.qkv_dim() * 4) {
        c.qkv_dim() * 4
    } else if within(f.ws.beta, f.ws.t_pad() * c.heads * 4) {
        c.heads * 4
    } else {
        panic!("argument {p} differs between verify and replay but is not a recorded input")
    }
}

/// 2026-10-08: A partial accept restores the checkpoint and replays rows `0..n`; each replayed
/// row's conv and recurrent launches equal the verify row's except for the recorded inputs, and
/// those point at the bytes the verify row read, though the workspace has since been
/// overwritten.
#[test]
fn replay_repeats_the_verify_rows_on_their_recorded_inputs() {
    let f = fixture();
    let c = cfg();
    let s = f.gpu.default_stream();
    let rec_rows = K - 1;
    let rec_bytes = rec_rows * c.replay_row_bytes();
    let rec = KdaVerifyRecord::new(&c, f.gpu.alloc(rec_bytes).unwrap(), rec_bytes);
    assert_eq!(rec.rows(), rec_rows);

    let (h_bytes, conv_bytes) = (c.recurrent_state_elems() * 4, c.conv_state_elems() * 4);
    fill(&f.gpu, f.state.recurrent, h_bytes, 1);
    fill(&f.gpu, f.state.conv, conv_bytes, 2);
    let pre_h = read(&f.gpu, f.state.recurrent, h_bytes);
    let pre_conv = read(&f.gpu, f.state.conv, conv_bytes);
    // 2026-10-08: The inputs the verify rows read; the mock runs no kernel, so the front end
    // leaves them as written here.
    fill(&f.gpu, f.ws.qkv_proj, K * c.conv_dim() * 2, 3);
    fill(&f.gpu, f.ws.gate, K * c.qkv_dim() * 4, 4);
    fill(&f.gpu, f.ws.beta, K * c.heads * 4, 5);
    let hidden = f.gpu.alloc(K * c.hidden * 2).unwrap();

    f.layer
        .checkpoint_state(&f.gpu, &f.state, &f.ckpt, s)
        .unwrap();
    let before = f.gpu.launches_snapshot().len();
    f.layer
        .decode_k(&f.gpu, hidden, K, &f.state, &f.ws, &[], s)
        .unwrap();
    let verify = row_launches(&f.gpu.launches_snapshot()[before..], &f.state);
    assert_eq!(
        verify.len(),
        2 * K,
        "one conv and one recurrent launch per row"
    );
    let verify_bytes: Vec<Vec<Vec<u8>>> = verify
        .iter()
        .map(|l| {
            l.args
                .iter()
                .map(|a| match a {
                    MockArg::Buffer(p) if read_ok(&f, *p) => read(&f.gpu, *p, ws_row_len(&f, *p)),
                    _ => Vec::new(),
                })
                .collect()
        })
        .collect();
    f.layer
        .record_verify_rows(&f.gpu, &f.ws, rec_rows, &rec, s)
        .unwrap();

    // 2026-10-08: Later layers reuse the workspace, and the verify advanced the state.
    fill(&f.gpu, f.ws.qkv_proj, K * c.conv_dim() * 2, 90);
    fill(&f.gpu, f.ws.gate, K * c.qkv_dim() * 4, 91);
    fill(&f.gpu, f.ws.beta, K * c.heads * 4, 92);
    fill(&f.gpu, f.state.recurrent, h_bytes, 93);
    fill(&f.gpu, f.state.conv, conv_bytes, 94);

    let accepted = 2;
    let before = f.gpu.launches_snapshot().len();
    f.layer
        .commit_replay(&f.gpu, &f.state, &f.ckpt, &rec, accepted, K, &f.ws, s)
        .unwrap();
    let all_replay = f.gpu.launches_snapshot()[before..].to_vec();
    let replay = row_launches(&all_replay, &f.state);
    assert_eq!(
        all_replay.len(),
        replay.len(),
        "the commit launches only row steps"
    );
    assert_eq!(replay.len(), 2 * accepted);
    assert_eq!(read(&f.gpu, f.state.recurrent, h_bytes), pre_h);
    assert_eq!(read(&f.gpu, f.state.conv, conv_bytes), pre_conv);

    for (i, (v, r)) in verify.iter().zip(&replay).enumerate() {
        assert_eq!(
            (v.func, v.grid, v.block, v.shared_mem, v.stream),
            (r.func, r.grid, r.block, r.shared_mem, r.stream),
            "launch {i}"
        );
        assert_eq!(v.args.len(), r.args.len(), "launch {i}");
        let mut differing = 0;
        for (j, (va, ra)) in v.args.iter().zip(&r.args).enumerate() {
            if va == ra {
                continue;
            }
            differing += 1;
            let (MockArg::Buffer(_), MockArg::Buffer(rp)) = (va, ra) else {
                panic!("launch {i} arg {j}: a scalar differs: {va:?} vs {ra:?}");
            };
            let want = &verify_bytes[i][j];
            assert!(!want.is_empty(), "launch {i} arg {j}: not a recorded input");
            assert_eq!(
                &read(&f.gpu, *rp, want.len()),
                want,
                "launch {i} arg {j}: the replay reads other bytes than the verify row"
            );
        }
        // 2026-10-08: conv: the pre-conv input; recurrent: decay and beta.
        assert_eq!(differing, if i % 2 == 0 { 1 } else { 2 }, "launch {i}");
    }
}

fn read_ok(f: &Fixture, p: DevicePtr) -> bool {
    let c = cfg();
    let within = |base: DevicePtr, bytes: usize| p.0 >= base.0 && p.0 < base.0 + bytes as u64;
    within(f.ws.qkv_proj, K * c.conv_dim() * 2)
        || within(f.ws.gate, f.ws.t_pad() * c.qkv_dim() * 4)
        || within(f.ws.beta, f.ws.t_pad() * c.heads * 4)
}

/// 2026-10-08: A full accept keeps the verify's final state: no copy, no launch.
#[test]
fn a_full_accept_touches_nothing() {
    let f = fixture();
    let c = cfg();
    let rec_bytes = (K - 1) * c.replay_row_bytes();
    let rec = KdaVerifyRecord::new(&c, f.gpu.alloc(rec_bytes).unwrap(), rec_bytes);
    let (d2d, launches) = (f.gpu.d2d_count(), f.gpu.launch_count());
    f.layer
        .commit_replay(&f.gpu, &f.state, &f.ckpt, &rec, K, K, &f.ws, 0)
        .unwrap();
    assert_eq!((f.gpu.d2d_count(), f.gpu.launch_count()), (d2d, launches));
}

/// 2026-10-08: The commit refuses an impossible count and a record too short to replay from,
/// before it copies anything.
#[test]
fn the_commit_refuses_what_it_cannot_replay() {
    let f = fixture();
    let c = cfg();
    let short_bytes = c.replay_row_bytes();
    let short = KdaVerifyRecord::new(&c, f.gpu.alloc(short_bytes).unwrap(), short_bytes);
    let d2d = f.gpu.d2d_count();
    for (accepted, k) in [(0, K), (K + 1, K), (2, K)] {
        assert!(
            f.layer
                .commit_replay(&f.gpu, &f.state, &f.ckpt, &short, accepted, k, &f.ws, 0)
                .is_err(),
            "accepted {accepted} of {k} with a 1-row record"
        );
    }
    assert_eq!(f.gpu.d2d_count(), d2d);
    assert!(
        f.layer
            .record_verify_rows(&f.gpu, &f.ws, 2, &short, 0)
            .is_err()
    );
}

/// 2026-10-08: The pool sizes each slot's verify record with `ssm_replay_row_bytes_for`, before
/// any layer exists; the layer lays its rows out with `replay_row_bytes`. On the real GLM-5.3
/// config the two must agree, or the record holds fewer rows than the pool meant it to.
#[test]
fn the_pool_sizes_the_record_rows_the_layer_lays_out() {
    let config = metrale_config::parse_config(include_str!(
        "../../../model-engine/tests/fixtures/glm53-nvfp4-9e0d74e3-config.json"
    ))
    .unwrap();
    let kda = Glm5NextKdaConfig::from_model_config(&config);
    assert_eq!(
        metrale_model_layers::ssm_reserve::ssm_replay_row_bytes_for(&config),
        kda.replay_row_bytes()
    );
    // 2026-10-08: 64 heads of 128: a 24576-channel BF16 input, 8192 FP32 decays, 64 betas.
    assert_eq!(kda.replay_row_bytes(), 24_576 * 2 + 8_192 * 4 + 64 * 4);
}
