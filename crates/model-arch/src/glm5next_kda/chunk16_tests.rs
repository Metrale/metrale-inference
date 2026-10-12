// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The launch plan of the chunked KDA scan on the mock backend: which rows take it,
//! the conv before it, and the prepare and scan grids and buffers. The numerics are measured on
//! a GPU by `examples/glm5next_kda_chunk16_bench.rs` against the per-token walk.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::decode_rows_tests::{
    CHUNK16_PREP, CHUNK16_SCAN, CONV_TOKENS, CONV_WINDOW, RECUR_SEQ, cfg, layer, seq_state, since,
};
use super::super::{Glm5NextKdaWorkspace, KdaSeqState};
use super::parse_min_rows;

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

/// 2026-10-10: `decode_k_modes` over `k` rows with floor `min`; the stateful launches it made.
fn stateful(
    k: usize,
    min: Option<usize>,
    snapshots: bool,
) -> (Vec<MockLaunch>, Glm5NextKdaWorkspace, KdaSeqState) {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 64).unwrap();
    let st = seq_state(&gpu);
    let hidden = gpu.alloc(k * 256 * 2).unwrap();
    let snaps: Vec<(DevicePtr, DevicePtr)> = if snapshots {
        (0..k - 1)
            .map(|_| (gpu.alloc(1 << 17).unwrap(), gpu.alloc(1 << 15).unwrap()))
            .collect()
    } else {
        Vec::new()
    };
    let from = gpu.launch_count();
    l.decode_k_modes(&gpu, hidden, k, &st, &ws, &snaps, 7, true, min, None)
        .unwrap();
    let kinds = [
        CONV_TOKENS,
        CONV_WINDOW,
        RECUR_SEQ,
        CHUNK16_PREP,
        CHUNK16_SCAN,
    ];
    let got = since(&gpu, from)
        .into_iter()
        .filter(|x| kinds.contains(&x.func))
        .collect();
    (got, ws, st)
}

/// 2026-10-10: More chunks than the workspace's padded rows hold are refused before launching.
#[test]
fn rows_past_the_workspace_are_refused() {
    let gpu = MockGpuBackend::new();
    let lay = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 16).unwrap();
    let st = seq_state(&gpu);
    let from = gpu.launch_count();
    assert!(lay.stateful_chunk16(&gpu, 17, &st, &ws, 7).is_err());
    assert_eq!(gpu.launch_count(), from);
}

/// 2026-10-10: 16 rows at a floor of 16 take the chunked scan through `decode_k_modes` (wider
/// sub-chunks run their projections on cuBLASLt, which the mock cannot): the per-token conv,
/// prepare, scan, and no per-token recurrence.
#[test]
fn a_subchunk_at_the_floor_takes_the_scan() {
    let (l, _ws, _st) = stateful(16, Some(16), false);
    let funcs: Vec<u64> = l.iter().map(|x| x.func).collect();
    assert_eq!(
        funcs,
        vec![CONV_TOKENS, CONV_WINDOW, CHUNK16_PREP, CHUNK16_SCAN]
    );
}

/// 2026-10-10: 40 rows: the per-token conv, then prepare over 3 chunks of 16 for both heads and
/// the scan over both heads' four 32-column blocks, reading the conv's q/k/v (k one `qkv` and v
/// two past q), gate and beta rows and writing `core`, on the layer's recurrent state.
#[test]
fn forty_rows_run_conv_prepare_and_scan_over_three_chunks() {
    let gpu = MockGpuBackend::new();
    let lay = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 64).unwrap();
    let st = seq_state(&gpu);
    let from = gpu.launch_count();
    lay.stateful_chunk16(&gpu, 40, &st, &ws, 7).unwrap();
    let l = since(&gpu, from);
    let funcs: Vec<u64> = l.iter().map(|x| x.func).collect();
    assert_eq!(
        funcs,
        vec![CONV_TOKENS, CONV_WINDOW, CHUNK16_PREP, CHUNK16_SCAN]
    );
    let c = cfg();
    let (qkv, cd) = (c.qkv_dim(), c.conv_dim());
    let (p, s) = (&l[2], &l[3]);
    assert_eq!((p.grid, p.block), ([3, 2, 1], [128, 1, 1]));
    assert_eq!(ptr(p, 0), ws.conv_out);
    assert_eq!(ptr(p, 1), ws.conv_out.offset(qkv * 2));
    assert_eq!(ptr(p, 2), ws.conv_out.offset(qkv * 4));
    assert_eq!((ptr(p, 3), ptr(p, 4)), (ws.gate, ws.beta));
    let scratch = [
        ws.chunk_u,
        ws.chunk_w,
        ws.q_f32,
        ws.k_f32,
        ws.v_f32,
        ws.chunk_gc,
    ];
    for (i, b) in scratch.iter().enumerate() {
        assert_eq!(ptr(p, 5 + i), *b, "prepare scratch {i}");
        assert_eq!(ptr(s, i), *b, "scan scratch {i}");
    }
    assert_eq!(
        (u32a(p, 11), u32a(p, 12), u32a(p, 13), u32a(p, 14)),
        (2, 40, cd as u32, qkv as u32)
    );
    assert_eq!((s.grid, s.block), ([2, 4, 1], [256, 1, 1]));
    assert_eq!((ptr(s, 6), ptr(s, 7)), (st.recurrent, ws.core));
    assert_eq!(
        (u32a(s, 8), u32a(s, 9), u32a(s, 10), u32a(s, 11)),
        (2, 40, 3, qkv as u32)
    );
}

/// 2026-10-10: Below the floor, without the lever, or with per-row snapshots (a verify), the
/// rows keep the per-token recurrence.
#[test]
fn narrow_unset_or_snapshotted_rows_keep_the_token_walk() {
    for (k, min, snaps) in [
        (15usize, Some(16), false),
        (16, None, false),
        (16, Some(16), true),
    ] {
        let (l, _ws, _st) = stateful(k, min, snaps);
        assert!(
            l.iter()
                .all(|x| x.func != CHUNK16_PREP && x.func != CHUNK16_SCAN),
            "k={k} min={min:?} snapshots={snaps}"
        );
    }
    let (l, _ws, _st) = stateful(15, Some(16), false);
    assert!(l.iter().any(|x| x.func == RECUR_SEQ));
}

/// 2026-10-10: The lever: unset and `0` off, a floor of at least 16, anything else an error.
#[test]
fn the_lever_parses_strictly() {
    assert_eq!(parse_min_rows(None), Ok(None));
    assert_eq!(parse_min_rows(Some("0")), Ok(None));
    assert_eq!(parse_min_rows(Some("16")), Ok(Some(16)));
    assert_eq!(parse_min_rows(Some("512")), Ok(Some(512)));
    for bad in ["15", "1", "on", "-4"] {
        assert!(parse_min_rows(Some(bad)).is_err(), "{bad}");
    }
}
