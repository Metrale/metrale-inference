// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: The launch plan of the 64-token chunked KDA scan on the mock backend: which rows
//! take it (ahead of the 16-token scan), the conv before it, and the prepare and scan grids,
//! shared memory and buffers. The numerics are measured on a GPU by
//! `examples/glm5next_kda_chunk64_bench.rs` against the per-token walk.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::Glm5NextKdaWorkspace;
use super::super::decode_rows_tests::{
    CHUNK16_PREP, CHUNK16_SCAN, CHUNK64_PREP, CHUNK64_SCAN, CONV_TOKENS, CONV_WINDOW, RECUR_SEQ,
    cfg, layer, seq_state, since,
};
use super::{CHUNK64, parse_min_rows64};

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

const KINDS: [u64; 7] = [
    CONV_TOKENS,
    CONV_WINDOW,
    RECUR_SEQ,
    CHUNK16_PREP,
    CHUNK16_SCAN,
    CHUNK64_PREP,
    CHUNK64_SCAN,
];

/// 2026-10-12: `decode_k_modes` over `k` rows with the two floors; the stateful launches.
fn stateful(k: usize, min16: Option<usize>, min64: Option<usize>, snapshots: bool) -> Vec<u64> {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 256).unwrap();
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
    l.decode_k_modes(&gpu, hidden, k, &st, &ws, &snaps, 7, true, min16, min64)
        .unwrap();
    since(&gpu, from)
        .into_iter()
        .filter(|x| KINDS.contains(&x.func))
        .map(|x| x.func)
        .collect()
}

/// 2026-10-12: At or past its floor the 64-token scan runs, ahead of the 16-token scan when both
/// are armed; below it, the 16-token scan (if armed) or the token walk; a verify (per-row
/// snapshots) never takes it.
#[test]
fn the_64_token_scan_takes_wide_unsnapshotted_rows_first() {
    let c64 = vec![CONV_TOKENS, CONV_WINDOW, CHUNK64_PREP, CHUNK64_SCAN];
    let c16 = vec![CONV_TOKENS, CONV_WINDOW, CHUNK16_PREP, CHUNK16_SCAN];
    assert_eq!(stateful(64, None, Some(64), false), c64);
    assert_eq!(stateful(64, Some(16), Some(64), false), c64);
    assert_eq!(stateful(63, Some(16), Some(64), false), c16);
    assert!(stateful(63, None, Some(64), false).contains(&RECUR_SEQ));
    let v = stateful(64, None, Some(64), true);
    assert!(!v.contains(&CHUNK64_PREP) && !v.contains(&CHUNK64_SCAN));
}

/// 2026-10-12: 130 rows: the conv, then prepare over 3 chunks of 64 for both heads (256 threads,
/// its shared memory) and the scan over both heads' four 32-column blocks (128 threads), reading
/// the conv's q/k/v (k one `qkv` and v two past q), gate and beta rows, writing `core`, on the
/// layer's recurrent state.
#[test]
fn rows_run_conv_prepare_and_scan_over_their_chunks() {
    let gpu = MockGpuBackend::new();
    let lay = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 130).unwrap();
    assert_eq!(
        ws.t_pad() % CHUNK64,
        0,
        "the workspace pads to whole 64-token chunks"
    );
    let st = seq_state(&gpu);
    let from = gpu.launch_count();
    lay.stateful_chunk64(&gpu, 130, &st, &ws, 7).unwrap();
    let l = since(&gpu, from);
    let funcs: Vec<u64> = l.iter().map(|x| x.func).collect();
    assert_eq!(
        funcs,
        vec![CONV_TOKENS, CONV_WINDOW, CHUNK64_PREP, CHUNK64_SCAN]
    );
    let c = cfg();
    let (qkv, cd) = (c.qkv_dim(), c.conv_dim());
    let (p, s) = (&l[2], &l[3]);
    assert_eq!((p.grid, p.block), ([3, 2, 1], [256, 1, 1]));
    assert_eq!(p.shared_mem, super::CHUNK64_PREP_SMEM);
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
        (2, 130, cd as u32, qkv as u32)
    );
    assert_eq!((s.grid, s.block), ([2, 4, 1], [128, 1, 1]));
    assert_eq!(s.shared_mem, super::CHUNK64_SCAN_SMEM);
    assert_eq!((ptr(s, 6), ptr(s, 7)), (st.recurrent, ws.core));
    assert_eq!(
        (u32a(s, 8), u32a(s, 9), u32a(s, 10), u32a(s, 11)),
        (2, 130, 3, qkv as u32)
    );
}

/// 2026-10-12: Rows past the workspace are refused before any launch.
#[test]
fn rows_past_the_workspace_are_refused() {
    let gpu = MockGpuBackend::new();
    let lay = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 64).unwrap();
    let st = seq_state(&gpu);
    let from = gpu.launch_count();
    assert!(lay.stateful_chunk64(&gpu, 65, &st, &ws, 7).is_err());
    assert_eq!(gpu.launch_count(), from);
}

/// 2026-10-12: The lever: unset and `0` off, a floor of at least 64, anything else an error.
#[test]
fn the_lever_parses_strictly() {
    assert_eq!(parse_min_rows64(None), Ok(None));
    assert_eq!(parse_min_rows64(Some("0")), Ok(None));
    assert_eq!(parse_min_rows64(Some("64")), Ok(Some(64)));
    assert_eq!(parse_min_rows64(Some("2048")), Ok(Some(2048)));
    for bad in ["63", "16", "on", "-4"] {
        assert!(parse_min_rows64(Some(bad)).is_err(), "{bad}");
    }
}
