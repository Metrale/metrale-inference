// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: The launch plan of the 64-token chunked KDA scan on the mock backend: which rows
//! take it (`scan_mode`, ahead of the 16-token scan), the conv before it, and the prepare and scan grids,
//! shared memory and buffers. The numerics are measured on a GPU by
//! `examples/glm5next_kda_chunk64_bench.rs` against the per-token walk.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};

use super::super::Glm5NextKdaWorkspace;
use super::super::Pack;
use super::super::decode::{ScanMode, scan_mode};
use super::super::decode_rows_tests::{
    CHUNK64_PREP, CHUNK64_SCAN, CONV_TOKENS, CONV_WINDOW, cfg, layer, seq_state, since,
};
use super::super::gate_at::GateAt;
use super::{CHUNK64, CHUNK64_SLAB};

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

/// 2026-10-12: The scan choice: at or past its floor the 64-token scan, ahead of the 16-token
/// scan when both are armed; below it, the 16-token scan (if armed) or neither; a verify (per-row
/// snapshots) never takes a scan. (`decode_k_modes` itself cannot run 64 rows on the mock: the
/// front end's projections past 16 rows go to cuBLASLt.)
#[test]
fn the_64_token_scan_takes_wide_unsnapshotted_rows_first() {
    assert_eq!(scan_mode(64, true, None, Some(64)), ScanMode::Chunk64);
    assert_eq!(scan_mode(64, true, Some(16), Some(64)), ScanMode::Chunk64);
    assert_eq!(
        scan_mode(4096, true, Some(16), Some(2048)),
        ScanMode::Chunk64
    );
    assert_eq!(scan_mode(63, true, Some(16), Some(64)), ScanMode::Chunk16);
    assert_eq!(
        scan_mode(2047, true, Some(16), Some(2048)),
        ScanMode::Chunk16
    );
    assert_eq!(scan_mode(63, true, None, Some(64)), ScanMode::None);
    assert_eq!(scan_mode(64, true, None, None), ScanMode::None);
    assert_eq!(scan_mode(64, false, Some(16), Some(64)), ScanMode::None);
}

/// 2026-10-12: `chunk64_applies` on a layer with the kernels bound: the floor must be set and
/// reached; unset is never taken.
#[test]
fn the_floor_gates_the_layer_predicate() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    assert!(l.chunk64_ready());
    assert!(l.chunk64_applies(64, Some(64)));
    assert!(l.chunk64_applies(4096, Some(2048)));
    assert!(!l.chunk64_applies(63, Some(64)));
    assert!(!l.chunk64_applies(4096, None));
}

/// 2026-10-12: 130 rows: the conv, then one slab: prepare over 3 chunks of 64 for both heads
/// (256 threads, its shared memory) and the scan over both heads' two 64-column blocks (256
/// threads), reading the conv's q/k/v (k one `qkv` and v two past q), gate and beta rows, the
/// factors in the six scratch buffers, writing `core`, on the layer's recurrent state.
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
    lay.stateful_chunk64(&gpu, 130, &st, &ws, Pack::InFrontEnd, GateAt::Kernel, 7)
        .unwrap();
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
        (
            u32a(p, 11),
            u32a(p, 12),
            u32a(p, 13),
            u32a(p, 14),
            u32a(p, 15)
        ),
        (2, 130, 0, cd as u32, qkv as u32)
    );
    assert_eq!((s.grid, s.block), ([2, 2, 1], [256, 1, 1]));
    assert_eq!(s.shared_mem, super::CHUNK64_SCAN_SMEM);
    assert_eq!((ptr(s, 6), ptr(s, 7)), (st.recurrent, ws.core));
    assert_eq!(
        (
            u32a(s, 8),
            u32a(s, 9),
            u32a(s, 10),
            u32a(s, 11),
            u32a(s, 12)
        ),
        (2, 130, 0, 3, qkv as u32)
    );
}

/// 2026-10-12: 11 chunks (700 rows) run as two full slabs of 4 and a slab of 3, each a prepare
/// over its chunks and a scan from its first chunk; the conv runs once.
#[test]
fn a_long_sub_chunk_runs_in_slabs() {
    assert_eq!(CHUNK64_SLAB, 4, "the slab cases below assume 4");
    let gpu = MockGpuBackend::new();
    let lay = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 700).unwrap();
    let st = seq_state(&gpu);
    let from = gpu.launch_count();
    lay.stateful_chunk64(&gpu, 700, &st, &ws, Pack::InFrontEnd, GateAt::Kernel, 7)
        .unwrap();
    let l = since(&gpu, from);
    let funcs: Vec<u64> = l.iter().map(|x| x.func).collect();
    assert_eq!(
        funcs,
        vec![
            CONV_TOKENS,
            CONV_WINDOW,
            CHUNK64_PREP,
            CHUNK64_SCAN,
            CHUNK64_PREP,
            CHUNK64_SCAN,
            CHUNK64_PREP,
            CHUNK64_SCAN
        ]
    );
    for (slab, (c0, n)) in [(0usize, 4u32), (4, 4), (8, 3)].into_iter().enumerate() {
        let (p, s) = (&l[2 + 2 * slab], &l[3 + 2 * slab]);
        assert_eq!(
            (p.grid[0], u32a(p, 13)),
            (n, c0 as u32),
            "slab {slab} prepare"
        );
        assert_eq!(
            (u32a(s, 10), u32a(s, 11)),
            (c0 as u32, n),
            "slab {slab} scan"
        );
    }
}

/// 2026-10-12: Rows past the workspace are refused before any launch.
#[test]
fn rows_past_the_workspace_are_refused() {
    let gpu = MockGpuBackend::new();
    let lay = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 64).unwrap();
    let st = seq_state(&gpu);
    let from = gpu.launch_count();
    assert!(
        lay.stateful_chunk64(&gpu, 65, &st, &ws, Pack::InFrontEnd, GateAt::Kernel, 7)
            .is_err()
    );
    assert_eq!(gpu.launch_count(), from);
}
