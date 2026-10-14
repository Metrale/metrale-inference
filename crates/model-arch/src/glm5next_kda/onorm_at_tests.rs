// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The launch plan of `METRALE_GLM_KDA_ONORM_IN_SCAN` on the mock backend. The back
//! end drops exactly the o_norm launch. Each slab's scan becomes `kda_chunk64_scan_onorm` with
//! the six o_norm arguments after its own (plain and conv-fused prepare). Everything else
//! launches as before, and the refusals come before any launch. Also: the row-offset scan
//! (`scan_chunk64_rows`) shifts every row-indexed buffer and nothing else. The bytes are
//! compared on a GPU by `examples/glm5next_kda_onorm_in_scan_parity.rs`.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};
use metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M;

use super::super::decode_rows_tests::{
    CHUNK64_PREP, CHUNK64_PREP_GRAW, CHUNK64_SCAN, CHUNK64_SCAN_ONORM, cfg, layer, seq_state, since,
};
use super::super::{GateAt, Glm5NextKdaLayer, Glm5NextKdaWorkspace, Pack};
use super::{NormAt, onorm_scratch_elems, parse_onorm_in_scan};

const ONORM: u64 = 0x30E;

type Key = (u64, [u32; 3], [u32; 3], u32, Vec<MockArg>);

fn key(l: &MockLaunch) -> Key {
    (l.func, l.grid, l.block, l.shared_mem, l.args.clone())
}

fn buf(p: DevicePtr) -> MockArg {
    MockArg::Buffer(p)
}

/// 2026-10-14: The six o_norm arguments of `kda_chunk64_scan_onorm`, in its order.
fn onorm_args(lay: &Glm5NextKdaLayer, ws: &Glm5NextKdaWorkspace) -> Vec<MockArg> {
    vec![
        buf(ws.out_gate),
        buf(lay.weights.o_norm.weight),
        buf(ws.o_norm_out),
        buf(ws.onorm_part),
        buf(ws.onorm_count),
        MockArg::Bytes(lay.cfg.rms_norm_eps.to_le_bytes().to_vec()),
    ]
}

/// 2026-10-14: `back_end` at `Scan` launches exactly what it launches at `Back` less the one
/// o_norm, which reads `core`, `out_gate`, the weight and writes `o_norm_out`; o_proj still runs
/// on `o_norm_out`. At most `DENSE_GEMV_BATCHM_MAX_M` rows: past it o_proj is a cuBLASLt call,
/// which the mock backend does not record.
#[test]
fn back_end_drops_only_the_o_norm_launch() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 300).unwrap();
    let t = DENSE_GEMV_BATCHM_MAX_M as usize;
    let run = |norm| {
        let from = gpu.launch_count();
        l.back_end(&gpu, t, &ws, norm, 7).unwrap();
        since(&gpu, from)
    };
    let (back, scan) = (run(NormAt::Back), run(NormAt::Scan));
    let norms: Vec<&MockLaunch> = back.iter().filter(|x| x.func == ONORM).collect();
    assert_eq!(norms.len(), 1);
    assert_eq!(norms[0].grid, [(t * cfg().heads) as u32, 1, 1]);
    assert_eq!(
        norms[0].args[..4],
        [
            buf(ws.core),
            buf(ws.out_gate),
            buf(l.weights.o_norm.weight),
            buf(ws.o_norm_out)
        ]
    );
    assert!(!scan.is_empty(), "o_proj runs at Scan");
    let less_norm: Vec<Key> = back.iter().filter(|x| x.func != ONORM).map(key).collect();
    assert_eq!(scan.iter().map(key).collect::<Vec<_>>(), less_norm);
}

/// 2026-10-14: 700 rows (slabs of 4, 4, 3), both prepares and both gate placements: each
/// `kda_chunk64_scan` becomes `kda_chunk64_scan_onorm` with the six o_norm arguments appended;
/// every other launch unchanged.
#[test]
fn the_scan_takes_the_o_norm_arguments() {
    let gpu = MockGpuBackend::new();
    let lay = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 700).unwrap();
    let st = seq_state(&gpu);
    for fused in [false, true] {
        for gate in [GateAt::Kernel, GateAt::Prepare] {
            let run = |norm| {
                let from = gpu.launch_count();
                lay.stateful_chunk64_with(
                    &gpu,
                    700,
                    &st,
                    &ws,
                    Pack::InFrontEnd,
                    fused,
                    gate,
                    norm,
                    7,
                )
                .unwrap();
                since(&gpu, from)
            };
            let (back, scan) = (run(NormAt::Back), run(NormAt::Scan));
            let what = format!("fused={fused} {gate:?}");
            assert_eq!(
                back.iter().filter(|x| x.func == CHUNK64_SCAN).count(),
                3,
                "{what}"
            );
            let want: Vec<Key> = back
                .iter()
                .map(|l| {
                    let mut k = key(l);
                    if l.func == CHUNK64_SCAN {
                        k.0 = CHUNK64_SCAN_ONORM;
                        k.4.extend(onorm_args(&lay, &ws));
                    }
                    k
                })
                .collect();
            assert_eq!(scan.iter().map(key).collect::<Vec<_>>(), want, "{what}");
        }
    }
}

/// 2026-10-14: The scratch the kernel indexes: `[slab][heads][64][head_dim / 32]` sums and one
/// counter per (slab chunk, head), both allocated at that size.
#[test]
fn the_scratch_holds_a_slab_of_sums_and_counters() {
    let c = cfg();
    let (part, count) = onorm_scratch_elems(&c);
    assert_eq!(part, 4 * c.heads * 64 * (c.head_dim / 32));
    assert_eq!(count, 4 * c.heads);
}

/// 2026-10-14: `scan_chunk64_rows` from row `r` launches what it launches from row 0 with every
/// row-indexed buffer (conv rows, gate rows, beta, core, out_gate, o_norm_out) `r` rows on, and
/// nothing else changed: the scratch, the state and the scalars are the same.
#[test]
fn the_row_offset_shifts_only_the_row_buffers() {
    let gpu = MockGpuBackend::new();
    let lay = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 700).unwrap();
    let st = seq_state(&gpu);
    let c = cfg();
    let (qkv, cd, h) = (c.qkv_dim(), c.conv_dim(), c.heads);
    let r = 192;
    for gate in [GateAt::Kernel, GateAt::Prepare] {
        for norm in [NormAt::Back, NormAt::Scan] {
            let run = |row0| {
                let from = gpu.launch_count();
                lay.scan_chunk64_rows(&gpu, row0, 300, &st, &ws, gate, norm, 7)
                    .unwrap();
                since(&gpu, from)
            };
            let (at0, at_r) = (run(0), run(r));
            let shifted = [
                (ws.conv_out, r * cd * 2),
                (ws.conv_out.offset(qkv * 2), r * cd * 2),
                (ws.conv_out.offset(qkv * 4), r * cd * 2),
                (ws.gate, r * qkv * 4),
                (ws.g_raw, r * qkv * 2),
                (ws.beta, r * h * 4),
                (ws.core, r * qkv * 4),
                (ws.out_gate, r * qkv * 2),
                (ws.o_norm_out, r * qkv * 2),
            ];
            let want: Vec<Key> = at0
                .iter()
                .map(|l| {
                    let mut k = key(l);
                    for a in k.4.iter_mut() {
                        if let Some(&(p, by)) = shifted.iter().find(|(p, _)| *a == buf(*p)) {
                            *a = buf(p.offset(by));
                        }
                    }
                    k
                })
                .collect();
            let what = format!("{gate:?} {norm:?}");
            assert_eq!(at0.len(), 4, "{what}: two slabs of prepare + scan");
            assert_eq!(at_r.iter().map(key).collect::<Vec<_>>(), want, "{what}");
        }
    }
}

/// 2026-10-14: `scan_chunk64_rows` refuses rows past the workspace before any launch.
#[test]
fn the_row_offset_refuses_rows_past_the_workspace() {
    let gpu = MockGpuBackend::new();
    let lay = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 700).unwrap();
    let st = seq_state(&gpu);
    let from = gpu.launch_count();
    for (row0, k) in [(0, 701), (600, 101), (700, 1), (5, 0)] {
        assert!(
            lay.scan_chunk64_rows(&gpu, row0, k, &st, &ws, GateAt::Kernel, NormAt::Back, 7)
                .is_err(),
            "{row0} + {k}"
        );
    }
    assert_eq!(gpu.launch_count(), from);
    lay.scan_chunk64_rows(&gpu, 600, 100, &st, &ws, GateAt::Kernel, NormAt::Back, 7)
        .unwrap();
}

/// 2026-10-14: Without `kda_chunk64_scan_onorm`, `Scan` launches nothing and errors, on both
/// prepares; `Back` on the same layer still runs.
#[test]
fn scan_refuses_without_the_kernel_before_any_launch() {
    let gpu = MockGpuBackend::new();
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 300).unwrap();
    let st = seq_state(&gpu);
    let mut lay = layer(&gpu);
    lay.kernels.seq.chunk64_scan_onorm = KernelHandle(0);
    for fused in [false, true] {
        let from = gpu.launch_count();
        let r = lay.stateful_chunk64_with(
            &gpu,
            130,
            &st,
            &ws,
            Pack::InFrontEnd,
            fused,
            GateAt::Kernel,
            NormAt::Scan,
            7,
        );
        assert!(r.is_err(), "fused={fused}");
        assert_eq!(gpu.launch_count(), from, "fused={fused}");
    }
    assert!(lay.onorm_in_scan_check().is_err());
    lay.stateful_chunk64_with(
        &gpu,
        130,
        &st,
        &ws,
        Pack::InFrontEnd,
        false,
        GateAt::Kernel,
        NormAt::Back,
        7,
    )
    .unwrap();
}

/// 2026-10-14: Off the 64-token scan the o_norm stays in the back end whatever the lever says.
#[test]
fn only_the_64_token_scan_moves_the_o_norm() {
    let gpu = MockGpuBackend::new();
    assert_eq!(layer(&gpu).norm_at_for(false).unwrap(), NormAt::Back);
}

/// 2026-10-14: The lever: unset and `0` off, `1` on, anything else an error.
#[test]
fn the_lever_parses_strictly() {
    assert_eq!(parse_onorm_in_scan(None), Ok(false));
    assert_eq!(parse_onorm_in_scan(Some("0")), Ok(false));
    assert_eq!(parse_onorm_in_scan(Some("1")), Ok(true));
    for bad in ["", "on", "true", "2", " 1"] {
        assert!(parse_onorm_in_scan(Some(bad)).is_err(), "{bad:?}");
    }
}

/// 2026-10-14: The prepare at row 0 is the plain prepare (no offset slips into the row-0 path):
/// the first slab's prepare reads `conv_out`, `gate`/`g_raw` and `beta` from their bases.
#[test]
fn the_row_zero_prepare_reads_the_buffer_bases() {
    let gpu = MockGpuBackend::new();
    let lay = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 300).unwrap();
    let st = seq_state(&gpu);
    for (gate, func, g) in [
        (GateAt::Kernel, CHUNK64_PREP, ws.gate),
        (GateAt::Prepare, CHUNK64_PREP_GRAW, ws.g_raw),
    ] {
        let from = gpu.launch_count();
        lay.scan_chunk64_rows(&gpu, 0, 130, &st, &ws, gate, NormAt::Back, 7)
            .unwrap();
        let l = since(&gpu, from);
        assert_eq!(l[0].func, func);
        assert_eq!(l[0].args[0], buf(ws.conv_out));
        assert_eq!(l[0].args[3], buf(g));
    }
}
