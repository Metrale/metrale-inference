// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The launch plan of the 64-token scan with the output norm in its epilogue
//! (`METRALE_GLM_KDA_SCAN_ONORM`) on the mock backend: each slab's scan becomes
//! `kda_chunk64_scan_onorm` with `core` replaced by the norm's operands in the kernel's order and
//! everything else unchanged, under every conv and gate placement; `back_end` drops exactly the
//! norm launch when the scan normed; and the refusals come before any launch. The bytes are
//! compared on a GPU by `examples/glm5next_kda_scan_onorm_gate.rs`.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::super::super::Glm5NextKdaWorkspace;
use super::super::super::Pack;
use super::super::super::decode_rows_tests::{
    CHUNK64_SCAN, CHUNK64_SCAN_ONORM, cfg, layer, seq_state, since,
};
use super::super::super::gate_at::GateAt;
use super::super::CHUNK64_SCAN_SMEM;
use super::{CHUNK64_SCAN_ONORM_SMEM, parse_scan_onorm};

/// 2026-10-15: `kda_o_norm_gated_bf16`'s handle in `decode_rows_tests::layer`.
const O_NORM: u64 = 0x30E;

/// 2026-10-15: The launches of one 700-row forward (11 chunks: slabs of 4, 4, 3), its workspace
/// and its norm weight.
fn run(
    fused: bool,
    gate: GateAt,
    onorm: bool,
) -> (Vec<MockLaunch>, Glm5NextKdaWorkspace, DevicePtr) {
    let gpu = MockGpuBackend::new();
    let lay = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 700).unwrap();
    let st = seq_state(&gpu);
    lay.stateful_chunk64_with(&gpu, 700, &st, &ws, Pack::InFrontEnd, fused, gate, onorm, 7)
        .unwrap();
    (since(&gpu, 0), ws, lay.weights.o_norm.weight)
}

/// 2026-10-15: Under every conv and gate placement the fused forward is the plain one launch
/// for launch, except that each of the three scans is `kda_chunk64_scan_onorm` with 1 KiB more
/// shared memory and, where the plain scan takes `core`, the output gate, the norm weight,
/// `o_norm_out` and the RMS eps; the arguments before and after are the plain scan's.
#[test]
fn each_scan_takes_the_norm_operands_in_place_of_core_and_nothing_else_changes() {
    for fused in [false, true] {
        for gate in [GateAt::Kernel, GateAt::Prepare] {
            let ((plain, pws, _), (on, ows, lay_w)) =
                (run(fused, gate, false), run(fused, gate, true));
            let at = format!("fused={fused} gate={gate:?}");
            assert_eq!(plain.len(), on.len(), "{at}");
            let mut scans = 0;
            for (p, o) in plain.iter().zip(&on) {
                assert_eq!(
                    (p.grid, p.block, p.stream),
                    (o.grid, o.block, o.stream),
                    "{at}"
                );
                if p.func != CHUNK64_SCAN {
                    assert_eq!((p.func, p.shared_mem), (o.func, o.shared_mem), "{at}");
                    assert_eq!(p.args, o.args, "{at}");
                    continue;
                }
                scans += 1;
                assert_eq!(o.func, CHUNK64_SCAN_ONORM, "{at}");
                assert_eq!(p.shared_mem, CHUNK64_SCAN_SMEM);
                assert_eq!(o.shared_mem, CHUNK64_SCAN_SMEM + 1024);
                assert_eq!(o.shared_mem, CHUNK64_SCAN_ONORM_SMEM);
                assert_eq!((p.args.len(), o.args.len()), (13, 16), "{at}");
                assert_eq!(p.args[..7], o.args[..7], "{at}");
                assert_eq!(p.args[8..], o.args[11..], "{at}");
                assert_eq!(p.args[7], MockArg::Buffer(pws.core), "{at}");
                assert_eq!(
                    o.args[7..11],
                    [
                        MockArg::Buffer(ows.out_gate),
                        MockArg::Buffer(lay_w),
                        MockArg::Buffer(ows.o_norm_out),
                        MockArg::Bytes(cfg().rms_norm_eps.to_le_bytes().to_vec()),
                    ],
                    "{at}"
                );
            }
            assert_eq!(scans, 3, "{at}");
        }
    }
}

/// 2026-10-15: `back_end` after a normed scan is the unnormed one without the norm launch:
/// `o_proj` still reads `o_norm_out`. At 16 rows `o_proj` is the batched GEMV, a launch the mock
/// records; above 16 it is cuBLASLt, which calls the driver directly and must not run here.
#[test]
fn back_end_drops_exactly_the_norm_launch_when_the_scan_normed() {
    const ROWS: usize = 16;
    let gpu = MockGpuBackend::new();
    let lay = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), ROWS).unwrap();
    lay.back_end(&gpu, ROWS, &ws, false, 7).unwrap();
    let from = gpu.launch_count();
    lay.back_end(&gpu, ROWS, &ws, true, 7).unwrap();
    let (plain, normed) = (since(&gpu, 0)[..from].to_vec(), since(&gpu, from));
    assert_eq!(plain[0].func, O_NORM);
    assert!(normed.iter().all(|l| l.func != O_NORM));
    assert_eq!(plain.len(), normed.len() + 1);
    for (p, n) in plain[1..].iter().zip(&normed) {
        assert_eq!((p.func, p.grid, &p.args), (n.func, n.grid, &n.args));
    }
}

/// 2026-10-15: No fused kernel, a device that places no cluster of it, or a head_dim other than
/// 128: an error before any launch. The mock's answer back at 24, the same forward runs.
#[test]
fn the_refusals_come_before_any_launch() {
    let gpu = MockGpuBackend::new();
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 130).unwrap();
    let st = seq_state(&gpu);
    let fwd = |lay: &super::Glm5NextKdaLayer| {
        lay.stateful_chunk64_with(
            &gpu,
            130,
            &st,
            &ws,
            Pack::InFrontEnd,
            false,
            GateAt::Kernel,
            true,
            7,
        )
    };
    let mut absent = layer(&gpu);
    absent.kernels.seq.chunk64_scan_onorm = KernelHandle(0);
    let from = gpu.launch_count();
    let e = fwd(&absent).unwrap_err().to_string();
    assert!(e.contains("needs kda_chunk64_scan_onorm"), "{e}");
    assert_eq!(gpu.launch_count(), from);

    let lay = layer(&gpu);
    gpu.set_max_active_clusters(0);
    let e = fwd(&lay).unwrap_err().to_string();
    assert!(e.contains("places no cluster"), "{e}");
    assert_eq!(gpu.launch_count(), from);
    gpu.set_max_active_clusters(24);
    fwd(&lay).unwrap();
    assert!(gpu.launch_count() > from);

    let mut narrow = layer(&gpu);
    narrow.cfg.head_dim = 64;
    let e = narrow.scan_onorm_check(&gpu).unwrap_err().to_string();
    assert!(e.contains("head_dim 64"), "{e}");
}

/// 2026-10-15: The lever: unset and `0` off, `1` on, anything else an error naming it.
#[test]
fn the_lever_parses_strictly() {
    assert_eq!(parse_scan_onorm(None), Ok(false));
    assert_eq!(parse_scan_onorm(Some("0")), Ok(false));
    assert_eq!(parse_scan_onorm(Some("1")), Ok(true));
    for bad in ["", "on", "true", "2", " 1"] {
        let e = parse_scan_onorm(Some(bad)).unwrap_err();
        assert!(e.contains("METRALE_GLM_KDA_SCAN_ONORM"), "{bad:?}: {e}");
    }
}
