// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The launch plan of `METRALE_GLM_KDA_GATE_IN_PREPARE` on the mock backend: the
//! front end drops exactly the gate launch, each slab's prepare becomes its `_graw` twin with
//! `ws.gate` replaced by the four gate arguments in the kernel's order (plain and conv-fused
//! prepare), everything else launches as before, and the refusals come before any launch. The
//! bytes are compared on a GPU by `examples/glm5next_kda_gate_in_prepare_parity.rs`.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::super::decode_rows_tests::{
    CHUNK64_PREP, CHUNK64_PREP_CONV, CHUNK64_PREP_CONV_GRAW, CHUNK64_PREP_GRAW, GATE, cfg, layer,
    seq_state, since,
};
use super::super::{Glm5NextKdaLayer, Glm5NextKdaWorkspace, Pack};
use super::{GateAt, parse_gate_in_prepare};

type Key = (u64, [u32; 3], [u32; 3], u32, Vec<MockArg>);

fn key(l: &MockLaunch) -> Key {
    (l.func, l.grid, l.block, l.shared_mem, l.args.clone())
}

fn buf(p: DevicePtr) -> MockArg {
    MockArg::Buffer(p)
}

fn f32_arg(v: f32) -> MockArg {
    MockArg::Bytes(v.to_le_bytes().to_vec())
}

fn u32_arg(v: u32) -> MockArg {
    MockArg::Bytes(v.to_le_bytes().to_vec())
}

/// 2026-10-14: The four gate arguments of the `_graw` entries, in their order.
fn graw_args(lay: &Glm5NextKdaLayer, ws: &Glm5NextKdaWorkspace) -> Vec<MockArg> {
    vec![
        buf(ws.g_raw),
        buf(lay.weights.dt_bias),
        buf(lay.weights.a_log),
        f32_arg(lay.cfg.gate_lower_bound),
    ]
}

/// 2026-10-14: The gate-kernel run of `rows` rows turned into what the gate-in-prepare run must
/// launch: each `from` prepare becomes `to` with its `ws.gate` argument (at `at`) replaced by
/// the four gate arguments; every other launch unchanged.
fn expected(
    base: &[MockLaunch],
    (from, to, at): (u64, u64, usize),
    gate_args: &[MockArg],
    gate: DevicePtr,
) -> Vec<Key> {
    base.iter()
        .map(|l| {
            let mut k = key(l);
            if l.func == from {
                assert_eq!(
                    k.4[at],
                    buf(gate),
                    "the gate-kernel prepare reads ws.gate at {at}"
                );
                k.0 = to;
                k.4.splice(at..=at, gate_args.iter().cloned());
            }
            k
        })
        .collect()
}

/// 2026-10-14: `front_end` at `Prepare` launches exactly what it launches at `Kernel` less the
/// one `kda_gate_bf16`, which reads `g_raw`, `dt_bias`, `A_log` and writes `ws.gate`; f_b still
/// runs (its launch is in both).
#[test]
fn front_end_drops_only_the_gate_launch() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 16).unwrap();
    let hidden = gpu.alloc(16 * cfg().hidden * 2).unwrap();
    let run = |gate| {
        let from = gpu.launch_count();
        l.front_end(&gpu, hidden, 7, &ws, Pack::InFrontEnd, gate, 0)
            .unwrap();
        since(&gpu, from)
    };
    let (kernel, prepare) = (run(GateAt::Kernel), run(GateAt::Prepare));
    let gates: Vec<&MockLaunch> = kernel.iter().filter(|x| x.func == GATE).collect();
    assert_eq!(gates.len(), 1);
    let c = cfg();
    assert_eq!(
        gates[0].args,
        vec![
            buf(ws.g_raw),
            buf(l.weights.dt_bias),
            buf(l.weights.a_log),
            buf(ws.gate),
            u32_arg(7),
            u32_arg(c.heads as u32),
            u32_arg(c.head_dim as u32),
            f32_arg(c.gate_lower_bound),
        ]
    );
    let less_gate: Vec<Key> = kernel.iter().filter(|x| x.func != GATE).map(key).collect();
    assert_eq!(prepare.iter().map(key).collect::<Vec<_>>(), less_gate);
}

/// 2026-10-14: 700 rows (slabs of 4, 4, 3) on the plain prepare: the conv, then per slab
/// `kda_chunk64_prepare_graw` in place of `kda_chunk64_prepare` with `ws.gate` (argument 3)
/// replaced by `g_raw, dt_bias, A_log, lower_bound`; the scans unchanged.
#[test]
fn the_plain_prepare_takes_the_gate_arguments() {
    let gpu = MockGpuBackend::new();
    let lay = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 700).unwrap();
    let st = seq_state(&gpu);
    let run = |gate| {
        let from = gpu.launch_count();
        lay.stateful_chunk64_with(&gpu, 700, &st, &ws, Pack::InFrontEnd, false, gate, 7)
            .unwrap();
        since(&gpu, from)
    };
    let (kernel, prepare) = (run(GateAt::Kernel), run(GateAt::Prepare));
    assert_eq!(kernel.iter().filter(|x| x.func == CHUNK64_PREP).count(), 3);
    let want = expected(
        &kernel,
        (CHUNK64_PREP, CHUNK64_PREP_GRAW, 3),
        &graw_args(&lay, &ws),
        ws.gate,
    );
    assert_eq!(prepare.iter().map(key).collect::<Vec<_>>(), want);
}

/// 2026-10-14: The same with the conv in the prepare, for both input layouts:
/// `kda_chunk64_prepare_conv_graw` with `ws.gate` (argument 10, after the stage) replaced.
#[test]
fn the_conv_prepare_takes_the_gate_arguments() {
    let gpu = MockGpuBackend::new();
    let lay = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 700).unwrap();
    let st = seq_state(&gpu);
    for pack in [Pack::InFrontEnd, Pack::InConv] {
        let run = |gate| {
            let from = gpu.launch_count();
            lay.stateful_chunk64_with(&gpu, 700, &st, &ws, pack, true, gate, 7)
                .unwrap();
            since(&gpu, from)
        };
        let (kernel, prepare) = (run(GateAt::Kernel), run(GateAt::Prepare));
        assert_eq!(
            kernel
                .iter()
                .filter(|x| x.func == CHUNK64_PREP_CONV)
                .count(),
            3,
            "{pack:?}"
        );
        let want = expected(
            &kernel,
            (CHUNK64_PREP_CONV, CHUNK64_PREP_CONV_GRAW, 10),
            &graw_args(&lay, &ws),
            ws.gate,
        );
        assert_eq!(
            prepare.iter().map(key).collect::<Vec<_>>(),
            want,
            "{pack:?}"
        );
    }
}

/// 2026-10-14: Without either `_graw` kernel, `Prepare` launches nothing and errors, on both
/// prepares; `Kernel` on the same layer still runs.
#[test]
fn prepare_refuses_without_the_kernels_before_any_launch() {
    let gpu = MockGpuBackend::new();
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 300).unwrap();
    let st = seq_state(&gpu);
    let mut no_plain = layer(&gpu);
    no_plain.kernels.seq.chunk64_prepare_graw = KernelHandle(0);
    let mut no_conv = layer(&gpu);
    no_conv.kernels.seq.chunk64_prepare_conv_graw = KernelHandle(0);
    for (name, lay) in [("no plain", no_plain), ("no conv", no_conv)] {
        for fused in [false, true] {
            let from = gpu.launch_count();
            assert!(
                lay.stateful_chunk64_with(
                    &gpu,
                    130,
                    &st,
                    &ws,
                    Pack::InFrontEnd,
                    fused,
                    GateAt::Prepare,
                    7
                )
                .is_err(),
                "{name} fused={fused}"
            );
            assert_eq!(gpu.launch_count(), from, "{name} fused={fused}");
        }
        assert!(lay.gate_in_prepare_check().is_err(), "{name}");
        lay.stateful_chunk64_with(
            &gpu,
            130,
            &st,
            &ws,
            Pack::InFrontEnd,
            false,
            GateAt::Kernel,
            7,
        )
        .unwrap();
    }
}

/// 2026-10-14: Off the 64-token scan the gate stays in the front end whatever the lever says.
#[test]
fn only_the_64_token_scan_moves_the_gate() {
    let gpu = MockGpuBackend::new();
    assert_eq!(layer(&gpu).gate_at_for(false).unwrap(), GateAt::Kernel);
}

/// 2026-10-14: The lever: unset and `0` off, `1` on, anything else an error.
#[test]
fn the_lever_parses_strictly() {
    assert_eq!(parse_gate_in_prepare(None), Ok(false));
    assert_eq!(parse_gate_in_prepare(Some("0")), Ok(false));
    assert_eq!(parse_gate_in_prepare(Some("1")), Ok(true));
    for bad in ["", "on", "true", "2", " 1"] {
        assert!(parse_gate_in_prepare(Some(bad)).is_err(), "{bad:?}");
    }
}
