// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The fused KDA gate launch on the mock backend: which buffers it reads and writes,
//! in which argument order, over which grid, and the refusals before any launch. The bytes against
//! the unfused chain are the GPU microtest's job (`examples/glm5next_c1_fuse_microtest.rs`).
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::weight_map::{Fp8Weight, WeightQuantFormat};

use super::super::Glm5NextKdaWorkspace;
use super::super::decode_rows_tests::{GATES_FUSED, GATES_ROWS, cfg, layer};
use super::GATES_MAX_ROWS;

fn fp8(gpu: &MockGpuBackend, n: u32, k: u32) -> Fp8Weight {
    Fp8Weight {
        weight: gpu.alloc(64).unwrap(),
        row_scale: gpu.alloc(64).unwrap(),
        n,
        k,
        scale_format: WeightQuantFormat::Fp8PerRow,
    }
}

fn ptr(l: &MockLaunch, i: usize) -> DevicePtr {
    match &l.args[i] {
        MockArg::Buffer(p) => *p,
        other => panic!("arg {i} is {other:?}, not a pointer"),
    }
}

fn word(l: &MockLaunch, i: usize) -> [u8; 4] {
    match &l.args[i] {
        MockArg::Bytes(b) => b[..4].try_into().unwrap(),
        other => panic!("arg {i} is {other:?}, not 4 bytes"),
    }
}

/// 2026-10-10: One launch over 2 x 256 / 16 = 32 warp tiles (4 blocks of 8 warps): the two
/// low-rank rows in, f_b's and g_b's FP8 rows and scales, g_raw and out_gate out, the gate's
/// dt_bias / A_log in and gate out, beta_bf16 in and beta out, then N, D, heads, lower bound.
#[test]
fn one_launch_reads_and_writes_the_unfused_chain_buffers_in_kernel_order() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let c = cfg();
    let ws = Glm5NextKdaWorkspace::new(&gpu, &c, 1).unwrap();
    let (fb, gb) = (fp8(&gpu, 256, 128), fp8(&gpu, 256, 128));
    let from = gpu.launches_snapshot().len();
    l.launch_gates_fused(&gpu, true, (Some(fb), Some(gb)), &ws, 7)
        .unwrap();
    let launches = gpu.launches_snapshot()[from..].to_vec();
    assert_eq!(launches.len(), 1);
    let k = &launches[0];
    assert_eq!(k.func, GATES_FUSED);
    assert_eq!((k.grid, k.block, k.stream), ([4, 1, 1], [256, 1, 1], 7));
    let want = [
        ws.lowrank,
        ws.lowrank_g,
        fb.weight,
        fb.row_scale,
        gb.weight,
        gb.row_scale,
        ws.g_raw,
        ws.out_gate,
        l.weights.dt_bias,
        l.weights.a_log,
        ws.gate,
        ws.beta_bf16,
        ws.beta,
    ];
    for (i, p) in want.iter().enumerate() {
        assert_eq!(ptr(k, i), *p, "pointer argument {i}");
    }
    assert_eq!(word(k, 13), 256u32.to_le_bytes());
    assert_eq!(word(k, 14), 128u32.to_le_bytes());
    assert_eq!(word(k, 15), 2u32.to_le_bytes());
    assert_eq!(word(k, 16), c.gate_lower_bound.to_le_bytes());
}

/// 2026-10-10: Without the low-rank group, without either FP8 copy, at another shape or without
/// the kernel, the lever is refused and nothing is launched.
#[test]
fn refusals_come_before_any_launch() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 1).unwrap();
    let ok = fp8(&gpu, 256, 128);
    let from = gpu.launches_snapshot().len();
    let cases = [
        (true, (Some(ok), None), "per-row FP8 tier"),
        (true, (None, Some(ok)), "per-row FP8 tier"),
        (false, (Some(ok), Some(ok)), "low-rank group"),
        (
            true,
            (Some(fp8(&gpu, 256, 256)), Some(ok)),
            "f_b is [256, 256]",
        ),
        (
            true,
            (Some(ok), Some(fp8(&gpu, 128, 128))),
            "g_b is [128, 128]",
        ),
    ];
    for (low, w, want) in cases {
        let e = l
            .launch_gates_fused(&gpu, low, w, &ws, 0)
            .unwrap_err()
            .to_string();
        assert!(e.contains(want), "{e}");
    }
    let mut no_kernel = l;
    no_kernel.kernels.gates_fused = KernelHandle(0);
    let e = no_kernel
        .launch_gates_fused(&gpu, true, (Some(ok), Some(ok)), &ws, 0)
        .unwrap_err()
        .to_string();
    assert!(e.contains("not in this target"), "{e}");
    assert_eq!(gpu.launches_snapshot().len(), from);
}

/// 2026-10-21: The rows kernel at 2 rows: one launch over the one-token kernel's grid, the same
/// buffers in the same order, then M = 2 ahead of N, D, heads and the lower bound.
#[test]
fn the_rows_kernel_takes_m_ahead_of_the_shape_arguments() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let c = cfg();
    let ws = Glm5NextKdaWorkspace::new(&gpu, &c, 2).unwrap();
    let (fb, gb) = (fp8(&gpu, 256, 128), fp8(&gpu, 256, 128));
    let from = gpu.launches_snapshot().len();
    assert!(
        l.try_gates_rows(&gpu, 2, true, (Some(fb), Some(gb)), &ws, 7)
            .unwrap()
    );
    let launches = gpu.launches_snapshot()[from..].to_vec();
    assert_eq!(launches.len(), 1);
    let k = &launches[0];
    assert_eq!(k.func, GATES_ROWS);
    assert_eq!((k.grid, k.block, k.stream), ([4, 1, 1], [256, 1, 1], 7));
    assert_eq!(ptr(k, 0), ws.lowrank);
    assert_eq!(ptr(k, 1), ws.lowrank_g);
    assert_eq!(ptr(k, 12), ws.beta);
    assert_eq!(word(k, 13), 2u32.to_le_bytes());
    assert_eq!(word(k, 14), 256u32.to_le_bytes());
    assert_eq!(word(k, 15), 128u32.to_le_bytes());
    assert_eq!(word(k, 16), 2u32.to_le_bytes());
    assert_eq!(word(k, 17), c.gate_lower_bound.to_le_bytes());
}

/// 2026-10-21: The rows kernel declines (nothing launched, the unfused chain runs) at 0 or more
/// than `GATES_MAX_ROWS` rows, with its lever off, and wherever the one-row lever is refused.
#[test]
fn the_rows_kernel_declines_out_of_range_off_or_refused() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), GATES_MAX_ROWS + 1).unwrap();
    let ok = fp8(&gpu, 256, 128);
    let from = gpu.launches_snapshot().len();
    let both = (Some(ok), Some(ok));
    for t in [0, GATES_MAX_ROWS + 1] {
        assert!(
            !l.try_gates_rows(&gpu, t, true, both, &ws, 0).unwrap(),
            "t={t}"
        );
    }
    for (low, w) in [
        (false, both),
        (true, (Some(ok), None)),
        (true, (None, Some(ok))),
        (true, (Some(fp8(&gpu, 256, 256)), Some(ok))),
    ] {
        assert!(!l.try_gates_rows(&gpu, 2, low, w, &ws, 0).unwrap());
    }
    let mut off = layer(&gpu);
    off.levers.kda.gates_rows = false;
    assert!(!off.try_gates_rows(&gpu, 2, true, both, &ws, 0).unwrap());
    let mut no_kernel = layer(&gpu);
    no_kernel.kernels.gates_rows = KernelHandle(0);
    assert!(
        !no_kernel
            .try_gates_rows(&gpu, 2, true, both, &ws, 0)
            .unwrap()
    );
    assert_eq!(gpu.launches_snapshot().len(), from);
    assert!(
        l.try_gates_rows(&gpu, GATES_MAX_ROWS, true, both, &ws, 0)
            .unwrap()
    );
    assert!(l.try_gates_rows(&gpu, 1, true, both, &ws, 0).unwrap());
}
