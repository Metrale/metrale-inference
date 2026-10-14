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
use super::super::decode_rows_tests::{GATES_FUSED, cfg, layer};

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
/// dt_bias / A_log in and gate out, beta_bf16 in and beta out, then M, N, D, heads, lower bound.
/// 2026-10-14: `t` rows take ceil(t / 8) token tiles on grid y: 1 at 1 and 8 rows, 2 at 9 and 16.
#[test]
fn one_launch_reads_and_writes_the_unfused_chain_buffers_in_kernel_order() {
    for (t, tiles) in [(1usize, 1u32), (2, 1), (8, 1), (9, 2), (16, 2), (17, 3)] {
        launch_at(t, tiles);
    }
}

fn launch_at(t: usize, tiles: u32) {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let c = cfg();
    let ws = Glm5NextKdaWorkspace::new(&gpu, &c, t).unwrap();
    let (fb, gb) = (fp8(&gpu, 256, 128), fp8(&gpu, 256, 128));
    let from = gpu.launches_snapshot().len();
    l.launch_gates_fused(&gpu, t, true, (Some(fb), Some(gb)), &ws, 7)
        .unwrap();
    let launches = gpu.launches_snapshot()[from..].to_vec();
    assert_eq!(launches.len(), 1);
    let k = &launches[0];
    assert_eq!(k.func, GATES_FUSED);
    assert_eq!((k.grid, k.block, k.stream), ([4, tiles, 1], [256, 1, 1], 7));
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
    assert_eq!(word(k, 13), (t as u32).to_le_bytes());
    assert_eq!(word(k, 14), 256u32.to_le_bytes());
    assert_eq!(word(k, 15), 128u32.to_le_bytes());
    assert_eq!(word(k, 16), 2u32.to_le_bytes());
    assert_eq!(word(k, 17), c.gate_lower_bound.to_le_bytes());
}

/// 2026-10-14: The widths that fuse are exactly those whose unfused projections run the W8A8
/// GEMV: 1..=W8A8_MAX_ROWS, less any the dequantize + cuBLASLt floor takes (its edge row
/// included), and never zero rows.
#[test]
fn fused_rows_are_the_w8a8_gemv_widths() {
    use super::{W8A8_MAX_ROWS, fused_rows};
    assert!(!fused_rows(0, None));
    assert!(fused_rows(1, None));
    assert!(fused_rows(W8A8_MAX_ROWS, None));
    assert!(!fused_rows(W8A8_MAX_ROWS + 1, None));
    assert!(fused_rows(16, Some(17)));
    assert!(!fused_rows(17, Some(17)));
    assert!(fused_rows(W8A8_MAX_ROWS, Some(1024)));
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
            .launch_gates_fused(&gpu, 1, low, w, &ws, 0)
            .unwrap_err()
            .to_string();
        assert!(e.contains(want), "{e}");
    }
    let mut no_kernel = l;
    no_kernel.kernels.gates_fused = KernelHandle(0);
    let e = no_kernel
        .launch_gates_fused(&gpu, 1, true, (Some(ok), Some(ok)), &ws, 0)
        .unwrap_err()
        .to_string();
    assert!(e.contains("not in this target"), "{e}");
    assert_eq!(gpu.launches_snapshot().len(), from);
}
