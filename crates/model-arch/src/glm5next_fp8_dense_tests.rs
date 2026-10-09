// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The FP8 dense tier's registry and dispatch on a recording mock backend, and the
//! shared expert's TP split under it.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::*;

/// 2026-10-09: Under the tier the shared expert (2048 wide) splits over three ranks in 128-wide
/// units, so every rank's down projection has whole W8A8 chunks; otherwise in the BF16 unit.
#[test]
fn the_shared_expert_splits_in_whole_fp8_chunks_under_the_tier() {
    let widths = |unit| -> Vec<usize> {
        (0..3)
            .map(|r| metrale_config::tp_split(2048, 3, r, unit).unwrap().len)
            .collect()
    };
    let fp8 = widths(shared_split_unit_for(true, 8));
    assert_eq!(fp8, vec![768, 640, 640]);
    assert!(fp8.iter().all(|k| k % FP8_K_UNIT == 0));
    assert_eq!(widths(shared_split_unit_for(false, 8)), vec![688, 680, 680]);
}

/// 2026-10-09: One test owns the process-wide registry. A registered weight runs W8A8 in
/// 256-row chunks (a quantize and its GEMVs each); an unregistered one declines; a launch at another
/// shape is an error; a K that is not whole 128-wide chunks or a registration before
/// `prepare` is refused; re-registering the same shape quantizes nothing new.
#[test]
fn registered_weights_run_w8a8_and_everything_else_declines_or_errors() {
    let gpu = MockGpuBackend::new();
    let quant = KernelHandle(0x77);
    let w = gpu.alloc(64 * 4096 * 2).unwrap();
    let e = register(&gpu, quant, w, 64, 4096, "early", 0)
        .unwrap_err()
        .to_string();
    assert!(e.contains("before prepare"), "{e}");

    prepare(&gpu, 4096).unwrap();
    register(&gpu, quant, w, 64, 4096, "q", 0).unwrap();
    let quantized = gpu
        .launches_snapshot()
        .iter()
        .filter(|l| l.func == 0x77)
        .count();
    assert_eq!(quantized, 1, "one weight quantization");
    register(&gpu, quant, w, 64, 4096, "q again", 0).unwrap();
    assert_eq!(
        gpu.launches_snapshot()
            .iter()
            .filter(|l| l.func == 0x77)
            .count(),
        1,
        "re-registering quantizes nothing"
    );
    let e = register(&gpu, quant, w, 32, 4096, "other shape", 0)
        .unwrap_err()
        .to_string();
    assert!(e.contains("registered as [64, 4096]"), "{e}");
    let odd = gpu.alloc(64 * 688 * 2).unwrap();
    let e = register(&gpu, quant, odd, 64, 688, "shared down", 0)
        .unwrap_err()
        .to_string();
    assert!(e.contains("multiple of 128"), "{e}");

    let (a, c) = (DevicePtr(0x1000_0000), DevicePtr(0x2000_0000));
    let before = gpu.launches_snapshot().len();
    assert!(proj_registered(&gpu, w, a, c, 300, 64, 4096, 3).unwrap());
    let l = gpu.launches_snapshot();
    let ours = &l[before..];
    // 2026-10-09: Two 256-row chunks (256 + 44), each one quantize; the GEMV launches 128 rows
    // at a time (`W8A8_LAUNCH_ROWS`), so the first chunk has two.
    assert_eq!(
        ours.iter().map(|l| l.grid[0]).collect::<Vec<_>>(),
        vec![256, 4, 4, 44, 4],
        "quantize grid = rows, GEMV grid = ceil(64 / 16)"
    );
    assert!(ours.iter().all(|l| l.stream == 3));

    let other = DevicePtr(0x3000_0000);
    let n = gpu.launches_snapshot().len();
    assert!(!proj_registered(&gpu, other, a, c, 1, 64, 4096, 0).unwrap());
    assert_eq!(
        gpu.launches_snapshot().len(),
        n,
        "a declined weight launches nothing"
    );
    let e = proj_registered(&gpu, w, a, c, 1, 64, 2048, 0)
        .unwrap_err()
        .to_string();
    assert!(e.contains("launched as [64, 2048]"), "{e}");
    assert_eq!(registered(), (1, 64 * (4096 + 4)));
}
