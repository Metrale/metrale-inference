// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `proj_split` on a recording mock backend: the quantization it makes or reuses,
//! the one split launch with each registered projection's output, and every case it declines
//! (launching nothing) or refuses. The bytes are checked on a GPU by
//! `examples/glm5next_dsa_proj_in_gate`.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::*;
use crate::glm5next_fp8_dense::{prepare, register, state};

const K: usize = 4096;
/// 2026-10-21: The DSA q_a, kv_a, wk and compress gate widths.
const NS: [usize; 4] = [1536, 512, 128, 128];

fn outs(ws: &[DevicePtr; 4]) -> Vec<SplitProj> {
    (0..4)
        .map(|i| SplitProj {
            b: ws[i],
            c: DevicePtr(0x7000_0000 + i as u64 * 0x10_0000),
            n: NS[i],
        })
        .collect()
}

fn set_stable(a: Option<u64>, quantized: Option<(u64, usize, usize, u64)>) {
    let mut g = state().lock().unwrap();
    let s = g.as_mut().unwrap();
    s.stable = a;
    s.quantized = quantized;
}

#[test]
fn four_registered_projections_run_as_one_quantization_and_one_split_launch() {
    let _serial = crate::glm5next_fp8_dense::lock_registries_for_test();
    let gpu = MockGpuBackend::new();
    prepare(&gpu, K, None).unwrap();
    let ws = [0x5100_0000u64, 0x5200_0000, 0x5300_0000, 0x5400_0000].map(DevicePtr);
    for (w, n) in ws.iter().zip(NS) {
        register(&gpu, KernelHandle(0x79), *w, n, K, "split", 0).unwrap();
    }
    let a = DevicePtr(0x6100_0000);
    set_stable(None, None);
    let before = gpu.launches_snapshot().len();
    assert!(proj_split_registered(&gpu, a, 2, K, &outs(&ws), 5).unwrap());
    let l = gpu.launches_snapshot()[before..].to_vec();
    assert_eq!(
        l.iter().map(|l| l.grid[0]).collect::<Vec<_>>(),
        vec![2, (1536 + 512 + 128 + 128) / 16],
        "one quantization of the two rows, one launch over every projection's tiles"
    );
    assert!(l.iter().all(|l| l.stream == 5));
    assert_eq!(l[0].args[0], MockArg::Buffer(a));
    let c: Vec<_> = outs(&ws).iter().map(|o| MockArg::Buffer(o.c)).collect();
    assert_eq!([&l[1].args[4], &l[1].args[8], &l[1].args[12], &l[1].args[16]], [&c[0], &c[1], &c[2], &c[3]]);

    // 2026-10-21: Under a stable guard whose quantization the scratch holds, no quantization.
    set_stable(Some(a.0), Some((a.0, 2, K, 5)));
    let n = gpu.launches_snapshot().len();
    assert!(proj_split_registered(&gpu, a, 2, K, &outs(&ws), 5).unwrap());
    assert_eq!(gpu.launches_snapshot().len() - n, 1, "the stable quantization is reused");
    // 2026-10-21: The split launch leaves the scratch holding `a`'s quantization.
    set_stable(Some(a.0), None);
    let n = gpu.launches_snapshot().len();
    assert!(proj_split_registered(&gpu, a, 2, K, &outs(&ws), 5).unwrap());
    assert!(proj_split_registered(&gpu, a, 2, K, &outs(&ws), 5).unwrap());
    assert_eq!(gpu.launches_snapshot().len() - n, 3, "quantize, split, split");
    set_stable(None, None);
}

#[test]
fn what_the_split_launch_cannot_serve_declines_or_errors_with_nothing_launched() {
    let _serial = crate::glm5next_fp8_dense::lock_registries_for_test();
    let gpu = MockGpuBackend::new();
    prepare(&gpu, K, None).unwrap();
    let ws = [0x5500_0000u64, 0x5600_0000, 0x5700_0000, 0x5800_0000].map(DevicePtr);
    for (w, n) in ws.iter().zip(NS) {
        register(&gpu, KernelHandle(0x79), *w, n, K, "split", 0).unwrap();
    }
    let odd = DevicePtr(0x5900_0000);
    register(&gpu, KernelHandle(0x79), odd, 520, K, "odd", 0).unwrap();
    let a = DevicePtr(0x6100_0000);
    let n0 = gpu.launches_snapshot().len();
    for m in [0, 9] {
        assert!(!proj_split_registered(&gpu, a, m, K, &outs(&ws), 0).unwrap(), "m={m}");
    }
    let mut unregistered = outs(&ws);
    unregistered[2].b = DevicePtr(0x5a00_0000);
    assert!(!proj_split_registered(&gpu, a, 1, K, &unregistered, 0).unwrap());
    let mut untiled = outs(&ws);
    untiled[0] = SplitProj { b: odd, c: untiled[0].c, n: 520 };
    assert!(!proj_split_registered(&gpu, a, 1, K, &untiled, 0).unwrap());
    let mut wrong = outs(&ws);
    wrong[1].n = 256;
    let e = proj_split_registered(&gpu, a, 1, K, &wrong, 0).unwrap_err().to_string();
    assert!(e.contains("launched as [256, 4096]"), "{e}");
    assert_eq!(gpu.launches_snapshot().len(), n0, "nothing launched");
    // 2026-10-21: The untiled projection last is a shape the kernel serves.
    let mut last = outs(&ws);
    last[3] = SplitProj { b: odd, c: last[3].c, n: 520 };
    assert!(proj_split_registered(&gpu, a, 1, K, &last, 0).unwrap());
}
