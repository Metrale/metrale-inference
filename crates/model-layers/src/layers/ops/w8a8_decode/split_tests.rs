// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `w8a8_gemv_split4`'s launch: the segment table it passes (pointers, pitches, the
//! row boundaries, an absent segment parked at N), its grid, and every refusal. The bytes are
//! checked on a GPU by `examples/glm5next_dsa_proj_in_gate`.

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};

use super::*;

fn seg(base: u64, n: u32, k: u32) -> SplitOut {
    SplitOut {
        w: Fp8Weight {
            weight: DevicePtr(base),
            row_scale: DevicePtr(base + 1),
            n,
            k,
            scale_format: WeightQuantFormat::Fp8PerRow,
        },
        out: DevicePtr(base + 2),
        ldc: n,
    }
}

fn scratch() -> W8a8Scratch {
    W8a8Scratch {
        q: DevicePtr(0x10),
        q_bytes: 8 * 4096,
        scale: DevicePtr(0x20),
        scale_bytes: 8 * 4,
    }
}

fn u32a(v: u32) -> MockArg {
    MockArg::Bytes(v.to_le_bytes().to_vec())
}

/// 2026-10-21: Four segments (the DSA q_a, kv_a, wk, compress gate shapes): one launch over
/// all their tiles, each segment's weight, scales, output and pitch in order, the boundaries the
/// prefix sums. Three segments: the fourth repeats segment 0 and the last boundary is N.
#[test]
fn the_launch_carries_each_segment_and_the_prefix_boundaries() {
    let gpu = MockGpuBackend::new();
    let s = [
        seg(0x1000, 1536, 4096),
        seg(0x2000, 512, 4096),
        seg(0x3000, 128, 4096),
        seg(0x4000, 128, 4096),
    ];
    w8a8_gemv_split4(&gpu, KernelHandle(9), &s, &scratch(), 2, 7).unwrap();
    let three = &s[..3];
    w8a8_gemv_split4(&gpu, KernelHandle(9), three, &scratch(), 1, 7).unwrap();
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 2);
    assert_eq!(l[0].grid, [(1536 + 512 + 256) / 16, 1, 1]);
    assert_eq!(l[0].block, [256, 1, 1]);
    assert_eq!(l[1].grid, [(1536 + 512 + 128) / 16, 1, 1]);
    let table = |segs: [&SplitOut; 4]| -> Vec<MockArg> {
        segs.iter()
            .flat_map(|s| {
                [
                    MockArg::Buffer(s.w.weight),
                    MockArg::Buffer(s.w.row_scale),
                    MockArg::Buffer(s.out),
                    u32a(s.ldc),
                ]
            })
            .collect()
    };
    assert_eq!(l[0].args[..2], [MockArg::Buffer(DevicePtr(0x10)), MockArg::Buffer(DevicePtr(0x20))]);
    assert_eq!(l[0].args[2..18], table([&s[0], &s[1], &s[2], &s[3]])[..]);
    assert_eq!(
        l[0].args[18..],
        [u32a(2), u32a(2304), u32a(4096), u32a(4096), u32a(1536), u32a(2048), u32a(2176)]
    );
    assert_eq!(l[1].args[2..18], table([&s[0], &s[1], &s[2], &s[0]])[..]);
    assert_eq!(
        l[1].args[18..],
        [u32a(1), u32a(2176), u32a(4096), u32a(4096), u32a(1536), u32a(2048), u32a(2176)]
    );
}

/// 2026-10-21: Every shape the kernel cannot serve is refused before a launch.
#[test]
fn shapes_the_kernel_cannot_serve_are_refused() {
    let gpu = MockGpuBackend::new();
    let ok = seg(0x1000, 1536, 4096);
    let bad_k = seg(0x2000, 512, 4000);
    let mut block = seg(0x2000, 512, 4096);
    block.w.scale_format = WeightQuantFormat::Fp8BlockScaled;
    let mut short_ldc = seg(0x2000, 512, 4096);
    short_ldc.ldc = 500;
    let odd = seg(0x2000, 520, 4096);
    let cases: Vec<(Vec<SplitOut>, usize, &str)> = vec![
        (vec![], 1, "0 projections"),
        (vec![ok; 5], 1, "5 projections"),
        (vec![ok], 0, "0 rows"),
        (vec![ok], 9, "9 rows"),
        (vec![ok, bad_k], 1, "projection 1 is [512, 4000]"),
        (vec![ok, block], 1, "projection 1 is [512, 4096] Fp8BlockScaled"),
        (vec![ok, short_ldc], 1, "projection 1: output ldc=500"),
        (vec![odd, ok], 1, "projection 0 has 520 rows"),
    ];
    for (segs, rows, want) in cases {
        let e = w8a8_gemv_split4(&gpu, KernelHandle(9), &segs, &scratch(), rows, 0).unwrap_err();
        assert!(e.to_string().contains(want), "{want}: {e}");
    }
    // 2026-10-21: The last segment may be any height; an unresolved kernel is refused.
    w8a8_gemv_split4(&gpu, KernelHandle(9), &[ok, odd], &scratch(), 1, 0).unwrap();
    let e = w8a8_gemv_split4(&gpu, KernelHandle(0), &[ok], &scratch(), 1, 0).unwrap_err();
    assert!(e.to_string().contains("not in this target"), "{e}");
    assert_eq!(gpu.launches_snapshot().len(), 1);
}
