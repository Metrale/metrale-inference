// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The one-row segmented GEMV launch: its contract against the kernel source, the
//! grid and shared-memory sizing, and the launch it makes on a recording mock backend.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};

use super::*;
use crate::weight_map::QuantizedWeight;

const CU: &str = include_str!("../../../../../kernels/gb10/common/w4a16_gemv_seg.cu");

/// 2026-10-10: The host's constants are the kernel's: block size, segment count, entries, and
/// the shared-memory layout the sizing assumes (256 uint4 per 128 chunks of 16 values).
#[test]
fn host_constants_match_the_kernel() {
    assert!(CU.contains(&format!(
        "#define SEG1_THREADS {}\n",
        W4A16_GEMV_SEG_WARPS * 32
    )));
    assert!(CU.contains(&format!("#define SEG1_WARPS {W4A16_GEMV_SEG_WARPS}\n")));
    assert_eq!(
        CU.matches("SEG1_PARAMS(").count(),
        W4A16_GEMV_SEG_MAX + 1,
        "one parameter set per segment, plus the macro's definition"
    );
    for r in W4A16_GEMV_SEG_TILES {
        assert!(CU.contains(&format!("SEG1_ENTRY({r})\n")), "tile {r}");
        assert_eq!(
            w4a16_gemv_seg_entry(r),
            Some(format!("w4a16_gemv_seg_r{r}")).as_deref()
        );
    }
    assert_eq!(w4a16_gemv_seg_entry(3), None);
    assert!(CU.contains("s_a[256u * j + 64u * (2u * (r & 1u) + h) + (r >> 1)] = A4[u];"));
    assert!(CU.contains("const unsigned int J = (K16 + 127) / 128;"));
}

/// 2026-10-10: The contract refuses each bound it names, and the shared memory covers the
/// permuted input at every K it admits, never above 48 KB.
#[test]
fn shape_contract_and_smem() {
    assert!(w4a16_gemv_seg_shape_ok(
        4096,
        &[2816, 2816, 2816, 128, 22, 128]
    ));
    assert!(w4a16_gemv_seg_shape_ok(768, &[4096]));
    assert!(w4a16_gemv_seg_shape_ok(11264, &[4096]));
    assert!(!w4a16_gemv_seg_shape_ok(4096, &[]));
    assert!(!w4a16_gemv_seg_shape_ok(4096, &[1; 7]));
    assert!(!w4a16_gemv_seg_shape_ok(4096, &[128, 0]));
    assert!(!w4a16_gemv_seg_shape_ok(0, &[128]));
    assert!(!w4a16_gemv_seg_shape_ok(4096 + 16, &[128]), "K % 32");
    assert!(w4a16_gemv_seg_shape_ok(W4A16_GEMV_SEG_MAX_K, &[1]));
    assert!(!w4a16_gemv_seg_shape_ok(W4A16_GEMV_SEG_MAX_K + 32, &[1]));
    for k in (32..=W4A16_GEMV_SEG_MAX_K).step_by(32) {
        let chunk_blocks = (k / 16).div_ceil(128);
        assert_eq!(w4a16_gemv_seg_smem(k), chunk_blocks * 256 * 16, "K {k}");
        assert!(w4a16_gemv_seg_smem(k) <= 48 * 1024);
    }
}

/// 2026-10-10: Tiles round each segment up separately; blocks give each warp a tile up to the cap.
#[test]
fn tiles_and_blocks() {
    assert_eq!(w4a16_gemv_seg_tiles(&[2816, 2816, 2816], 2), 4224);
    assert_eq!(w4a16_gemv_seg_tiles(&[128, 22, 128], 4), 32 + 6 + 32);
    assert_eq!(w4a16_gemv_seg_tiles(&[3, 3], 2), 4);
    assert_eq!(w4a16_gemv_seg_blocks(4224, 192), 192);
    assert_eq!(w4a16_gemv_seg_blocks(70, 192), 9);
    assert_eq!(w4a16_gemv_seg_blocks(8, 192), 1);
    assert_eq!(w4a16_gemv_seg_blocks(1, 192), 1);
}

fn bytes(a: &MockArg) -> &[u8] {
    match a {
        MockArg::Bytes(b) => b,
        MockArg::Buffer(p) => panic!("expected bytes, got buffer {p}"),
    }
}

fn ptr(a: &MockArg) -> DevicePtr {
    match a {
        MockArg::Buffer(p) => *p,
        MockArg::Bytes(_) => panic!("expected a buffer"),
    }
}

/// 2026-10-10: Two segments: the grid, block and shared memory, the segments in the kernel's
/// order, the four empty slots, then K; a refused shape or tile height launches nothing.
#[test]
fn the_launch_carries_each_segment() {
    let gpu = MockGpuBackend::new();
    let w = |base: u64, s2: f32| QuantizedWeight {
        weight: DevicePtr(base),
        weight_scale: DevicePtr(base + 0x100),
        weight_scale_2: s2,
        ..QuantizedWeight::null()
    };
    let segs = [
        W4a16Seg {
            weight: w(0x1000, 0.5),
            output: DevicePtr(0x9000),
            n: 768,
        },
        W4a16Seg {
            weight: w(0x2000, 0.25),
            output: DevicePtr(0xA000),
            n: 768,
        },
    ];
    let a = DevicePtr(0x5000);
    w4a16_gemv_seg(&gpu, a, &segs, 4096, 2, 100, 3).unwrap();
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 1);
    let l = &l[0];
    assert_eq!(
        (l.grid, l.block, l.shared_mem, l.stream),
        ([96, 1, 1], [256, 1, 1], 8192, 3)
    );
    assert_eq!(l.args.len(), 1 + 5 * W4A16_GEMV_SEG_MAX + 1);
    assert_eq!(ptr(&l.args[0]), a);
    for (i, g) in segs.iter().enumerate() {
        let b = 1 + 5 * i;
        assert_eq!(ptr(&l.args[b]), g.weight.weight);
        assert_eq!(ptr(&l.args[b + 1]), g.weight.weight_scale);
        assert_eq!(bytes(&l.args[b + 2]), g.weight.weight_scale_2.to_le_bytes());
        assert_eq!(ptr(&l.args[b + 3]), g.output);
        assert_eq!(bytes(&l.args[b + 4]), g.n.to_le_bytes());
    }
    for i in segs.len()..W4A16_GEMV_SEG_MAX {
        let b = 1 + 5 * i;
        assert_eq!(ptr(&l.args[b]), DevicePtr::NULL);
        assert_eq!(bytes(&l.args[b + 4]), 0u32.to_le_bytes());
    }
    assert_eq!(bytes(&l.args[31]), 4096u32.to_le_bytes());
    assert!(
        gpu.kernel_lookups_snapshot()
            .contains(&(W4A16_GEMV_SEG_MODULE.into(), "w4a16_gemv_seg_r2".into()))
    );
    assert!(w4a16_gemv_seg(&gpu, a, &segs, 4096 + 16, 2, 100, 3).is_err());
    assert!(w4a16_gemv_seg(&gpu, a, &segs, 4096, 3, 100, 3).is_err());
    assert!(w4a16_gemv_seg(&gpu, a, &segs, 4096, 2, 0, 3).is_err());
    assert_eq!(gpu.launches_snapshot().len(), 1);
}
