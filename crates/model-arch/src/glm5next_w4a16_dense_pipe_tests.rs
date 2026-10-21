// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `METRALE_GLM_W4A16_GEMV_PIPE`: which launches it moves to `w4a16_gemv_pipe` (the
//! 1- and 2-row launches of `METRALE_GLM_W4A16_GEMV_SEG`, and a lone 2-row projection from K
//! 2048 up), on a recording mock backend; and its switch.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::gpu::mock::{MockGpuBackend, MockLaunch};
use metrale_model_layers::lever_equivalence::settings_from;

use super::super::*;
use super::{OFF, ptr_arg, register_all, u32_arg};
use crate::glm5next_dense_levers::DenseLevers;

/// 2026-10-21: The serving plan on GB10 (48 SMs): GEMV rows 4, the segmented GEMV and the pipe.
const PIPE: SegPlan = SegPlan {
    one_row: true,
    gemv_rows: 4,
    sms: 48,
    pipe: true,
    ..OFF
};

fn looked_up(gpu: &MockGpuBackend, module: &str, entry: &str) -> bool {
    gpu.kernel_lookups_snapshot()
        .iter()
        .any(|(md, f)| md == module && f == entry)
}

fn one(gpu: &MockGpuBackend, start: usize) -> MockLaunch {
    let l = gpu.launches_snapshot()[start..].to_vec();
    assert_eq!(l.len(), 1, "one launch");
    l[0].clone()
}

/// 2026-10-21: The pipe takes a launch exactly where the segmented GEMV would at 1 and 2 rows,
/// and needs the lever, the segmented GEMV and (at 2 rows) the GEMV tier.
#[test]
fn pipe_rows_follow_the_segmented_gemv_at_one_and_two_rows() {
    assert!(PIPE.pipe_rows(1) && PIPE.pipe_rows(2));
    assert!(!PIPE.pipe_rows(3) && !PIPE.pipe_rows(4) && !PIPE.pipe_rows(0));
    assert!(PIPE.seg_gemv(3), "3 rows stay on w4a16_gemv_seg_m3");
    let off = SegPlan {
        pipe: false,
        ..PIPE
    };
    assert!(!off.pipe_rows(1) && !off.pipe_rows(2));
    let no_seg = SegPlan {
        one_row: false,
        ..PIPE
    };
    assert!(!no_seg.pipe_rows(1) && !no_seg.pipe_rows(2));
    let no_tier = SegPlan {
        gemv_rows: 0,
        ..PIPE
    };
    assert!(no_tier.pipe_rows(1) && !no_tier.pipe_rows(2));
}

/// 2026-10-21: The shared down (K 768) at 1 and 2 rows and KDA o (K 2816) at 2 rows are one
/// `w4a16_gemv_pipe_m<M>` launch on one resident wave (192 blocks, M staged rows); KDA o at one
/// row keeps `w4a16_gemv` (7 arguments) and at 3 rows the batch tier; 3 rows of the shared down
/// keep `w4a16_gemv_seg_m3`. With the lever off, KDA o at 2 rows is `w4a16_gemv_batch2` again
/// and the shared down `w4a16_gemv_seg_m2`.
#[test]
fn lone_projections_take_the_pipe_at_one_and_two_rows() {
    let _serial = crate::glm5next_fp8_dense::lock_registries_for_test();
    let gpu = MockGpuBackend::new();
    let keys = register_all(&gpu, &[(4096, 768), (4096, 2816)]);
    let (down, o) = (keys[0], keys[1]);
    let (a, c) = (DevicePtr(0x1000_0000), DevicePtr(0x2000_0000));
    for (key, k, m) in [(down, 768usize, 1usize), (down, 768, 2), (o, 2816, 2)] {
        let start = gpu.launches_snapshot().len();
        assert!(proj_registered(&gpu, PIPE, key, a, c, m, 4096, k, 5).unwrap());
        let l = one(&gpu, start);
        let j = (k as u32 / 16).div_ceil(128);
        assert_eq!(
            (l.grid, l.block, l.shared_mem, l.stream),
            ([192, 1, 1], [256, 1, 1], m as u32 * j * 4096 + 64, 5),
            "K {k}, {m} rows"
        );
        assert_eq!(l.args.len(), 32);
        assert_eq!(ptr_arg(&l.args[0]), a);
        assert_eq!(ptr_arg(&l.args[1]), key);
        assert_eq!(ptr_arg(&l.args[4]), c);
        assert_eq!(u32_arg(&l.args[5]), 4096);
        assert_eq!(u32_arg(&l.args[31]), k as u32);
        assert!(looked_up(&gpu, "w4a16_gemv_pipe", &format!("w4a16_gemv_pipe_m{m}")));
    }
    let start = gpu.launches_snapshot().len();
    assert!(proj_registered(&gpu, PIPE, o, a, c, 1, 4096, 2816, 5).unwrap());
    assert_eq!(one(&gpu, start).args.len(), 7, "KDA o, one row: w4a16_gemv");
    let start = gpu.launches_snapshot().len();
    assert!(proj_registered(&gpu, PIPE, o, a, c, 3, 4096, 2816, 5).unwrap());
    assert_eq!(one(&gpu, start).args.len(), 7, "KDA o, 3 rows: w4a16_gemv_batch3");
    assert!(looked_up(&gpu, "w4a16_gemv", "w4a16_gemv_batch3"));
    let start = gpu.launches_snapshot().len();
    assert!(proj_registered(&gpu, PIPE, down, a, c, 3, 4096, 768, 5).unwrap());
    assert_eq!(one(&gpu, start).shared_mem, 3 * 4096 + 64);
    assert!(looked_up(&gpu, "w4a16_gemv_seg", "w4a16_gemv_seg_m3"));

    let off = SegPlan {
        pipe: false,
        ..PIPE
    };
    let start = gpu.launches_snapshot().len();
    assert!(proj_registered(&gpu, off, o, a, c, 2, 4096, 2816, 5).unwrap());
    let l = one(&gpu, start);
    assert_eq!((l.args.len(), l.grid), (7, [1024, 1, 1]), "w4a16_gemv_batch2");
    assert!(looked_up(&gpu, "w4a16_gemv", "w4a16_gemv_batch2"));
    let start = gpu.launches_snapshot().len();
    assert!(proj_registered(&gpu, off, down, a, c, 2, 4096, 768, 5).unwrap());
    assert_eq!(one(&gpu, start).grid, [512, 1, 1], "w4a16_gemv_seg_m2, uncapped");
    assert!(looked_up(&gpu, "w4a16_gemv_seg", "w4a16_gemv_seg_m2"));
}

/// 2026-10-21: The KDA six-group at 1 and 2 rows is one pipe launch with every member, one
/// output per warp (1091 blocks: more than three resident waves); the shared gate|up one of 192.
#[test]
fn groups_take_the_pipe_at_one_and_two_rows() {
    let _serial = crate::glm5next_fp8_dense::lock_registries_for_test();
    let gpu = MockGpuBackend::new();
    let k = 4096usize;
    let ns = [2816usize, 2816, 2816, 128, 22, 128];
    let keys = register_all(&gpu, &ns.map(|n| (n, k)));
    let outs: Vec<DevicePtr> = (0..6)
        .map(|i| DevicePtr(0x2000_0000 + i * 0x100_0000))
        .collect();
    let a = DevicePtr(0x1000_0000);
    for m in 1..=2usize {
        let members: Vec<_> = (0..6).map(|i| (keys[i], outs[i], ns[i])).collect();
        let start = gpu.launches_snapshot().len();
        assert!(proj_group_registered(&gpu, PIPE, &members, a, m, k, 0).unwrap());
        let l = one(&gpu, start);
        assert_eq!((l.grid, l.shared_mem), ([1091, 1, 1], m as u32 * 8192 + 64));
        for i in 0..6 {
            let b = 1 + 5 * i;
            assert_eq!(ptr_arg(&l.args[b]), keys[i], "member {i}");
            assert_eq!(ptr_arg(&l.args[b + 3]), outs[i], "member {i}");
            assert_eq!(u32_arg(&l.args[b + 4]), ns[i] as u32, "member {i}");
        }
        let start = gpu.launches_snapshot().len();
        assert!(proj_group_registered(&gpu, PIPE, &members[3..5], a, m, k, 0).unwrap());
        assert_eq!(one(&gpu, start).grid, [19, 1, 1], "150 rows");
        assert!(looked_up(&gpu, "w4a16_gemv_pipe", &format!("w4a16_gemv_pipe_m{m}")));
    }
}

/// 2026-10-21: The lever is on by default and `0` switches it off; a value the switch rule cannot
/// read is refused.
#[test]
fn the_pipe_lever_is_on_unless_switched_off() {
    let read = |pairs: &[(&str, &str)]| {
        DenseLevers::from_settings(&settings_from(pairs).expect("settings resolve"))
            .expect("levers resolve")
            .gemv_pipe
    };
    assert!(read(&[]));
    assert!(read(&[("METRALE_GLM_W4A16_GEMV_PIPE", "1")]));
    assert!(!read(&[("METRALE_GLM_W4A16_GEMV_PIPE", "0")]));
    assert!(settings_from(&[("METRALE_GLM_W4A16_GEMV_PIPE", "garbage")]).is_err());
}
