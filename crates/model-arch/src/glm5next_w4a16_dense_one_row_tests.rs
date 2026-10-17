// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `METRALE_GLM_W4A16_GEMV_SEG`: its parse, and the one-row launches it moves to
//! `w4a16_gemv_seg` (a lone projection, a group of up to six), on a recording mock backend.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

use super::super::*;
use super::{OFF, ptr_arg, register_all, u32_arg};
use crate::glm5next_w4a16_seg::parse_gemv_seg;
use metrale_model_layers::layers::ops::w4a16_gemv_seg_policy;

/// 2026-10-10: The plan the lever publishes on GB10 (48 SMs), seg mode off.
const ONE_ROW: SegPlan = SegPlan {
    one_row: true,
    sms: 48,
    ..OFF
};

/// 2026-10-10: `0`/unset off, `1` on under the tier; `1` without the tier or another value is
/// refused.
#[test]
fn gemv_seg_lever_parses_and_refuses() {
    assert_eq!(parse_gemv_seg(None, false), Ok(false));
    assert_eq!(parse_gemv_seg(Some("0"), true), Ok(false));
    assert_eq!(parse_gemv_seg(Some("1"), true), Ok(true));
    assert!(parse_gemv_seg(Some("1"), false).is_err());
    for bad in ["", "on", "2", "fuse"] {
        assert!(parse_gemv_seg(Some(bad), true).is_err(), "{bad:?} accepted");
    }
}

/// 2026-10-10: `(segment pointers, scale2, output, n)` of launch argument slot `i`, then K.
fn seg_args(
    l: &metrale_gpu_runtime::gpu::mock::MockLaunch,
    i: usize,
) -> (DevicePtr, DevicePtr, u32) {
    let b = 1 + 5 * i;
    (
        ptr_arg(&l.args[b]),
        ptr_arg(&l.args[b + 3]),
        u32_arg(&l.args[b + 4]),
    )
}

/// 2026-10-10: Under the lever one row of a lone projection below K 2048 (the shared down, K 768)
/// is one `w4a16_gemv_seg` launch (one segment, the policy's tile and block cap); without it, the
/// 7-argument `w4a16_gemv`. 2026-10-11: A lone projection from K 2048 up (KDA o, K 2816) keeps
/// `w4a16_gemv` under the lever. Two rows are untouched by the lever.
#[test]
fn one_row_takes_the_segmented_gemv_only_under_the_lever() {
    let _serial = crate::glm5next_fp8_dense::lock_registries_for_test();
    let gpu = MockGpuBackend::new();
    let (n, k) = (4096usize, 768usize);
    let keys = register_all(&gpu, &[(n, k), (4096, 2816)]);
    let key = keys[0];
    let (a, c) = (DevicePtr(0x1000_0000), DevicePtr(0x2000_0000));
    let start = gpu.launches_snapshot().len();
    assert!(proj_registered(&gpu, ONE_ROW, key, a, c, 1, n, k, 7).unwrap());
    let l = gpu.launches_snapshot()[start..].to_vec();
    assert_eq!(l.len(), 1);
    let (r, cap) = w4a16_gemv_seg_policy(k as u32, 48);
    let tiles = (n as u32).div_ceil(r);
    assert_eq!(l[0].grid, [tiles.div_ceil(8).min(cap), 1, 1]);
    assert_eq!((l[0].block, l[0].stream), ([256, 1, 1], 7));
    assert_eq!(l[0].args.len(), 32);
    assert_eq!(ptr_arg(&l[0].args[0]), a);
    assert_eq!(seg_args(&l[0], 0), (key, c, n as u32));
    assert_eq!(seg_args(&l[0], 1), (DevicePtr::NULL, DevicePtr::NULL, 0));
    assert_eq!(u32_arg(&l[0].args[31]), k as u32);
    let entry = format!("w4a16_gemv_seg_r{r}");
    assert!(
        gpu.kernel_lookups_snapshot()
            .iter()
            .any(|(m, f)| m == "w4a16_gemv_seg" && *f == entry)
    );

    let start = gpu.launches_snapshot().len();
    assert!(proj_registered(&gpu, OFF, key, a, c, 1, n, k, 7).unwrap());
    assert_eq!(gpu.launches_snapshot()[start].args.len(), 7, "the GEMV");
    let start = gpu.launches_snapshot().len();
    assert!(proj_registered(&gpu, ONE_ROW, keys[1], a, c, 1, 4096, 2816, 7).unwrap());
    assert_eq!(
        gpu.launches_snapshot()[start].args.len(),
        7,
        "KDA o keeps the GEMV"
    );
    let start = gpu.launches_snapshot().len();
    assert!(proj_registered(&gpu, ONE_ROW, key, a, c, 2, n, k, 7).unwrap());
    let off2 = gpu.launches_snapshot()[start..].to_vec();
    let start = gpu.launches_snapshot().len();
    assert!(proj_registered(&gpu, OFF, key, a, c, 2, n, k, 7).unwrap());
    let l2 = gpu.launches_snapshot()[start..].to_vec();
    assert_eq!(off2.len(), l2.len());
    for (x, y) in off2.iter().zip(&l2) {
        assert_eq!(
            (x.func, x.grid, &x.args),
            (y.func, y.grid, &y.args),
            "2 rows unchanged"
        );
    }
}

/// 2026-10-10: At one row under the lever a group of six (the KDA q, k, v, f_a, b, g_a) is one
/// launch carrying every member in order; a group with an unregistered member declines without
/// a launch; without the lever one row declines; seven members are refused.
#[test]
fn one_row_groups_are_one_launch_of_up_to_six() {
    let _serial = crate::glm5next_fp8_dense::lock_registries_for_test();
    let gpu = MockGpuBackend::new();
    let k = 4096usize;
    let ns = [2816usize, 2816, 2816, 128, 22, 128];
    let keys = register_all(&gpu, &ns.map(|n| (n, k)));
    let outs: Vec<DevicePtr> = (0..6)
        .map(|i| DevicePtr(0x2000_0000 + i * 0x100_0000))
        .collect();
    let members: Vec<_> = (0..6).map(|i| (keys[i], outs[i], ns[i])).collect();
    let a = DevicePtr(0x1000_0000);

    let start = gpu.launches_snapshot().len();
    assert!(!proj_group_registered(&gpu, OFF, &members, a, 1, k, 0).unwrap());
    let mut stray = members.clone();
    stray[4].0 = DevicePtr(0x7000_0000);
    assert!(!proj_group_registered(&gpu, ONE_ROW, &stray, a, 1, k, 0).unwrap());
    assert_eq!(
        gpu.launches_snapshot().len(),
        start,
        "a decline launches nothing"
    );

    assert!(proj_group_registered(&gpu, ONE_ROW, &members, a, 1, k, 4).unwrap());
    let l = gpu.launches_snapshot()[start..].to_vec();
    assert_eq!(l.len(), 1);
    for i in 0..6 {
        assert_eq!(
            seg_args(&l[0], i),
            (keys[i], outs[i], ns[i] as u32),
            "member {i}"
        );
    }
    let seven: Vec<_> = members.iter().chain(&members[..1]).copied().collect();
    assert!(proj_group_registered(&gpu, ONE_ROW, &seven, a, 1, k, 0).is_err());
    assert_eq!(gpu.launches_snapshot().len(), start + 1);
}

/// 2026-10-11: The lever with GEMV rows 4 on GB10.
const ROWS4: SegPlan = SegPlan {
    gemv_rows: 4,
    ..ONE_ROW
};

/// 2026-10-11: Under the lever, a GEMV-tier launch of 2..=4 rows of a lone projection below
/// K 2048 (the shared down, K 768) is one `w4a16_gemv_seg_m<M>` launch (M staged rows of shared
/// memory); without the lever the same rows take `w4a16_gemv_batch<M>` (7 or 8 arguments), and so
/// does a lone K 2816 projection (KDA o) under it; 5 rows (no M-row point) keep the batch tier.
#[test]
fn gemv_tier_rows_take_the_m_row_point_under_the_lever() {
    let _serial = crate::glm5next_fp8_dense::lock_registries_for_test();
    let gpu = MockGpuBackend::new();
    let (n, k) = (4096usize, 768usize);
    let keys = register_all(&gpu, &[(n, k), (4096, 2816)]);
    let key = keys[0];
    let (a, c) = (DevicePtr(0x1000_0000), DevicePtr(0x2000_0000));
    for m in 2..=4usize {
        let start = gpu.launches_snapshot().len();
        assert!(proj_registered(&gpu, ROWS4, key, a, c, m, n, k, 3).unwrap());
        let l = gpu.launches_snapshot()[start..].to_vec();
        assert_eq!(l.len(), 1, "{m} rows");
        assert_eq!(l[0].args.len(), 32, "{m} rows: the segmented GEMV");
        assert_eq!(l[0].shared_mem, 4096 * m as u32 + 64);
        assert_eq!(seg_args(&l[0], 0), (key, c, n as u32));
        let name = format!("w4a16_gemv_seg_m{m}");
        assert!(
            gpu.kernel_lookups_snapshot()
                .iter()
                .any(|(md, f)| md == "w4a16_gemv_seg" && *f == name)
        );
        let start = gpu.launches_snapshot().len();
        let off = SegPlan {
            one_row: false,
            ..ROWS4
        };
        assert!(proj_registered(&gpu, off, key, a, c, m, n, k, 3).unwrap());
        assert!(
            gpu.launches_snapshot()[start].args.len() <= 8,
            "{m} rows: the batch tier"
        );
        let start = gpu.launches_snapshot().len();
        assert!(proj_registered(&gpu, ROWS4, keys[1], a, c, m, 4096, 2816, 3).unwrap());
        assert!(
            gpu.launches_snapshot()[start].args.len() <= 8,
            "{m} rows: KDA o keeps the batch tier"
        );
    }
    let plan8 = SegPlan {
        gemv_rows: 8,
        ..ONE_ROW
    };
    let start = gpu.launches_snapshot().len();
    assert!(proj_registered(&gpu, plan8, key, a, c, 5, n, k, 3).unwrap());
    assert_eq!(
        gpu.launches_snapshot()[start].args.len(),
        8,
        "5 rows: w4a16_gemv_batch5"
    );
}

/// 2026-10-11: A six-member group at 3 GEMV-tier rows is one M-row launch with every member;
/// without GEMV rows (no tier at 3 rows) or without the lever it declines without a launch, and
/// `seg_gemv_group` says which.
#[test]
fn six_member_groups_run_at_gemv_tier_rows() {
    let _serial = crate::glm5next_fp8_dense::lock_registries_for_test();
    let gpu = MockGpuBackend::new();
    let k = 4096usize;
    let ns = [2816usize, 2816, 2816, 128, 22, 128];
    let keys = register_all(&gpu, &ns.map(|n| (n, k)));
    let outs: Vec<DevicePtr> = (0..6)
        .map(|i| DevicePtr(0x2000_0000 + i * 0x100_0000))
        .collect();
    let members: Vec<_> = (0..6).map(|i| (keys[i], outs[i], ns[i])).collect();
    let a = DevicePtr(0x1000_0000);
    let start = gpu.launches_snapshot().len();
    for plan in [
        ONE_ROW,
        SegPlan {
            one_row: false,
            ..ROWS4
        },
    ] {
        assert!(!seg_gemv_group(plan, 3).unwrap());
        assert!(!proj_group_registered(&gpu, plan, &members, a, 3, k, 0).unwrap());
    }
    assert_eq!(
        gpu.launches_snapshot().len(),
        start,
        "declines launch nothing"
    );
    assert!(seg_gemv_group(ROWS4, 3).unwrap());
    assert!(!seg_gemv_group(ROWS4, 5).unwrap());
    assert!(proj_group_registered(&gpu, ROWS4, &members, a, 3, k, 0).unwrap());
    let l = gpu.launches_snapshot()[start..].to_vec();
    assert_eq!(l.len(), 1);
    assert_eq!(l[0].shared_mem, 3 * 8192 + 64);
    for i in 0..6 {
        assert_eq!(
            seg_args(&l[0], i),
            (keys[i], outs[i], ns[i] as u32),
            "member {i}"
        );
    }
}
