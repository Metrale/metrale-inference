// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The W4A16 dense tier's decisions (which projection gets which tier, the TP units,
//! the shapes it takes), its registry and dispatch on a recording mock backend, and its refusals.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::*;

/// 2026-10-09: The plans the dispatch tests run under (48 SMs: GB10).
const OFF: SegPlan = SegPlan {
    mode: SegMode::Off,
    sms: 0,
};
const FUSE: SegPlan = SegPlan {
    mode: SegMode::Fuse,
    sms: 48,
};
const SPLIT: SegPlan = SegPlan {
    mode: SegMode::Split,
    sms: 48,
};

/// 2026-10-09: The KDA q/k/v, f_a, b, g_a, o and the shared expert go W4A16; KDA f_b/g_b and
/// every DSA projection stay FP8; a name without a decision is refused.
#[test]
fn each_projection_has_its_tier_and_unknown_names_are_refused() {
    let w4 = [
        "kda.q_proj",
        "kda.k_proj",
        "kda.v_proj",
        "kda.f_a_proj",
        "kda.b_proj",
        "kda.g_a_proj",
        "kda.o_proj",
        "shared_experts.gate_proj",
        "shared_experts.up_proj",
        "shared_experts.down_proj",
    ];
    let fp8 = [
        "kda.f_b_proj",
        "kda.g_b_proj",
        "dsa.q_a_proj",
        "dsa.q_absorb",
        "dsa.kv_a_proj",
        "dsa.o_absorb",
        "dsa.indexer.wk",
        "dsa.indexer.compress_gate",
    ];
    for n in w4 {
        assert_eq!(tier_of(n, false).unwrap(), ProjTier::W4a16, "{n}");
    }
    for n in fp8 {
        assert_eq!(tier_of(n, false).unwrap(), ProjTier::Fp8, "{n}");
    }
    for n in ["dsa.wq_b", "dsa.weights_proj", "mlp.gate", "kda.q_proj "] {
        let e = tier_of(n, false).unwrap_err().to_string();
        assert!(e.contains("no tier decided"), "{n}: {e}");
    }
}

/// 2026-10-09: The KDA head division's channel unit is the row-tile unit under the tier and 1
/// (head by head, the `declared`/`fp8` split) otherwise.
#[test]
fn the_kda_channel_unit_follows_the_tier() {
    assert_eq!(kda_channel_unit_for(true), W4A16_K_UNIT);
    assert_eq!(kda_channel_unit_for(false), 1);
    assert_eq!(
        kda_channel_unit(),
        1,
        "tests run at the default tier, declared"
    );
}

/// 2026-10-09: The shapes the tier takes at GLM-5.3's TP=3 widths, and the ones it refuses: the
/// head-by-head o_proj K 2688, the BF16 shared split's 688/680, f_b's K 128, empty shapes.
#[test]
fn the_tier_takes_whole_row_tile_units_only() {
    for (n, k) in [
        (2816, 4096),
        (2560, 4096),
        (128, 4096),
        (20, 4096),
        (4096, 2816),
        (4096, 2560),
        (768, 4096),
        (4096, 768),
        (4096, 512),
    ] {
        assert!(shape_ok(n, k), "[{n}, {k}]");
    }
    for (n, k) in [
        (4096, 2688),
        (4096, 688),
        (4096, 680),
        (4096, 640),
        (2816, 128),
        (0, 4096),
        (4096, 0),
    ] {
        assert!(!shape_ok(n, k), "[{n}, {k}]");
    }
}

/// 2026-10-09: Exactly one slot holding the old address is retargeted; none or two are refused,
/// and a refusal changes no slot.
#[test]
fn retarget_moves_exactly_one_slot() {
    let (a, b, key) = (DevicePtr(0x10), DevicePtr(0x20), DevicePtr(0x99));
    let (mut s0, mut s1) = (a, b);
    retarget(&mut [&mut s0, &mut s1], b, key).unwrap();
    assert_eq!((s0, s1), (a, key));

    let e = retarget(&mut [&mut s0, &mut s1], DevicePtr(0x30), key)
        .unwrap_err()
        .to_string();
    assert!(e.contains("no weight field"), "{e}");

    let (mut d0, mut d1) = (a, a);
    let e = retarget(&mut [&mut d0, &mut d1], a, key)
        .unwrap_err()
        .to_string();
    assert!(e.contains("two weight fields"), "{e}");
}

/// 2026-10-09: The kernels the tier needs are looked up when it loads; a missing row-tile entry
/// point is refused there, not at the first launch.
#[test]
fn missing_kernels_are_refused_at_load() {
    let gpu = MockGpuBackend::new();
    gpu.deny_kernel(W4A16_TC_ROWS_MODULE, "w4a16_tc_rows_64");
    let e = Nvfp4QuantKernels::load(&gpu).unwrap_err();
    assert!(format!("{e:#}").contains("w4a16_tc_rows_64"), "{e:#}");
    let gpu = MockGpuBackend::new();
    gpu.deny_kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4");
    assert!(Nvfp4QuantKernels::load(&gpu).is_err());
    assert!(Nvfp4QuantKernels::load(&MockGpuBackend::new()).is_ok());
}

fn u32_arg(a: &MockArg) -> u32 {
    match a {
        MockArg::Bytes(b) => u32::from_le_bytes(b[..4].try_into().unwrap()),
        MockArg::Buffer(p) => panic!("expected a u32, got buffer {p}"),
    }
}

fn ptr_arg(a: &MockArg) -> DevicePtr {
    match a {
        MockArg::Buffer(p) => *p,
        MockArg::Bytes(_) => panic!("expected a buffer"),
    }
}

/// 2026-10-09: A refused K quantizes nothing. A registered weight quantizes once (absmax and
/// quantize), keys on its packed buffer, and runs in 64-row launches over consecutive row
/// ranges of A and C (150 rows: 64 + 64 + 22) with the registered NVFP4 buffers; an unregistered
/// weight (the BF16 address included) declines without a launch; another shape is an error.
#[test]
fn registered_weights_run_in_whole_row_chunks_and_everything_else_declines_or_errors() {
    let _serial = crate::glm5next_fp8_dense::lock_registries_for_test();
    let gpu = MockGpuBackend::new();
    let kernels = Nvfp4QuantKernels::load(&gpu).unwrap();
    let before = registered();
    let (n, k) = (96usize, 512usize);
    let bf16 = gpu.alloc(n * k * 2).unwrap();

    let launches = gpu.launches_snapshot().len();
    let e = register(&gpu, &kernels, bf16, n, 384, "o_proj", 0)
        .unwrap_err()
        .to_string();
    assert!(e.contains("multiple of 256"), "{e}");
    assert_eq!(
        gpu.launches_snapshot().len(),
        launches,
        "a refusal launches nothing"
    );

    let key = register(&gpu, &kernels, bf16, n, k, "q_proj", 0).unwrap();
    assert_ne!(key, bf16);
    assert_eq!(
        gpu.launches_snapshot().len(),
        launches + 2,
        "absmax, then quantize"
    );
    let after = registered();
    assert_eq!(
        (after.0 - before.0, after.1 - before.1, after.2 - before.2),
        (1, n * k / 2 + n * k / 16, n * k * 2)
    );

    let (a, c) = (DevicePtr(0x1000_0000), DevicePtr(0x2000_0000));
    let start = gpu.launches_snapshot().len();
    assert!(proj_registered(&gpu, OFF, key, a, c, 150, n, k, 7).unwrap());
    let l = gpu.launches_snapshot();
    let ours = &l[start..];
    assert_eq!(ours.len(), 3);
    for (i, (l, rows)) in ours.iter().zip([64u32, 64, 22]).enumerate() {
        let done = i * 64;
        assert_eq!(l.grid, [n.div_ceil(64) as u32, 1, 1]);
        assert_eq!(l.stream, 7);
        assert_eq!(
            ptr_arg(&l.args[0]),
            a.offset(done * k * 2),
            "A rows of launch {i}"
        );
        assert_eq!(ptr_arg(&l.args[1]), key, "the registered packed weight");
        assert_eq!(
            ptr_arg(&l.args[4]),
            c.offset(done * n * 2),
            "C rows of launch {i}"
        );
        let ints: Vec<u32> = l.args[5..].iter().map(u32_arg).collect();
        assert_eq!(ints, vec![rows, n as u32, k as u32, k as u32, n as u32]);
    }

    let n_launches = gpu.launches_snapshot().len();
    assert!(!proj_registered(&gpu, OFF, bf16, a, c, 1, n, k, 0).unwrap());
    assert!(!proj_registered(&gpu, OFF, DevicePtr(0x3000_0000), a, c, 1, n, k, 0).unwrap());
    assert_eq!(gpu.launches_snapshot().len(), n_launches);
    let e = proj_registered(&gpu, OFF, key, a, c, 1, n, 256, 0)
        .unwrap_err()
        .to_string();
    assert!(e.contains("launched as [96, 256]"), "{e}");
    assert!(
        !proj(&gpu, key, a, c, 1, n, k, 0).unwrap(),
        "the published tier (declared in tests) declines"
    );
}

/// 2026-10-09: `METRALE_GLM_DSA_W4A16` moves exactly the DSA q_a / absorbed q / absorbed o, is
/// refused without the `w4a16` tier, and takes only 0 or 1.
#[test]
fn the_dsa_absorbed_lever_moves_three_projections_and_needs_the_tier() {
    for n in ["dsa.q_a_proj", "dsa.q_absorb", "dsa.o_absorb"] {
        assert_eq!(tier_of(n, true).unwrap(), ProjTier::W4a16, "{n}");
    }
    for n in [
        "dsa.kv_a_proj",
        "dsa.indexer.wk",
        "dsa.indexer.compress_gate",
        "kda.f_b_proj",
    ] {
        assert_eq!(tier_of(n, true).unwrap(), ProjTier::Fp8, "{n}");
    }
    assert!(tier_of("dsa.wq_b", true).is_err());
    assert_eq!(parse_dsa_absorbed(None, false), Ok(false));
    assert_eq!(parse_dsa_absorbed(Some("0"), true), Ok(false));
    assert_eq!(parse_dsa_absorbed(Some("1"), true), Ok(true));
    assert!(
        parse_dsa_absorbed(Some("1"), false)
            .unwrap_err()
            .contains("w4a16")
    );
    assert!(parse_dsa_absorbed(Some("on"), true).is_err());
}

/// 2026-10-09: `METRALE_GLM_W4A16_SEG` takes 0, fuse or split, and fuse/split only with the tier.
#[test]
fn the_seg_lever_parses_three_values_and_needs_the_tier() {
    use crate::glm5next_w4a16_seg::parse_seg;
    assert_eq!(parse_seg(None, false), Ok(SegMode::Off));
    assert_eq!(parse_seg(Some("0"), false), Ok(SegMode::Off));
    assert_eq!(parse_seg(Some("fuse"), true), Ok(SegMode::Fuse));
    assert_eq!(parse_seg(Some("split"), true), Ok(SegMode::Split));
    assert!(
        parse_seg(Some("split"), false)
            .unwrap_err()
            .contains("w4a16")
    );
    for v in ["1", "on", "Split", ""] {
        assert!(parse_seg(Some(v), true).is_err(), "{v:?}");
    }
}

/// 2026-10-09: Register `[n, k]` weights on `gpu`, returning their keys.
fn register_all(gpu: &MockGpuBackend, shapes: &[(usize, usize)]) -> Vec<DevicePtr> {
    let kernels = Nvfp4QuantKernels::load(gpu).unwrap();
    shapes
        .iter()
        .map(|&(n, k)| {
            let bf16 = gpu.alloc(n * k * 2).unwrap();
            register(gpu, &kernels, bf16, n, k, "test", 0).unwrap()
        })
        .collect()
}

/// 2026-10-09: Under `fuse` and `split` a registered projection of 2+ rows runs on the
/// segmented tiles, one segment, 64-row chunks, grid (tiles, split) with the split of its shape
/// (a 768-wide K-4096 weight: 1 under fuse, 4 under split); one row keeps the GEMV's 7 args.
#[test]
fn seg_modes_move_two_rows_and_up_to_the_segmented_tiles() {
    let _serial = crate::glm5next_fp8_dense::lock_registries_for_test();
    let gpu = MockGpuBackend::new();
    let (n, k) = (768usize, 4096usize);
    let key = register_all(&gpu, &[(n, k)])[0];
    let (a, c) = (DevicePtr(0x1000_0000), DevicePtr(0x2000_0000));
    for (plan, s) in [(FUSE, 1u32), (SPLIT, 4)] {
        let start = gpu.launches_snapshot().len();
        assert!(proj_registered(&gpu, plan, key, a, c, 70, n, k, 5).unwrap());
        let l = gpu.launches_snapshot();
        let ours = &l[start..];
        assert_eq!(ours.len(), 2, "64 + 6 rows");
        for (i, (l, rows)) in ours.iter().zip([64u32, 6]).enumerate() {
            assert_eq!(l.grid, [12, s, 1]);
            assert_eq!(l.args.len(), 19);
            assert_eq!(ptr_arg(&l.args[0]), a.offset(i * 64 * k * 2));
            assert_eq!(ptr_arg(&l.args[1]), key);
            assert_eq!(ptr_arg(&l.args[4]), c.offset(i * 64 * n * 2));
            assert_eq!(u32_arg(&l.args[5]), n as u32);
            assert_eq!(u32_arg(&l.args[10]), 0, "segment 1 empty");
            assert_eq!(u32_arg(&l.args[16]), rows);
        }
        let start = gpu.launches_snapshot().len();
        assert!(proj_registered(&gpu, plan, key, a, c, 1, n, k, 5).unwrap());
        assert_eq!(
            gpu.launches_snapshot()[start..][0].args.len(),
            7,
            "the GEMV"
        );
    }
}

/// 2026-10-09: A group declines without a launch under `off`, at one row, or with a member not
/// registered; otherwise it is one launch per 64 rows over every member (KDA f_a, g_a, b: 2 + 2
/// + 1 tiles, split 4 under `split`), and a member at another shape is an error.
#[test]
fn a_group_is_one_launch_or_declines_whole() {
    let _serial = crate::glm5next_fp8_dense::lock_registries_for_test();
    let gpu = MockGpuBackend::new();
    let k = 4096usize;
    let keys = register_all(&gpu, &[(128, k), (128, k), (22, k)]);
    let outs = [
        DevicePtr(0x2000_0000),
        DevicePtr(0x3000_0000),
        DevicePtr(0x4000_0000),
    ];
    let ns = [128usize, 128, 22];
    let members: Vec<_> = (0..3).map(|i| (keys[i], outs[i], ns[i])).collect();
    let a = DevicePtr(0x1000_0000);
    let n0 = gpu.launches_snapshot().len();
    assert!(!proj_group_registered(&gpu, OFF, &members, a, 16, k, 0).unwrap());
    assert!(!proj_group_registered(&gpu, SPLIT, &members, a, 1, k, 0).unwrap());
    let mut stray = members.clone();
    stray[1].0 = DevicePtr(0x7000_0000);
    assert!(!proj_group_registered(&gpu, SPLIT, &stray, a, 16, k, 0).unwrap());
    assert_eq!(
        gpu.launches_snapshot().len(),
        n0,
        "a decline launches nothing"
    );

    for (plan, s) in [(FUSE, 1u32), (SPLIT, 4)] {
        let start = gpu.launches_snapshot().len();
        assert!(proj_group_registered(&gpu, plan, &members, a, 100, k, 2).unwrap());
        let l = gpu.launches_snapshot();
        let ours = &l[start..];
        assert_eq!(ours.len(), 2, "64 + 36 rows");
        for (j, l) in ours.iter().enumerate() {
            assert_eq!(l.grid, [5, s, 1]);
            assert_eq!(ptr_arg(&l.args[0]), a.offset(j * 64 * k * 2));
            for i in 0..3 {
                assert_eq!(ptr_arg(&l.args[1 + 5 * i]), keys[i]);
                assert_eq!(
                    ptr_arg(&l.args[4 + 5 * i]),
                    outs[i].offset(j * 64 * ns[i] * 2)
                );
                assert_eq!(u32_arg(&l.args[5 + 5 * i]), ns[i] as u32);
            }
        }
    }
    let mut wrong = members.clone();
    wrong[2].2 = 20;
    let e = proj_group_registered(&gpu, SPLIT, &wrong, a, 16, k, 0)
        .unwrap_err()
        .to_string();
    assert!(e.contains("launched as [20, 4096]"), "{e}");
    assert!(
        !proj_group(&gpu, &members, a, 16, k, 0).unwrap(),
        "the published tier (declared in tests) declines"
    );
}
