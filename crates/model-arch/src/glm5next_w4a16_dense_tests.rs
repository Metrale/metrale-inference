// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The W4A16 dense tier's decisions (which projection gets which tier, the TP units,
//! the shapes it takes), its registry and dispatch on a recording mock backend, and its refusals.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::*;

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
        assert_eq!(tier_of(n).unwrap(), ProjTier::W4a16, "{n}");
    }
    for n in fp8 {
        assert_eq!(tier_of(n).unwrap(), ProjTier::Fp8, "{n}");
    }
    for n in ["dsa.wq_b", "dsa.weights_proj", "mlp.gate", "kda.q_proj "] {
        let e = tier_of(n).unwrap_err().to_string();
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
    assert!(proj_registered(&gpu, key, a, c, 150, n, k, 7).unwrap());
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
    assert!(!proj_registered(&gpu, bf16, a, c, 1, n, k, 0).unwrap());
    assert!(!proj_registered(&gpu, DevicePtr(0x3000_0000), a, c, 1, n, k, 0).unwrap());
    assert_eq!(gpu.launches_snapshot().len(), n_launches);
    let e = proj_registered(&gpu, key, a, c, 1, n, 256, 0)
        .unwrap_err()
        .to_string();
    assert!(e.contains("launched as [96, 256]"), "{e}");
    assert!(
        !proj(&gpu, key, a, c, 1, n, k, 0).unwrap(),
        "the published tier (declared in tests) declines"
    );
}
