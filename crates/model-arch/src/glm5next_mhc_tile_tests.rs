// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Unit tests of `glm5next_mhc_tile`'s selection, lever parsing and launch shape.

use super::*;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};

fn site() -> Glm5NextMhcSiteWeights {
    Glm5NextMhcSiteWeights {
        hc_fn: DevicePtr(0x100),
        hc_fn_bf16: true,
        hc_scale: DevicePtr(0x200),
        hc_base: DevicePtr(0x300),
        mix: DevicePtr(0x400),
        done: DevicePtr(0x500),
    }
}

/// 2026-10-15: Only `1` turns the lever on; a typo must not silently keep it off.
#[test]
fn the_lever_takes_one_or_zero_and_refuses_anything_else() {
    assert!(!parse_pre_tile(None).unwrap());
    assert!(!parse_pre_tile(Some("0")).unwrap());
    assert!(parse_pre_tile(Some("1")).unwrap());
    for bad in ["", "true", "on", "2", " 1"] {
        assert!(
            parse_pre_tile(Some(bad)).is_err(),
            "{bad:?} must be refused"
        );
    }
}

/// 2026-10-15: `GlmHctLayout<1>` at four stages: the ring (4 x (24 x 256 x 2 + 4 x 256 x 4))
/// outgrows the trees (25 x 256 x 4) and the tail (8192 x 2), then 4 x 24 mixes, 4 pre weights
/// and 33 warp sums, all FP32. The kernel traps on less.
#[test]
fn the_shared_memory_is_the_kernel_layouts_sum() {
    assert_eq!(tile_smem_bytes(), 65_536 + 384 + 16 + 132);
    assert!(tile_smem_bytes() <= 99 * 1024, "one block must fit an SM");
}

/// 2026-10-15: Every condition of the module invariants is needed; dropping any one falls back.
#[test]
fn the_tile_kernel_needs_every_condition() {
    let gpu = MockGpuBackend::new();
    let k = Glm5NextMhcKernels::resolve(&gpu).expect("mock resolves every kernel");
    let (s, w) = (DevicePtr(0x1000), DevicePtr(0x2000));
    let min = HC_PRE_NORM_ROWS_MIN_TOKENS as u32;
    let pick = |k: &Glm5NextMhcKernels, t, h, hc, s, w, on| {
        tile_kernel(k, t, h, hc, s, w, on).map(|x| x.0)
    };
    assert_eq!(
        pick(&k, min, 4096, 4, s, w, true),
        Some(k.hc_pre_norm_tile.0)
    );
    assert_eq!(
        pick(&k, 2736, 8192, 4, s, w, true),
        Some(k.hc_pre_norm_tile.0)
    );
    assert_eq!(pick(&k, 2736, 4096, 4, s, w, false), None, "lever off");
    assert_eq!(
        pick(&k, min - 1, 4096, 4, s, w, true),
        None,
        "too few tokens"
    );
    assert_eq!(pick(&k, 2736, 4096, 2, s, w, true), None, "hc 2");
    assert_eq!(
        pick(&k, 2736, 4064, 4, s, w, true),
        None,
        "H not a multiple of 64"
    );
    assert_eq!(
        pick(&k, 2736, 8256, 4, s, w, true),
        None,
        "H past the tail buffer"
    );
    assert_eq!(
        pick(&k, 2736, 4096, 4, DevicePtr(0x1008), w, true),
        None,
        "streams at 8 B"
    );
    assert_eq!(
        pick(&k, 2736, 4096, 4, s, DevicePtr(0x2008), true),
        None,
        "hc_fn at 8 B"
    );
    let mut absent = k;
    absent.hc_pre_norm_tile = KernelHandle(0);
    assert_eq!(
        pick(&absent, 2736, 4096, 4, s, w, true),
        None,
        "not in the build"
    );
}

/// 2026-10-15: tok4's arguments in tok4's order on ceil(T / 4) blocks of 256 threads with the
/// layout's shared memory; a dropped or swapped pointer would read the wrong buffer silently.
#[test]
fn the_launch_passes_tok4s_arguments_on_the_tile_grid() {
    let gpu = MockGpuBackend::new();
    let k = Glm5NextMhcKernels::resolve(&gpu).expect("mock resolves every kernel");
    let outs = [
        DevicePtr(0x3000),
        DevicePtr(0x5000),
        DevicePtr(0x6000),
        DevicePtr(0x4000),
    ];
    launch_tile(
        &gpu,
        k.hc_pre_norm_tile,
        DevicePtr(0x1000),
        &site(),
        DevicePtr(0x2000),
        outs,
        684,
        [4096, 4, 20],
        [1e-6, 1e-6, 1e-5],
        7,
    )
    .unwrap();
    let l = gpu.launches_snapshot().last().cloned().expect("one launch");
    assert_eq!(l.func, k.hc_pre_norm_tile.0);
    assert_eq!(l.grid, [171, 1, 1]);
    assert_eq!(l.block, [256, 1, 1]);
    assert_eq!(l.shared_mem, tile_smem_bytes());
    let ptrs = [
        0x1000u64, 0x100, 0x200, 0x300, 0x400, 0x500, 0x2000, 0x3000, 0x5000, 0x6000, 0x4000,
    ];
    for (i, p) in ptrs.iter().enumerate() {
        assert_eq!(
            l.args[i],
            MockArg::Buffer(DevicePtr(*p)),
            "pointer argument {i}"
        );
    }
    let u32s = |v: u32| MockArg::Bytes(v.to_le_bytes().to_vec());
    let f32s = |v: f32| MockArg::Bytes(v.to_le_bytes().to_vec());
    assert_eq!(l.args[11], u32s(4096));
    assert_eq!(l.args[12], u32s(4));
    assert_eq!(l.args[13], u32s(20));
    assert_eq!(l.args[14], f32s(1e-6));
    assert_eq!(l.args[16], f32s(1e-5));
    assert_eq!(l.args[17], u32s(684), "the token count closes the list");
    assert_eq!(l.args.len(), 18);
}
