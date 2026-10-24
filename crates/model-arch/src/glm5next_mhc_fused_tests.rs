// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Tests for the fused mHC launchers (`glm5next_mhc_fused.rs`).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::*;
use metrale_gpu_runtime::gpu::KernelHandle;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

fn kernels(fused: bool) -> Glm5NextMhcKernels {
    let k = |h| KernelHandle(h);
    Glm5NextMhcKernels {
        hc_expand: k(1),
        hc_pre: k(2),
        hc_mix: k(3),
        hc_mix_bf16: k(4),
        hc_mix_bf16_rows: k(5),
        hc_finish: k(6),
        hc_pre_norm: k(if fused { 9 } else { 0 }),
        hc_pre_norm_rows: k(if fused { 10 } else { 0 }),
        hc_pre_norm_pf: k(if fused { 13 } else { 0 }),
        hc_pre_norm_sk: k(if fused { 14 } else { 0 }),
        hc_pre_norm_tok: k(if fused { 15 } else { 0 }),
        hc_pre_norm_tile: k(if fused { 17 } else { 0 }),
        hc_pre_norm_c1: k(if fused { 19 } else { 0 }),
        hc_post: k(7),
        hc_post_rank_sum: k(11),
        hc_post_rank_sum_vec8: k(18),
        hc_post_pre_norm: k(12),
        hc_post_pre_norm_tok: k(if fused { 16 } else { 0 }),
        hc_post_pre_norm_c1: k(0),
        hc_head: k(8),
        levers: crate::glm5next_mhc_levers::MhcLevers::declared(),
    }
}

/// 2026-10-11: The prefetching kernel is taken only when asked, at GLM-5.3's shape, below the
/// 8-row threshold and when the build has it; its grid is the per-row kernel's.
#[test]
fn the_prefetching_pre_norm_needs_the_lever_the_shape_and_the_kernel() {
    // KernelHandle has no PartialEq; compare the raw handles.
    let pick = |k: &Glm5NextMhcKernels, rows: u32, h: u32, hc: u32, lever: bool| {
        let (kh, grid) = pre_norm_kernel(k, rows, h, hc, lever);
        (kh.0, grid)
    };
    let k = kernels(true);
    let mix = mix_hc(4) as u32;
    assert_eq!(pick(&k, 1, 4096, 4, true), (k.hc_pre_norm_pf.0, mix));
    assert_eq!(pick(&k, 16, 4096, 4, true), (k.hc_pre_norm_pf.0, mix));
    assert_eq!(pick(&k, 1, 4096, 4, false), (k.hc_pre_norm.0, mix));
    assert_eq!(pick(&k, 1, 5120, 4, true), (k.hc_pre_norm.0, mix));
    assert_eq!(pick(&k, 1, 4096, 3, true).0, k.hc_pre_norm.0);
    let rows = HC_PRE_NORM_ROWS_MIN_TOKENS as u32;
    assert_eq!(pick(&k, rows, 4096, 4, true).0, k.hc_pre_norm_rows.0);
    let mut absent = kernels(true);
    absent.hc_pre_norm_pf = KernelHandle(0);
    assert_eq!(pick(&absent, 1, 4096, 4, true).0, absent.hc_pre_norm.0);
}

/// 2026-10-11: The split-k kernel needs the lever, fewer than 64 tokens, GLM-5.3's shape, the
/// kernel in the build and a `mix` scratch that holds 64 x 25 floats per token.
#[test]
fn the_split_k_pre_norm_needs_the_lever_the_shape_the_kernel_and_the_scratch() {
    let k = kernels(true);
    let got = |k: &Glm5NextMhcKernels, t: u32, h: u32, hc: u32, on: bool, cap: usize| {
        splitk_kernel(k, t, h, hc, on, cap).map(|(kh, g)| (kh.0, g))
    };
    assert_eq!(
        got(&k, 1, 4096, 4, true, 256),
        Some((k.hc_pre_norm_sk.0, 64))
    );
    assert_eq!(
        got(&k, 3, 4096, 4, true, 256),
        Some((k.hc_pre_norm_sk.0, 64))
    );
    // 4 tokens x 64 x 25 = 6400 floats > 256 x 24 = 6144: the scratch is too small.
    assert_eq!(got(&k, 4, 4096, 4, true, 256), None);
    assert_eq!(
        got(&k, 4, 4096, 4, true, 2048),
        Some((k.hc_pre_norm_sk.0, 64))
    );
    assert_eq!(got(&k, 1, 4096, 4, false, 2048), None);
    assert_eq!(got(&k, 64, 4096, 4, true, 2048), None);
    assert_eq!(got(&k, 1, 5120, 4, true, 2048), None);
    assert_eq!(got(&k, 1, 4096, 3, true, 2048), None);
    let mut absent = kernels(true);
    absent.hc_pre_norm_sk = KernelHandle(0);
    assert_eq!(got(&absent, 1, 4096, 4, true, 2048), None);
}

fn site(bf16: bool) -> Glm5NextMhcSiteWeights {
    Glm5NextMhcSiteWeights {
        hc_fn: DevicePtr(0x100),
        hc_fn_bf16: bf16,
        hc_scale: DevicePtr(0x200),
        hc_base: DevicePtr(0x300),
        mix: DevicePtr(0x400),
        done: DevicePtr(0x480),
        mix_tokens: crate::glm5next_mhc::MHC_MIX_MAX_TOKENS,
    }
}

fn launch(
    gpu: &MockGpuBackend,
    k: &Glm5NextMhcKernels,
    w: &Glm5NextMhcSiteWeights,
    t: u32,
    h: u32,
    hc: u32,
) -> Result<()> {
    let p = DevicePtr(0x500);
    glm_hc_pre_norm(
        gpu, k, p, w, p, p, p, p, p, t, h, hc, 20, 1e-6, 1e-6, 1e-6, 0,
    )
}

/// 2026-10-09: The fused launch takes the 8-row grid from `HC_PRE_NORM_ROWS_MIN_TOKENS`
/// tokens and the per-row grid below, one launch either way.
/// 2026-10-21: A decode step (1 or 2 tokens) takes `glm5next_hc_pre_norm_c1` on the per-row
/// grid unless `METRALE_GLM_MHC_PRE_C1=0`.
#[test]
fn grid_follows_the_unfused_mix_choice() {
    let gpu = MockGpuBackend::new();
    let k = kernels(true);
    let mut off = kernels(true);
    off.levers.pre_c1 = false;
    off.levers.pre_tile = false; // 2026-10-22: default on; pin the grids the tile takes over.
    let first_with = |k: &Glm5NextMhcKernels, t: u32| {
        let from = gpu.launch_count();
        launch(&gpu, k, &site(true), t, 4096, 4).unwrap();
        assert_eq!(gpu.launch_count(), from + 1, "one launch per site");
        let l = gpu.launches_snapshot()[from].clone();
        (l.func, l.grid[0], l.grid[1])
    };
    assert_eq!(first_with(&k, 1), (19, 1, 24));
    assert_eq!(first_with(&k, 2), (19, 2, 24));
    assert_eq!(first_with(&k, 3), (9, 3, 24));
    assert_eq!(first_with(&off, 1), (9, 1, 24));
    assert_eq!(first_with(&off, 2), (9, 2, 24));
    let mut absent = kernels(true);
    absent.hc_pre_norm_c1 = KernelHandle(0);
    assert_eq!(first_with(&absent, 2), (9, 2, 24));
    let k = off;
    let first = |t: u32| {
        let from = gpu.launch_count();
        launch(&gpu, &k, &site(true), t, 4096, 4).unwrap();
        assert_eq!(gpu.launch_count(), from + 1, "one launch per site");
        let l = gpu.launches_snapshot()[from].clone();
        (l.func, l.grid[0], l.grid[1])
    };
    let min = HC_PRE_NORM_ROWS_MIN_TOKENS as u32;
    assert_eq!(first(1), (9, 1, 24));
    assert_eq!(first(16), (9, 16, 24));
    assert_eq!(first(min - 1), (9, min - 1, 24));
    assert_eq!(first(min), (10, min, 3));
    assert_eq!(first(128), (10, 128, 3));
}

/// 2026-10-09: Every shape the kernel cannot run is refused before a launch: FP32 `hc_fn`,
/// a missing kernel, `hc_mult` past 4, a row past the shared `y` copy or one the emulated
/// norm cannot tile, and more tokens than the scratch holds.
#[test]
fn unsupported_shapes_fail_before_launching() {
    let gpu = MockGpuBackend::new();
    let on = kernels(true);
    let cap = MHC_MIX_MAX_TOKENS as u32;
    let cases: [(&Glm5NextMhcKernels, bool, u32, u32, u32); 7] = [
        (&on, false, 1, 4096, 4),
        (&kernels(false), true, 1, 4096, 4),
        (&on, true, 1, 4096, 5),
        (&on, true, 1, 8194, 4),
        (&on, true, 1, 4098, 4),
        (&on, true, 1, 1000, 4),
        (&on, true, cap + 1, 4096, 4),
    ];
    for (k, bf16, t, h, hc) in cases {
        let before = gpu.launch_count();
        assert!(
            launch(&gpu, k, &site(bf16), t, h, hc).is_err(),
            "T={t} H={h} hc={hc} bf16={bf16}"
        );
        assert_eq!(gpu.launch_count(), before, "refused before launching");
    }
    assert!(launch(&gpu, &on, &site(true), cap, 8192, 4).is_ok());
    assert!(launch(&gpu, &on, &site(true), 1, 256, 1).is_ok());
}

/// 2026-10-10: `glm_hc_post_rank_sum` passes its arguments in the kernel's parameter order
/// (`glm5next_hc_post_sum.cu`: block_out, peers, slot_elems, n_ranks, my_rank, residual,
/// post, comb, out, H, hc) on `hc_post`'s grid, and refuses a missing kernel or rows past a
/// peer slot.
#[test]
fn post_rank_sum_launches_in_the_kernels_argument_order() {
    use metrale_gpu_runtime::gpu::mock::MockArg;
    let gpu = MockGpuBackend::new();
    let p = metrale_comm::PeerPartials {
        recv: 0x9000,
        slot_elems: 8192,
        world_size: 3,
        rank: 2,
    };
    let (x, res, post, comb) = (
        DevicePtr(0x10),
        DevicePtr(0x20),
        DevicePtr(0x30),
        DevicePtr(0x40),
    );
    glm_hc_post_rank_sum(
        &gpu,
        &kernels(true),
        x,
        &p,
        res,
        post,
        comb,
        res,
        2,
        4096,
        4,
        7,
    )
    .unwrap();
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 1);
    assert_eq!(
        (l[0].func, l[0].grid, l[0].block, l[0].stream),
        (11, [2, 16, 1], [256, 1, 1], 7)
    );
    let b = |v: &[u8]| MockArg::Bytes(v.to_vec());
    assert_eq!(
        l[0].args,
        vec![
            MockArg::Buffer(x),
            MockArg::Buffer(DevicePtr(0x9000)),
            b(&8192u64.to_le_bytes()),
            b(&3i32.to_le_bytes()),
            b(&2i32.to_le_bytes()),
            MockArg::Buffer(res),
            MockArg::Buffer(post),
            MockArg::Buffer(comb),
            MockArg::Buffer(res),
            b(&4096u32.to_le_bytes()),
            b(&4u32.to_le_bytes()),
            b(&2u32.to_le_bytes()),
        ]
    );
    assert!(
        glm_hc_post_rank_sum(
            &gpu,
            &Glm5NextMhcKernels {
                hc_post_rank_sum: KernelHandle(0),
                ..kernels(true)
            },
            x,
            &p,
            res,
            post,
            comb,
            res,
            1,
            4096,
            4,
            0
        )
        .is_err()
    );
    assert!(
        glm_hc_post_rank_sum(
            &gpu,
            &kernels(true),
            x,
            &p,
            res,
            post,
            comb,
            res,
            3,
            4096,
            4,
            0
        )
        .is_err(),
        "3 x 4096 elements exceed an 8192-element slot"
    );
    assert_eq!(
        gpu.launches_snapshot().len(),
        1,
        "a refused launch enqueues nothing"
    );
}
