// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Shape tests for the GLM mHC launchers (`glm5next_mhc.rs`).

use super::*;

/// 2026-09-25: `mix_hc` is the pre, post and comb row counts added, for `hc_mult` 1 to 8.
#[test]
fn mix_hc_splits_into_pre_post_and_comb() {
    for hc in 1usize..=8 {
        assert_eq!(mix_hc(hc), hc + hc + hc * hc, "pre + post + comb rows");
    }
    assert_eq!(mix_hc(2), 8);
}

/// 2026-09-25: `GLM_HC_MAX_MIX` in `glm5next_mhc.cu` is 24, which is `mix_hc(4)`; `hc_mult` 5
/// would exceed it.
#[test]
fn the_kernels_mix_bound_is_hc_mult_four() {
    assert_eq!(mix_hc(4), 24, "GLM_HC_MAX_MIX in glm5next_mhc.cu");
    assert!(mix_hc(5) > 24, "hc_mult 5 would exceed the kernel's bound");
}

/// 2026-09-25: At the shipped `PREFILL_ROWS` (16) the bound is the 256-token floor.
/// 2026-10-17: A wider prefill width raises it to that width, and nothing lowers it.
#[test]
fn the_mix_bound_never_goes_below_the_shipped_floor() {
    let floor = MHC_MIX_MAX_TOKENS;
    assert_eq!(
        mhc_mix_max_tokens(crate::glm5next_layer::PREFILL_ROWS),
        floor
    );
    assert_eq!(mhc_mix_max_tokens(1), floor);
    assert_eq!(mhc_mix_max_tokens(floor), floor);
    assert_eq!(mhc_mix_max_tokens(floor + 1), floor + 1);
}

/// 2026-09-25: Total `mix` scratch over 90 sites (45 text layers, 2 sites each) at
/// `hc_mult` 4, the config fixture's values.
#[test]
fn the_mix_scratch_costs_megabytes_not_gigabytes() {
    let total = |tokens: usize| tokens * mix_hc(4) * 4 * 90;
    assert_eq!(
        total(256),
        2_211_840,
        "2.2 MB for the whole model at the floor"
    );
    assert_eq!(total(1024), 8_847_360, "8.8 MB at a 1024-row sub-chunk");
}

/// 2026-10-09: `glm_hc_pre` sends a BF16 site to the 8-row mix kernel from
/// `HC_MIX_ROWS_MIN_TOKENS` tokens (grid y = ceil(24 / 8)), and below it to the per-row one
/// (grid y = 24); an FP32 site always takes the per-row FP32 kernel.
#[test]
fn wide_token_counts_take_the_grouped_mix_kernel() {
    use metrale_gpu_runtime::gpu::DevicePtr;
    use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
    let gpu = MockGpuBackend::new();
    let kernels = test_kernels();
    let site = |bf16| Glm5NextMhcSiteWeights {
        hc_fn: DevicePtr(0x100),
        hc_fn_bf16: bf16,
        hc_scale: DevicePtr(0x200),
        hc_base: DevicePtr(0x300),
        mix: DevicePtr(0x400),
        done: DevicePtr(0x480),
        mix_tokens: crate::glm5next_mhc::MHC_MIX_MAX_TOKENS,
    };
    let first_mix = |t: u32, bf16: bool| {
        let from = gpu.launch_count();
        let p = DevicePtr(0x500);
        glm_hc_pre(
            &gpu,
            &kernels,
            p,
            &site(bf16),
            p,
            p,
            p,
            t,
            4096,
            4,
            20,
            1e-6,
            1e-6,
            0,
        )
        .unwrap();
        let l = gpu.launches_snapshot()[from].clone();
        (l.func, l.grid[1])
    };
    let min = HC_MIX_ROWS_MIN_TOKENS as u32;
    assert_eq!(first_mix(min, true), (5, 3));
    assert_eq!(first_mix(128, true), (5, 3));
    assert_eq!(first_mix(min - 1, true), (4, 24));
    assert_eq!(first_mix(1, true), (4, 24));
    assert_eq!(first_mix(128, false), (3, 24));
}

/// 2026-10-21: Every kernel present under a distinct mock handle, at the declared levers (moved
/// here from `wide_token_counts_take_the_grouped_mix_kernel`).
pub(crate) fn test_kernels() -> Glm5NextMhcKernels {
    let k = |h| KernelHandle(h);
    Glm5NextMhcKernels {
        hc_expand: k(1),
        hc_pre: k(2),
        hc_mix: k(3),
        hc_mix_bf16: k(4),
        hc_mix_bf16_rows: k(5),
        hc_finish: k(6),
        hc_pre_norm: k(9),
        hc_pre_norm_rows: k(10),
        hc_pre_norm_pf: k(13),
        hc_pre_norm_sk: k(14),
        hc_pre_norm_tok: k(15),
        hc_pre_norm_tile: k(17),
        hc_pre_norm_c1: k(19),
        hc_post: k(7),
        hc_post_rank_sum: k(11),
        hc_post_rank_sum_vec8: k(18),
        hc_post_pre_norm: k(12),
        hc_post_pre_norm_tok: k(16),
        hc_head: k(8),
        levers: crate::glm5next_mhc_levers::MhcLevers::declared(),
    }
}
