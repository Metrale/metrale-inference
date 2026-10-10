// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Host launcher of `glm5next_hc_pre_norm`: one mHC site's pre-mix (`hc_mix_bf16`
//! then `hc_finish`) and the RMS norm of its collapsed row, in one launch with the bits of the
//! three.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - `glm_hc_pre_norm` writes what `glm_hc_pre` followed by `rms_norm_vanilla` (block
//!   `min(hidden, 1024)`) writes, byte for byte: `y_out`, `post_out`, `comb_out` and
//!   `normed_out` (`examples/glm5next_hc_fused_gate.rs`).
//! - Both grids give every mix the same bits (the per-row and 8-row unfused mix kernels agree,
//!   and the gate compares the fused kernel against whichever the unfused path picks), so the
//!   grid is chosen for speed alone: per row below `HC_PRE_NORM_ROWS_MIN_TOKENS`.
//! - It refuses every shape the kernel's shared memory and per-`hc_mult` instantiations do not
//!   cover, and an FP32 `hc_fn`.
//! - 2026-10-10: `glm_hc_post_rank_sum` (`glm5next_hc_post_sum.cu`) writes what
//!   `bf16_add_rank_sum` followed by `glm_hc_post` writes, byte for byte
//!   (`examples/glm5next_hc_post_sum_gate.rs`).

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::glm5next_mhc::{
    Glm5NextMhcKernels, Glm5NextMhcSiteWeights, HC_MIX_ROWS, MHC_MIX_MAX_TOKENS, collapse_blocks,
    mhc_mix_max_tokens, mix_hc,
};

/// 2026-10-09: `#define GLM_HCF_MAX_H` in `glm5next_mhc_fused.cu`: the widest hidden row the
/// kernel's shared copy of `y` holds.
pub const HC_PRE_NORM_MAX_HIDDEN: u32 = 8192;

/// 2026-10-09: `#define GLM_HCF_MAX_MULT` in `glm5next_mhc_fused.cu`: the kernel instantiates
/// `hc_mult` 1 through 4.
pub const HC_PRE_NORM_MAX_MULT: u32 = 4;

/// 2026-10-09: Why `glm_hc_pre_norm` cannot run a shape, or `None` when it can. The kernel
/// emulates `rms_norm_vanilla`'s `min(hidden, 1024)` threads as whole warps of pairs, so that
/// width must be a multiple of 32, and its tail moves four columns per load, so the row must be
/// a multiple of 4.
pub fn hc_pre_norm_refusal(hidden_size: u32, hc_mult: u32) -> Option<String> {
    if hc_mult == 0 || hc_mult > HC_PRE_NORM_MAX_MULT {
        return Some(format!(
            "hc_mult {hc_mult} is outside 1..={HC_PRE_NORM_MAX_MULT}"
        ));
    }
    if hidden_size == 0 || hidden_size > HC_PRE_NORM_MAX_HIDDEN {
        return Some(format!(
            "hidden {hidden_size} is outside 1..={HC_PRE_NORM_MAX_HIDDEN}"
        ));
    }
    if !hidden_size.is_multiple_of(4) || !hidden_size.min(1024).is_multiple_of(32) {
        return Some(format!(
            "hidden {hidden_size}: needs a multiple of 4 with min(hidden, 1024) a multiple of 32"
        ));
    }
    None
}

/// 2026-10-09: Fewest tokens `glm_hc_pre_norm` sends to the 8-row grid. Measured on GB10
/// (H 4096, hc 4, 90 distinct sites in a graph), us per site per-row vs 8-row: T=16 24.7 vs
/// 38.6, T=32 50.1 vs 64.5, T=64 99.2 vs 83.5, T=128 195.6 vs 166.7. The unfused path switches
/// at `HC_MIX_ROWS_MIN_TOKENS` (32); the fused per-row grid stays ahead up to between 32 and 64.
pub const HC_PRE_NORM_ROWS_MIN_TOKENS: usize = 64;

/// 2026-10-09: `glm_hc_pre` plus the RMS norm of `y_out` with `norm_w` into `normed_out`, in
/// one launch of `glm5next_hc_pre_norm` (or `_rows` from `HC_PRE_NORM_ROWS_MIN_TOKENS` tokens).
/// `norm_eps` is the stream RMS epsilon `glm_hc_pre` takes; `rms_eps` the norm's. Errors on a
/// shape `hc_pre_norm_refusal` names, an FP32 `hc_fn`, a kernel the build lacks, or more tokens
/// than `mhc_mix_max_tokens()`.
#[allow(clippy::too_many_arguments)]
pub fn glm_hc_pre_norm(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextMhcKernels,
    streams: DevicePtr,
    w: &Glm5NextMhcSiteWeights,
    norm_w: DevicePtr,
    y_out: DevicePtr,
    normed_out: DevicePtr,
    post_out: DevicePtr,
    comb_out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    sinkhorn_iters: u32,
    norm_eps: f32,
    hc_eps: f32,
    rms_eps: f32,
    stream: u64,
) -> Result<()> {
    if let Some(why) = hc_pre_norm_refusal(hidden_size, hc_mult) {
        bail!("glm_hc_pre_norm: {why}");
    }
    if !w.hc_fn_bf16 {
        bail!("glm_hc_pre_norm: the site's hc_fn is FP32; the fused kernel reads BF16 only");
    }
    let mix_cap = mhc_mix_max_tokens();
    if num_tokens as usize > mix_cap {
        bail!(
            "glm_hc_pre_norm: {num_tokens} tokens exceeds the {mix_cap}-token `mix` and `done` \
             scratch (floor {MHC_MIX_MAX_TOKENS})"
        );
    }
    let mix_rows = mix_hc(hc_mult as usize) as u32;
    let (kernel, grid_y) = if num_tokens as usize >= HC_PRE_NORM_ROWS_MIN_TOKENS {
        (
            kernels.hc_pre_norm_rows,
            mix_rows.div_ceil(HC_MIX_ROWS as u32),
        )
    } else {
        (kernels.hc_pre_norm, mix_rows)
    };
    if kernel.0 == 0 {
        bail!("glm_hc_pre_norm: the glm5next_mhc_fused kernels are not in this build");
    }
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, grid_y, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(w.hc_fn)
        .arg_ptr(w.hc_scale)
        .arg_ptr(w.hc_base)
        .arg_ptr(w.mix)
        .arg_ptr(w.done)
        .arg_ptr(norm_w)
        .arg_ptr(y_out)
        .arg_ptr(post_out)
        .arg_ptr(comb_out)
        .arg_ptr(normed_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(sinkhorn_iters)
        .arg_f32(norm_eps)
        .arg_f32(hc_eps)
        .arg_f32(rms_eps)
        .launch(stream)
}

/// 2026-10-10: The one-shot all-reduce's sum and `glm_hc_post` in one launch
/// (`glm5next_hc_post_rank_sum`): `block_out` holds this rank's partial and `partials` says where
/// the other ranks' landed (`CommBackend::exchange_partials`). Writes the reduced row back to
/// `block_out` and the post-mix to `out`, both byte-identical to `bf16_add_rank_sum` followed by
/// `glm_hc_post`. Errors when the kernel is not loaded or the rows exceed a peer slot.
#[allow(clippy::too_many_arguments)]
pub fn glm_hc_post_rank_sum(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    block_out: DevicePtr,
    partials: &metrale_comm::PeerPartials,
    residual: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        kernel.0 != 0,
        "glm5next_hc_post_rank_sum is not loaded on this target"
    );
    let elems = num_tokens as usize * hidden_size as usize;
    anyhow::ensure!(
        elems <= partials.slot_elems,
        "{elems} elements exceed the one-shot peer slot of {}",
        partials.slot_elems
    );
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, collapse_blocks(hidden_size), 1])
        .block([256, 1, 1])
        .arg_ptr(block_out)
        .arg_ptr(DevicePtr(partials.recv))
        .arg_u64(partials.slot_elems as u64)
        .arg_i32(partials.world_size as i32)
        .arg_i32(partials.rank as i32)
        .arg_ptr(residual)
        .arg_ptr(post)
        .arg_ptr(comb)
        .arg_ptr(out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

#[cfg(test)]
mod tests {
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
            hc_post: k(7),
            hc_post_rank_sum: k(11),
            hc_head: k(8),
        }
    }

    fn site(bf16: bool) -> Glm5NextMhcSiteWeights {
        Glm5NextMhcSiteWeights {
            hc_fn: DevicePtr(0x100),
            hc_fn_bf16: bf16,
            hc_scale: DevicePtr(0x200),
            hc_base: DevicePtr(0x300),
            mix: DevicePtr(0x400),
            done: DevicePtr(0x480),
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
    #[test]
    fn grid_follows_the_unfused_mix_choice() {
        let gpu = MockGpuBackend::new();
        let k = kernels(true);
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
        let cap = mhc_mix_max_tokens() as u32;
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
            KernelHandle(11),
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
            ]
        );
        assert!(
            glm_hc_post_rank_sum(
                &gpu,
                KernelHandle(0),
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
                KernelHandle(11),
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
}
