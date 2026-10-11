// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Host launcher of `glm5next_hc_post_pre_norm`: one token's mHC post-mix of the
//! previous site (with the one-shot all-reduce's sum in front when the partials are given) and
//! the next site's pre-norm, in one launch with the bits of `bf16_add_rank_sum` (or none),
//! `glm5next_hc_post` and `glm5next_hc_pre_norm`.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - The post-mixed streams go to `streams_out`, never in place: every block of the kernel reads
//!   the streams it post-mixes. `post`/`comb` are read (the previous site's) and written (the
//!   next site's) in place: every block reads them before the tail writes.
//! - The summed block output is not written back to `own`; the caller must not read it after.
//! - It refuses every shape the kernel does not cover rather than launching a kernel that would
//!   write nothing.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::glm5next_mhc::{Glm5NextMhcKernels, Glm5NextMhcSiteWeights, mix_hc};

/// 2026-10-10: `#define GLM_HCF_POST_FOLD_MAX_H` in `glm5next_mhc_post_fold.cu`: each thread holds
/// `hidden / 256` columns of every stream in registers.
pub const POST_FOLD_MAX_HIDDEN: u32 = 4096;

/// 2026-10-10: The buffers of one folded launch.
#[derive(Clone, Copy, Debug)]
pub struct PostFoldIo {
    /// 2026-10-10: This rank's BF16 block output (the partial when `partials` is given, else
    /// the already-reduced output).
    pub own: DevicePtr,
    /// 2026-10-10: The FP32 streams before the post-mix, and where the post-mixed ones go.
    pub residual: DevicePtr,
    pub streams_out: DevicePtr,
    /// 2026-10-10: The previous site's post/comb on entry; the next site's on return.
    pub post: DevicePtr,
    pub comb: DevicePtr,
    /// 2026-10-10: The next site's norm weight, collapsed row and normed row.
    pub norm_w: DevicePtr,
    pub y_out: DevicePtr,
    pub normed_out: DevicePtr,
}

/// 2026-10-10: Launch the fold for one token. `w` is the next site's weights; `partials` where
/// the other ranks' partials landed (`CommBackend::exchange_partials`), or `None` when `own`
/// already holds the sum. `[hidden, hc_mult, sinkhorn_iters]` and
/// `[norm_eps, hc_eps, rms_eps]` as `glm_hc_pre_norm` takes them.
pub fn glm_hc_post_pre_norm(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextMhcKernels,
    w: &Glm5NextMhcSiteWeights,
    io: PostFoldIo,
    partials: Option<&metrale_comm::PeerPartials>,
    [hidden, hc_mult, sinkhorn_iters]: [u32; 3],
    [norm_eps, hc_eps, rms_eps]: [f32; 3],
    stream: u64,
) -> Result<()> {
    if kernels.hc_post_pre_norm.0 == 0 {
        bail!("glm_hc_post_pre_norm: glm5next_hc_post_pre_norm is not in this build");
    }
    if !(1..=4).contains(&hc_mult) {
        bail!("glm_hc_post_pre_norm: hc_mult {hc_mult} is outside 1..=4");
    }
    if hidden == 0 || !hidden.is_multiple_of(256) || hidden > POST_FOLD_MAX_HIDDEN {
        bail!(
            "glm_hc_post_pre_norm: hidden {hidden} is not a multiple of 256 up to \
             {POST_FOLD_MAX_HIDDEN}"
        );
    }
    if !w.hc_fn_bf16 {
        bail!("glm_hc_post_pre_norm: the site's hc_fn is FP32; the kernel reads BF16 only");
    }
    if io.streams_out == io.residual || io.streams_out.0 == 0 {
        bail!("glm_hc_post_pre_norm: the post-mixed streams need their own buffer");
    }
    if let Some(p) = partials
        && (hidden as usize) > p.slot_elems
    {
        bail!(
            "glm_hc_post_pre_norm: {hidden} elements exceed the one-shot peer slot of {}",
            p.slot_elems
        );
    }
    let (peers, slot, n_ranks, rank) = match partials {
        Some(p) => (
            DevicePtr(p.recv),
            p.slot_elems as u64,
            p.world_size as i32,
            p.rank as i32,
        ),
        None => (DevicePtr::NULL, 0, 0, 0),
    };
    KernelLaunch::new(gpu, kernels.hc_post_pre_norm)
        .grid([1, mix_hc(hc_mult as usize) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(io.own)
        .arg_ptr(peers)
        .arg_u64(slot)
        .arg_i32(n_ranks)
        .arg_i32(rank)
        .arg_ptr(io.residual)
        .arg_ptr(io.post)
        .arg_ptr(io.comb)
        .arg_ptr(io.streams_out)
        .arg_ptr(w.hc_fn)
        .arg_ptr(w.hc_scale)
        .arg_ptr(w.hc_base)
        .arg_ptr(w.mix)
        .arg_ptr(w.done)
        .arg_ptr(io.norm_w)
        .arg_ptr(io.y_out)
        .arg_ptr(io.post)
        .arg_ptr(io.comb)
        .arg_ptr(io.normed_out)
        .arg_u32(hidden)
        .arg_u32(hc_mult)
        .arg_u32(sinkhorn_iters)
        .arg_f32(norm_eps)
        .arg_f32(hc_eps)
        .arg_f32(rms_eps)
        .launch(stream)
}

#[cfg(test)]
#[path = "glm5next_mhc_post_fold_tests.rs"]
mod tests;
