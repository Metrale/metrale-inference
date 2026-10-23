// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The launch of a per-row-grid fused mHC pre-norm entry (`glm5next_hc_pre_norm`,
//! `_pf64`, `_sk64`, `_c1`): one block row per token, `grid_y` blocks per token, 256 threads,
//! the site's weights and scratch and the four outputs. Moved from `glm_hc_pre_norm_with` so the
//! hand-written choice and a plan's step launch one way.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: refuses an absent kernel (handle 0).

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::glm5next_mhc::Glm5NextMhcSiteWeights;

/// 2026-10-23: Launch `kernel` over `num_tokens` tokens at `grid_y` blocks a token; `outs` is
/// `[y, post, comb, normed]`, `dims` `[hidden, hc_mult, sinkhorn_iters]`, `eps`
/// `[norm, hc, rms]`.
#[allow(clippy::too_many_arguments)]
pub fn launch(
    gpu: &dyn GpuBackend,
    (kernel, grid_y): (KernelHandle, u32),
    streams: DevicePtr,
    w: &Glm5NextMhcSiteWeights,
    norm_w: DevicePtr,
    [y_out, post_out, comb_out, normed_out]: [DevicePtr; 4],
    num_tokens: u32,
    [hidden_size, hc_mult, sinkhorn_iters]: [u32; 3],
    [norm_eps, hc_eps, rms_eps]: [f32; 3],
    stream: u64,
) -> Result<()> {
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
