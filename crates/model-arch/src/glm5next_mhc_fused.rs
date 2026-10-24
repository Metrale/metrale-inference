// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Host launcher of `glm5next_hc_pre_norm`: one mHC site's pre-mix (`hc_mix_bf16`
//! then `hc_finish`) and the RMS norm of its collapsed row, in one launch with the bits of the
//! three.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - `glm_hc_pre_norm` writes what `glm_hc_pre` followed by `rms_norm_vanilla` (block
//!   `min(hidden, 1024)`) writes, byte for byte: `y_out`, `post_out`, `comb_out` and
//!   `normed_out` (`examples/glm5next_hc_fused_gate/`).
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
    Glm5NextMhcKernels, Glm5NextMhcSiteWeights, HC_MIX_ROWS, MHC_MIX_MAX_TOKENS, mix_hc,
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
/// than the site's `mix_tokens`.
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
    // 2026-10-15: The cp.async-ring tile kernel takes its own launch (`glm5next_mhc_tile`).
    if crate::glm5next_mhc_tile::try_launch_tile(
        gpu,
        kernels,
        streams,
        w,
        norm_w,
        [y_out, post_out, comb_out, normed_out],
        num_tokens,
        [hidden_size, hc_mult, sinkhorn_iters],
        [norm_eps, hc_eps, rms_eps],
        stream,
    )? {
        return Ok(());
    }
    glm_hc_pre_norm_with(
        gpu,
        kernels,
        streams,
        w,
        norm_w,
        y_out,
        normed_out,
        post_out,
        comb_out,
        num_tokens,
        hidden_size,
        hc_mult,
        sinkhorn_iters,
        norm_eps,
        hc_eps,
        rms_eps,
        PreNormChoice {
            prefetch: kernels.levers.pre_prefetch,
            splitk: kernels.levers.pre_splitk,
            tok: kernels.levers.pre_tok,
            c1: kernels.levers.pre_c1,
        },
        stream,
    )
}

/// 2026-10-21: Which of the optional pre-norm kernels a launch may take (each its lever in
/// `glm_hc_pre_norm`; explicit here so a gate can run both arms). Precedence: `tok`, then
/// `splitk`, then `c1` (`glm5next_mhc_c1`), then `prefetch`.
#[derive(Clone, Copy, Debug)]
pub struct PreNormChoice {
    pub prefetch: bool,
    pub splitk: bool,
    pub tok: bool,
    pub c1: bool,
}

/// 2026-10-11: k slices per token of `glm5next_hc_pre_norm_sk64` (`GLM_HCF_SK_SLICES`) and the
/// floats each slice stores in the site's `mix` scratch (`GLM_HCF_SK_VALS`: the RMS partial and
/// the 24 mix partials).
pub const HC_PRE_NORM_SK_SLICES: u32 = 64;
pub const HC_PRE_NORM_SK_VALS: usize = 25;

/// 2026-10-11: Whether the site's `mix` scratch (`mix_tokens * mix_rows` floats) holds
/// the split-k partials of `num_tokens` tokens.
pub fn splitk_fits(num_tokens: u32, mix_rows: u32, mix_cap_tokens: usize) -> bool {
    num_tokens as usize * HC_PRE_NORM_SK_SLICES as usize * HC_PRE_NORM_SK_VALS
        <= mix_cap_tokens * mix_rows as usize
}

/// 2026-10-11: `hc_mult * H` the prefetching kernel is compiled for: 64 loads per thread at the
/// 256-thread block (GLM-5.3: hc 4, H 4096).
pub const HC_PRE_NORM_PF_DIM: u32 = 64 * 256;

/// 2026-10-11: `(kernel, grid y)` for `num_tokens` tokens: the 8-row grid from
/// `HC_PRE_NORM_ROWS_MIN_TOKENS`; below it the prefetching per-row kernel when `prefetch` is
/// asked, the shape is the one it is compiled for and the build has it; else the per-row kernel.
pub fn pre_norm_kernel(
    kernels: &Glm5NextMhcKernels,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    prefetch: bool,
) -> (KernelHandle, u32) {
    let mix_rows = mix_hc(hc_mult as usize) as u32;
    if num_tokens as usize >= HC_PRE_NORM_ROWS_MIN_TOKENS {
        return (
            kernels.hc_pre_norm_rows,
            mix_rows.div_ceil(HC_MIX_ROWS as u32),
        );
    }
    if prefetch
        && hc_mult == 4
        && hc_mult * hidden_size == HC_PRE_NORM_PF_DIM
        && kernels.hc_pre_norm_pf.0 != 0
    {
        return (kernels.hc_pre_norm_pf, mix_rows);
    }
    (kernels.hc_pre_norm, mix_rows)
}

/// 2026-10-11: `(glm5next_hc_pre_norm_sk64, 64)` when `splitk` is asked, the tokens are below
/// `HC_PRE_NORM_ROWS_MIN_TOKENS`, the shape is the one it is compiled for (hc 4,
/// `hc_mult * H == 64 * 256`), the build has it and the site's `mix` scratch
/// (`mix_cap_tokens` tokens) holds the slice partials; `None` otherwise.
pub fn splitk_kernel(
    kernels: &Glm5NextMhcKernels,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    splitk: bool,
    mix_cap_tokens: usize,
) -> Option<(KernelHandle, u32)> {
    let mix_rows = mix_hc(hc_mult as usize) as u32;
    (splitk
        && (num_tokens as usize) < HC_PRE_NORM_ROWS_MIN_TOKENS
        && hc_mult == 4
        && hc_mult * hidden_size == HC_PRE_NORM_SK_SLICES * 256
        && kernels.hc_pre_norm_sk.0 != 0
        && splitk_fits(num_tokens, mix_rows, mix_cap_tokens))
    .then_some((kernels.hc_pre_norm_sk, HC_PRE_NORM_SK_SLICES))
}

/// 2026-10-15: What every pre-norm launch refuses: a shape `hc_pre_norm_refusal` names, an FP32
/// `hc_fn`, more tokens than the site's `mix` scratch. Returns that scratch's token bound.
pub(crate) fn pre_norm_checks(
    w: &Glm5NextMhcSiteWeights,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
) -> Result<usize> {
    if let Some(why) = hc_pre_norm_refusal(hidden_size, hc_mult) {
        bail!("glm_hc_pre_norm: {why}");
    }
    if !w.hc_fn_bf16 {
        bail!("glm_hc_pre_norm: the site's hc_fn is FP32; the fused kernel reads BF16 only");
    }
    let mix_cap = w.mix_tokens;
    if num_tokens as usize > mix_cap {
        bail!(
            "glm_hc_pre_norm: {num_tokens} tokens exceeds the {mix_cap}-token `mix` and `done` \
             scratch (floor {MHC_MIX_MAX_TOKENS})"
        );
    }
    Ok(mix_cap)
}

/// 2026-10-11: [`glm_hc_pre_norm`] with the optional kernels' choice explicit (`choice`; the
/// bit-identity gates run both arms).
#[allow(clippy::too_many_arguments)]
pub fn glm_hc_pre_norm_with(
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
    choice: PreNormChoice,
    stream: u64,
) -> Result<()> {
    let PreNormChoice {
        prefetch,
        splitk,
        tok,
        c1,
    } = choice;
    let mix_cap = pre_norm_checks(w, num_tokens, hidden_size, hc_mult)?;
    // 2026-10-12: The token-tiled grid takes its own launch (`glm5next_mhc_tok`).
    if let Some(kernel) = crate::glm5next_mhc_tok::tok_kernel(kernels, num_tokens, tok) {
        return crate::glm5next_mhc_tok::launch_tok(
            gpu,
            kernel,
            streams,
            w,
            norm_w,
            [y_out, post_out, comb_out, normed_out],
            num_tokens,
            [hidden_size, hc_mult, sinkhorn_iters],
            [norm_eps, hc_eps, rms_eps],
            stream,
        );
    }
    let (kernel, grid_y) =
        splitk_kernel(kernels, num_tokens, hidden_size, hc_mult, splitk, mix_cap)
            .or_else(|| {
                crate::glm5next_mhc_c1::c1_kernel(kernels, num_tokens, hidden_size, hc_mult, c1)
                    .map(|k| (k, mix_hc(hc_mult as usize) as u32))
            })
            .unwrap_or_else(|| {
                pre_norm_kernel(kernels, num_tokens, hidden_size, hc_mult, prefetch)
            });
    crate::glm5next_mhc_pre_launch::launch(
        gpu,
        (kernel, grid_y),
        streams,
        w,
        norm_w,
        [y_out, post_out, comb_out, normed_out],
        num_tokens,
        [hidden_size, hc_mult, sinkhorn_iters],
        [norm_eps, hc_eps, rms_eps],
        stream,
    )
}

/// 2026-10-10: The one-shot all-reduce's sum and `glm_hc_post` in one launch
/// (`glm5next_hc_post_rank_sum`): `block_out` holds this rank's partial and `partials` says where
/// the other ranks' landed (`CommBackend::exchange_partials`). Writes the reduced row back to
/// `block_out` and the post-mix to `out`, both byte-identical to `bf16_add_rank_sum` followed by
/// `glm_hc_post`. Errors when the kernel is not loaded or the rows exceed a peer slot.
/// 2026-10-14: The grid is `METRALE_GLM_MHC_POST_SUM_CTAS`'s (`glm5next_hc_post_sum_grid`).
/// 2026-10-17: Launches `kernels.hc_post_rank_sum` at `kernels.levers.post_sum_ctas`.
#[allow(clippy::too_many_arguments)]
pub fn glm_hc_post_rank_sum(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextMhcKernels,
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
    glm_hc_post_rank_sum_with(
        gpu,
        kernels.hc_post_rank_sum,
        block_out,
        partials,
        [residual, post, comb, out],
        [num_tokens, hidden_size, hc_mult],
        kernels.levers.post_sum_ctas,
        stream,
    )
}

/// 2026-10-14: [`glm_hc_post_rank_sum`] at an explicit block cap (`None`: the uncapped grid);
/// `streams` is `[residual, post, comb, out]`, `dims` `[num_tokens, hidden_size, hc_mult]`.
#[allow(clippy::too_many_arguments)]
pub fn glm_hc_post_rank_sum_with(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    block_out: DevicePtr,
    partials: &metrale_comm::PeerPartials,
    [residual, post, comb, out]: [DevicePtr; 4],
    [num_tokens, hidden_size, hc_mult]: [u32; 3],
    ctas: Option<u32>,
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
    let (grid, block) =
        crate::glm5next_hc_post_sum_grid::launch_shape(num_tokens, hidden_size, ctas);
    KernelLaunch::new(gpu, kernel)
        .grid(grid)
        .block(block)
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
        .arg_u32(num_tokens)
        .launch(stream)
}

#[cfg(test)]
#[path = "glm5next_mhc_fused_tests.rs"]
mod tests;
