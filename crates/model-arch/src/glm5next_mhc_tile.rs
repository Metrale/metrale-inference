// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The cp.async-ring mHC pre-norm (`glm5next_hc_pre_norm_tile8` in
//! `glm5next_mhc_tile.cu`), `METRALE_GLM_MHC_PRE_TILE=1`: eight tokens per 512-thread block,
//! each 256-wide k step of `hc_fn` and of the block's FP32 streams staged through shared memory
//! `HC_PRE_TILE_STAGES - 1` steps ahead, instead of `glm5next_hc_pre_norm_tok4`'s four tokens
//! per block loading straight into registers. Each thread keeps tok4's chains, trees and tails,
//! so every output byte is tok4's (`examples/glm5next_mhc_tile_gate.rs`).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Taken only with the lever on, the kernel in the build, `HC_PRE_NORM_ROWS_MIN_TOKENS` or more
//!   tokens, hc 4, H a multiple of 64, and 16-byte aligned `streams` and `hc_fn`; anything else
//!   takes the path the lever-off build takes. The caller has already refused the shapes the
//!   fused kernels do not cover (`hc_pre_norm_refusal`, H at most 8192).
//! - The launch's dynamic shared memory is `tile_smem_bytes()`, the sum `GlmHctLayout<2>` makes
//!   in the kernel source at `GLM_HCT_STAGES` 3.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::glm5next_mhc::{Glm5NextMhcKernels, Glm5NextMhcSiteWeights};
use crate::glm5next_mhc_fused::{HC_PRE_NORM_MAX_HIDDEN, HC_PRE_NORM_ROWS_MIN_TOKENS};

/// 2026-10-15: The lever; `1` takes the tile kernel, unset or `0` keeps the lever-off path,
/// anything else is refused at the first wide pre-norm.
pub const PRE_TILE_ENV: &str = "METRALE_GLM_MHC_PRE_TILE";

/// 2026-10-15: `glm5next_hc_pre_norm_tile8`: `NG` 2 groups of 256 threads, four tokens each.
pub const HC_PRE_TILE_GROUPS: u32 = 2;
/// 2026-10-15: Tokens per block (`GLM_HCT_TOK * NG`).
pub const HC_PRE_TILE_TOKENS: u32 = 4 * HC_PRE_TILE_GROUPS;
/// 2026-10-15: `GLM_HCT_STAGES` the module is compiled with (the source's default).
pub const HC_PRE_TILE_STAGES: u32 = 3;
/// 2026-10-15: `GLM_HCT_HC`: the only `hc_mult` the kernel is compiled for.
pub const HC_PRE_TILE_MULT: u32 = 4;

/// 2026-10-15: `GlmHctLayout<NG>::BYTES`: the ring (per stage the 24 BF16 `hc_fn` rows and the
/// block's FP32 stream rows of one 256-wide step), reused by the trees and the tails, whichever
/// is largest; then the mixes, the pre weights and the norm's warp sums.
pub const fn tile_smem_bytes() -> u32 {
    let mix = (2 + HC_PRE_TILE_MULT) * HC_PRE_TILE_MULT;
    let stage = mix * 256 * 2 + HC_PRE_TILE_TOKENS * 256 * 4;
    let ring = HC_PRE_TILE_STAGES * stage;
    let red = HC_PRE_TILE_GROUPS * (mix + 1) * 256 * 4;
    let y = HC_PRE_TILE_GROUPS * HC_PRE_NORM_MAX_HIDDEN * 2;
    let a = if ring > red { ring } else { red };
    let reuse = if a > y { a } else { y };
    reuse
        + HC_PRE_TILE_TOKENS * mix * 4
        + HC_PRE_TILE_GROUPS * HC_PRE_TILE_MULT * 4
        + HC_PRE_TILE_GROUPS * 33 * 4
}

/// 2026-10-15: Parse the lever's value: `Ok(true)` for `1`, `Ok(false)` unset or `0`.
pub fn parse_pre_tile(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => anyhow::bail!("{PRE_TILE_ENV}={other}: expected 1 or 0"),
    }
}

/// 2026-10-15: The lever, read once, with a one-time readback line when on.
pub fn pre_tile() -> Result<bool> {
    static ON: std::sync::OnceLock<std::result::Result<bool, String>> = std::sync::OnceLock::new();
    ON.get_or_init(|| {
        let on = parse_pre_tile(std::env::var(PRE_TILE_ENV).ok().as_deref())
            .map_err(|e| e.to_string())?;
        if on {
            tracing::info!(
                "{PRE_TILE_ENV}=1: wide GLM mHC pre-norms (hc 4) run {HC_PRE_TILE_TOKENS} \
                 tokens per block through a {HC_PRE_TILE_STAGES}-stage cp.async ring \
                 (glm5next_hc_pre_norm_tile8, {} B shared, same bits as tok4)",
                tile_smem_bytes()
            );
        }
        Ok(on)
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// 2026-10-15: The tile kernel for this launch when `tile` is asked and every condition of the
/// module invariants holds; `None` otherwise.
pub fn tile_kernel(
    kernels: &Glm5NextMhcKernels,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    streams: DevicePtr,
    hc_fn: DevicePtr,
    tile: bool,
) -> Option<KernelHandle> {
    (tile
        && num_tokens as usize >= HC_PRE_NORM_ROWS_MIN_TOKENS
        && hc_mult == HC_PRE_TILE_MULT
        && hidden_size.is_multiple_of(64)
        && hidden_size <= HC_PRE_NORM_MAX_HIDDEN
        && streams.0.is_multiple_of(16)
        && hc_fn.0.is_multiple_of(16)
        && kernels.hc_pre_norm_tile.0 != 0)
        .then_some(kernels.hc_pre_norm_tile)
}

/// 2026-10-15: One launch of `kernel` over `num_tokens` tokens: tok4's arguments in tok4's order
/// (`glm5next_mhc_tok::launch_tok`), on `ceil(num_tokens / 8)` blocks of 512 threads with
/// `tile_smem_bytes()` of dynamic shared memory.
#[allow(clippy::too_many_arguments)]
pub fn launch_tile(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    w: &Glm5NextMhcSiteWeights,
    norm_w: DevicePtr,
    outs: [DevicePtr; 4],
    num_tokens: u32,
    dims: [u32; 3],
    eps: [f32; 3],
    stream: u64,
) -> Result<()> {
    let [y_out, post_out, comb_out, normed_out] = outs;
    let [hidden_size, hc_mult, sinkhorn_iters] = dims;
    let [norm_eps, hc_eps, rms_eps] = eps;
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens.div_ceil(HC_PRE_TILE_TOKENS), 1, 1])
        .block([256 * HC_PRE_TILE_GROUPS, 1, 1])
        .shared_mem(tile_smem_bytes())
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
        .arg_u32(num_tokens)
        .launch(stream)
}

/// 2026-10-15: `glm_hc_pre_norm`'s tile branch: when the lever is on and `tile_kernel` takes the
/// launch, run the pre-norm refusals and launch it (`Ok(true)`); otherwise `Ok(false)` and the
/// caller takes its lever-off path. A bad lever value is an error.
#[allow(clippy::too_many_arguments)]
pub fn try_launch_tile(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextMhcKernels,
    streams: DevicePtr,
    w: &Glm5NextMhcSiteWeights,
    norm_w: DevicePtr,
    outs: [DevicePtr; 4],
    num_tokens: u32,
    dims: [u32; 3],
    eps: [f32; 3],
    stream: u64,
) -> Result<bool> {
    let [hidden_size, hc_mult, _] = dims;
    let tile = pre_tile()?;
    let Some(kernel) = tile_kernel(
        kernels,
        num_tokens,
        hidden_size,
        hc_mult,
        streams,
        w.hc_fn,
        tile,
    ) else {
        return Ok(false);
    };
    crate::glm5next_mhc_fused::pre_norm_checks(w, num_tokens, hidden_size, hc_mult)?;
    launch_tile(
        gpu, kernel, streams, w, norm_w, outs, num_tokens, dims, eps, stream,
    )?;
    Ok(true)
}

#[cfg(test)]
#[path = "glm5next_mhc_tile_tests.rs"]
mod tests;
