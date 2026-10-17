// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `METRALE_GLM_MHC_POST_PRE_FUSED=1`: in the SP-mHC layer body
//! (`glm5next_layer/steps/forward_sp.rs`) the attention site's reduce + post-mix
//! (`glm5next_hc_post_rank_sum`) and the FFN site's pre-norm (`glm5next_hc_pre_norm_tok4`) of a
//! row piece run as one launch, `glm5next_hc_post_pre_norm_tok4` (`glm5next_mhc_tok.cu`). The
//! pair reads the new streams back from DRAM after writing them; the fused kernel reads them
//! back, in the same block, from L2, and drops a launch per piece.
//!
//! Numerics: bit-identical (class BI). The kernel forms every new stream with
//! `glm5next_hc_post_rank_sum`'s expressions in its order and then runs the tok4 pre-norm's
//! body unchanged on them, so `block_out`, the streams, `y`, `post`, `comb`, `normed` and the
//! `mix` scratch hold the pair's bytes (`examples/glm5next_hc_post_pre_tok_bench.rs` checks every
//! one). Every pre-norm grid writes the same bytes (`glm5next_mhc_fused`, `glm5next_mhc_tok`),
//! so the pair it replaces may have run any of them.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Off unless `MhcLevers::post_pre_fused`; on, it needs `METRALE_GLM_SP_MHC=1` (the only body
//!   that runs the pair), or the startup resolve fails (`glm5next_mhc_levers`), and the kernel in
//!   the build, or the first SP-mHC forward fails.
//! - Taken per piece from `HC_PRE_NORM_ROWS_MIN_TOKENS` rows, where the unfused pre-norm runs the
//!   wide grids; narrower pieces keep the two launches.
//! - One launch per piece whatever `METRALE_GLM_MHC_PRE_SLICE` says. Slicing keeps the bytes,
//!   except that slices narrower than `HC_PRE_NORM_ROWS_MIN_TOKENS` under
//!   `METRALE_GLM_MHC_PRE_SPLITK=1` take the split-k grid (class RO); this kernel then writes the
//!   unsplit bytes those levers alone would not.

use anyhow::{Result, bail, ensure};
use metrale_comm::PeerPartials;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::glm5next_mhc::{Glm5NextMhcKernels, Glm5NextMhcSiteWeights};
use crate::glm5next_mhc_fused::{HC_PRE_NORM_ROWS_MIN_TOKENS, hc_pre_norm_refusal};
use crate::glm5next_mhc_tok::HC_PRE_NORM_TOK;

/// 2026-10-14: The fused kernel for a `num_tokens`-row piece: `None` when the lever is off or the
/// piece is narrower than `HC_PRE_NORM_ROWS_MIN_TOKENS` (the caller runs the pair); an error
/// when the lever is on and the build lacks the kernel.
pub fn kernel_for(
    kernels: &Glm5NextMhcKernels,
    on: bool,
    num_tokens: u32,
) -> Result<Option<KernelHandle>> {
    if !on || (num_tokens as usize) < HC_PRE_NORM_ROWS_MIN_TOKENS {
        return Ok(None);
    }
    ensure!(
        kernels.hc_post_pre_norm_tok.0 != 0,
        "METRALE_GLM_MHC_POST_PRE_FUSED=1 needs glm5next_hc_post_pre_norm_tok4 on this target"
    );
    Ok(Some(kernels.hc_post_pre_norm_tok))
}

/// 2026-10-14: One launch's operands. The post-mix: `block_out` holds this rank's partial of the
/// rows and `partials` the other ranks' (`glm_hc_post_rank_sum`'s), `residual`/`post`/`comb` the
/// site's streams and mixes, `streams_out` the new streams (may be `residual`). The pre-norm of
/// the next site on `streams_out`: `site`, `norm_w`, and its outputs `y`, `post_out`, `comb_out`
/// (may be `post`/`comb`) and `normed`. `dims` is `[hidden, hc_mult, sinkhorn_iters]`, `eps`
/// `[norm_eps, hc_eps, rms_eps]` as for `glm_hc_pre_norm`.
pub struct PostPreTok<'a> {
    pub block_out: DevicePtr,
    pub partials: &'a PeerPartials,
    pub residual: DevicePtr,
    pub post: DevicePtr,
    pub comb: DevicePtr,
    pub streams_out: DevicePtr,
    pub site: &'a Glm5NextMhcSiteWeights,
    pub norm_w: DevicePtr,
    pub y: DevicePtr,
    pub post_out: DevicePtr,
    pub comb_out: DevicePtr,
    pub normed: DevicePtr,
    pub num_tokens: u32,
    pub dims: [u32; 3],
    pub eps: [f32; 3],
}

/// 2026-10-14: Launch `kernel` (`glm5next_hc_post_pre_norm_tok4`) on `ceil(num_tokens / 4)`
/// blocks. Errors, before launching, on what either half of the pair refuses: a pre-norm shape
/// `hc_pre_norm_refusal` names, an FP32 `hc_fn`, more tokens than the `mix` scratch holds, rows
/// past a peer slot, or no peers to sum.
pub fn launch(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: &PostPreTok,
    stream: u64,
) -> Result<()> {
    let [hidden, hc_mult, sinkhorn_iters] = a.dims;
    let [norm_eps, hc_eps, rms_eps] = a.eps;
    if let Some(why) = hc_pre_norm_refusal(hidden, hc_mult) {
        bail!("glm5next_hc_post_pre_norm_tok4: {why}");
    }
    ensure!(
        a.site.hc_fn_bf16,
        "glm5next_hc_post_pre_norm_tok4: the site's hc_fn is FP32; the kernel reads BF16 only"
    );
    let mix_cap = a.site.mix_tokens;
    ensure!(
        a.num_tokens as usize <= mix_cap,
        "glm5next_hc_post_pre_norm_tok4: {} tokens exceed the {mix_cap}-token `mix` scratch",
        a.num_tokens
    );
    let elems = a.num_tokens as usize * hidden as usize;
    ensure!(
        elems <= a.partials.slot_elems,
        "glm5next_hc_post_pre_norm_tok4: {elems} elements exceed the peer slot of {}",
        a.partials.slot_elems
    );
    ensure!(
        a.partials.world_size >= 1 && a.partials.rank < a.partials.world_size,
        "glm5next_hc_post_pre_norm_tok4: rank {} of {} ranks",
        a.partials.rank,
        a.partials.world_size
    );
    KernelLaunch::new(gpu, kernel)
        .grid([a.num_tokens.div_ceil(HC_PRE_NORM_TOK), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a.block_out)
        .arg_ptr(DevicePtr(a.partials.recv))
        .arg_u64(a.partials.slot_elems as u64)
        .arg_i32(a.partials.world_size as i32)
        .arg_i32(a.partials.rank as i32)
        .arg_ptr(a.residual)
        .arg_ptr(a.post)
        .arg_ptr(a.comb)
        .arg_ptr(a.streams_out)
        .arg_ptr(a.site.hc_fn)
        .arg_ptr(a.site.hc_scale)
        .arg_ptr(a.site.hc_base)
        .arg_ptr(a.site.mix)
        .arg_ptr(a.site.done)
        .arg_ptr(a.norm_w)
        .arg_ptr(a.y)
        .arg_ptr(a.post_out)
        .arg_ptr(a.comb_out)
        .arg_ptr(a.normed)
        .arg_u32(hidden)
        .arg_u32(hc_mult)
        .arg_u32(sinkhorn_iters)
        .arg_f32(norm_eps)
        .arg_f32(hc_eps)
        .arg_f32(rms_eps)
        .arg_u32(a.num_tokens)
        .launch(stream)
}

#[cfg(test)]
#[path = "glm5next_mhc_post_pre_tok_tests.rs"]
mod tests;
