// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: `--glm-mhc-post-pre` (R3): at the decode widths (1..=2 rows: decode and the K=2
//! verify) the attention site's one-shot reduce + post-mix (`glm5next_hc_post_rank_sum`) and the
//! FFN site's pre-norm (`glm5next_hc_pre_norm_c1`) run as one launch,
//! `glm5next_hc_post_pre_norm_c1` (`glm5next_mhc_post_pre_c1.cu`, an instantiation of the c1 body
//! `glm5next_mhc_c1_body.cuh`): one launch and one stream edge fewer per layer.
//!
//! Numerics: bit-identical (class BI). Each block forms the token's new streams with the post-sum's
//! expressions in its order and runs the c1 body on them, so `block_out`, the streams, `y`, `post`,
//! `comb`, `normed` and the `mix` scratch hold the pair's bytes (`glm5next_hc_post_pre_c1_gate`).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Taken only where the pair would run exactly `glm5next_hc_post_rank_sum` then
//!   `glm5next_hc_pre_norm_c1` ([`kernel_for`]): the lever on, the reduce exchanged its partials,
//!   no split-k or token-tiled pre-norm ahead of c1, and c1 itself taken; otherwise the caller runs
//!   the pair. The lever on with the kernel absent is an error, never a quiet fallback.
//! - In place only: the new streams overwrite `residual` ([`launch`] refuses another output).

use anyhow::{Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::glm5next_mhc::{Glm5NextMhcKernels, mix_hc};
use crate::glm5next_mhc_fused::{HC_PRE_NORM_ROWS_MIN_TOKENS, hc_pre_norm_refusal};
use crate::glm5next_mhc_post_pre_tok::PostPreTok;

/// 2026-10-24: The kernel module and entry.
pub const HC_POST_PRE_C1_MODULE: &str = "glm5next_mhc_post_pre_c1";
pub const HC_POST_PRE_C1_KERNEL: &str = "glm5next_hc_post_pre_norm_c1";
/// 2026-10-24: The most ranks the kernel sums (`glmc1::MAX_RANKS`).
pub const HC_POST_PRE_C1_MAX_RANKS: usize = 8;

/// 2026-10-24: The fused kernel for a `num_tokens`-row step when the pair it replaces would run
/// `glm5next_hc_post_rank_sum` (`exchanged`: the reduce left its partials) then the c1 pre-norm
/// (`pre_fused`: the site runs the fused pre-norm, `METRALE_GLM_MHC_FUSED` and `fused_runs`; then
/// `glm_hc_pre_norm`'s precedence: no tile or tok4 grid, which start at
/// `HC_PRE_NORM_ROWS_MIN_TOKENS`, no split-k, c1 on and taking the shape). `None`: the caller runs
/// the pair. An error when the lever is on, the pair would be taken, and the build lacks it.
pub fn kernel_for(
    kernels: &Glm5NextMhcKernels,
    [on, exchanged, pre_fused]: [bool; 3],
    [num_tokens, hidden, hc_mult]: [u32; 3],
) -> Result<Option<KernelHandle>> {
    let l = &kernels.levers;
    if !on
        || !exchanged
        || !pre_fused
        || l.pre_splitk
        || num_tokens as usize >= HC_PRE_NORM_ROWS_MIN_TOKENS
        || crate::glm5next_mhc_c1::c1_kernel(kernels, num_tokens, hidden, hc_mult, l.pre_c1)
            .is_none()
    {
        return Ok(None);
    }
    ensure!(
        kernels.hc_post_pre_norm_c1.0 != 0,
        "--glm-mhc-post-pre needs {HC_POST_PRE_C1_KERNEL} on this target"
    );
    Ok(Some(kernels.hc_post_pre_norm_c1))
}

/// 2026-10-24: Launch `kernel` (`glm5next_hc_post_pre_norm_c1`) on grid `(T, mix rows)`. Errors,
/// before launching, on what either half of the pair refuses (a pre-norm shape
/// `hc_pre_norm_refusal` names, an FP32 `hc_fn`, more tokens than the `mix` scratch holds, rows
/// past a peer slot, a rank outside the world), more ranks than the kernel sums, or an output
/// other than `residual`.
pub fn launch(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: &PostPreTok,
    stream: u64,
) -> Result<()> {
    let [hidden, hc_mult, sinkhorn_iters] = a.dims;
    let [norm_eps, hc_eps, rms_eps] = a.eps;
    if let Some(why) = hc_pre_norm_refusal(hidden, hc_mult) {
        bail!("{HC_POST_PRE_C1_KERNEL}: {why}");
    }
    ensure!(
        a.site.hc_fn_bf16,
        "{HC_POST_PRE_C1_KERNEL}: the site's hc_fn is FP32; the kernel reads BF16 only"
    );
    ensure!(
        a.num_tokens as usize <= a.site.mix_tokens,
        "{HC_POST_PRE_C1_KERNEL}: {} tokens exceed the {}-token `mix` scratch",
        a.num_tokens,
        a.site.mix_tokens
    );
    let elems = a.num_tokens as usize * hidden as usize;
    ensure!(
        elems <= a.partials.slot_elems,
        "{HC_POST_PRE_C1_KERNEL}: {elems} elements exceed the peer slot of {}",
        a.partials.slot_elems
    );
    let world = a.partials.world_size;
    ensure!(
        (1..=HC_POST_PRE_C1_MAX_RANKS).contains(&world) && a.partials.rank < world,
        "{HC_POST_PRE_C1_KERNEL}: rank {} of {world} ranks (at most {HC_POST_PRE_C1_MAX_RANKS})",
        a.partials.rank
    );
    ensure!(
        a.streams_out == a.residual,
        "{HC_POST_PRE_C1_KERNEL}: the new streams overwrite the old ones (in place only)"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([a.num_tokens, mix_hc(hc_mult as usize) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(a.residual)
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
        .arg_ptr(a.block_out)
        .arg_ptr(DevicePtr(a.partials.recv))
        .arg_u64(a.partials.slot_elems as u64)
        .arg_i32(world as i32)
        .arg_i32(a.partials.rank as i32)
        .arg_ptr(a.post)
        .arg_ptr(a.comb)
        .arg_ptr(a.residual)
        .launch(stream)
}

#[cfg(test)]
#[path = "glm5next_mhc_post_pre_c1_tests.rs"]
mod tests;
