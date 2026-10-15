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
    Glm5NextMhcKernels, Glm5NextMhcSiteWeights, HC_MIX_ROWS, MHC_MIX_MAX_TOKENS,
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
        pre_prefetch(),
        pre_splitk(),
        crate::glm5next_mhc_tok::pre_tok(),
        stream,
    )
}

/// 2026-10-11: `METRALE_GLM_MHC_PRE_PREFETCH=1`: the per-row pre-norm takes
/// `glm5next_hc_pre_norm_pf64` where the shape allows (`pre_norm_kernel`): each thread issues all
/// 64 of its stream loads and its 64 `hc_fn` loads before accumulating them in the same order,
/// instead of eight at a time, so the 24-block launch at one token stops waiting on load
/// latency. Same bits (`glm5next_hc_fused_gate` checks it). Off until measured. Read once.
pub fn pre_prefetch() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_GLM_MHC_PRE_PREFETCH").as_deref() == Ok("1"))
}

/// 2026-10-11: `METRALE_GLM_MHC_PRE_SPLITK=1`: below `HC_PRE_NORM_ROWS_MIN_TOKENS` the pre-norm
/// takes `glm5next_hc_pre_norm_sk64` where the shape allows and the site's `mix` scratch holds
/// the slice partials (`pre_norm_kernel`): the mix runs on 64 blocks per token (k split) instead
/// of 24 (one per mix row). Reorders the mix sums, so it is accuracy class RO. Off until
/// measured. Read once.
pub fn pre_splitk() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_GLM_MHC_PRE_SPLITK").as_deref() == Ok("1"))
}

/// 2026-10-11: k slices per token of `glm5next_hc_pre_norm_sk64` (`GLM_HCF_SK_SLICES`) and the
/// floats each slice stores in the site's `mix` scratch (`GLM_HCF_SK_VALS`: the RMS partial and
/// the 24 mix partials).
pub const HC_PRE_NORM_SK_SLICES: u32 = 64;
pub const HC_PRE_NORM_SK_VALS: usize = 25;

/// 2026-10-11: Whether the site's `mix` scratch (`mhc_mix_max_tokens() * mix_rows` floats) holds
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
    let mix_cap = mhc_mix_max_tokens();
    if num_tokens as usize > mix_cap {
        bail!(
            "glm_hc_pre_norm: {num_tokens} tokens exceeds the {mix_cap}-token `mix` and `done` \
             scratch (floor {MHC_MIX_MAX_TOKENS})"
        );
    }
    Ok(mix_cap)
}

/// 2026-10-11: [`glm_hc_pre_norm`] with the prefetching kernel's choice explicit (`prefetch`;
/// the bit-identity gate runs both arms).
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
    prefetch: bool,
    splitk: bool,
    tok: bool,
    stream: u64,
) -> Result<()> {
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
        splitk_kernel(kernels, num_tokens, hidden_size, hc_mult, splitk, mix_cap).unwrap_or_else(
            || pre_norm_kernel(kernels, num_tokens, hidden_size, hc_mult, prefetch),
        );
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
/// 2026-10-14: The grid is `METRALE_GLM_MHC_POST_SUM_CTAS`'s (`glm5next_hc_post_sum_grid`).
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
    let ctas = crate::glm5next_hc_post_sum_grid::ctas()?;
    glm_hc_post_rank_sum_with(
        gpu,
        kernel,
        block_out,
        partials,
        [residual, post, comb, out],
        [num_tokens, hidden_size, hc_mult],
        ctas,
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
            hc_pre_norm_pf: k(if fused { 13 } else { 0 }),
            hc_pre_norm_sk: k(if fused { 14 } else { 0 }),
            hc_pre_norm_tok: k(if fused { 15 } else { 0 }),
            hc_pre_norm_tile: k(if fused { 17 } else { 0 }),
            hc_post: k(7),
            hc_post_rank_sum: k(11),
            hc_post_pre_norm: k(12),
            hc_post_pre_norm_tok: k(if fused { 16 } else { 0 }),
            hc_head: k(8),
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
                b(&2u32.to_le_bytes()),
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
