// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host launchers for the GLM-5.3 mHC kernels: expand, pre (the `hc_mix` +
//! `hc_finish` pair), post, and the head mean.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Every kernel resolves from `GLM5NEXT_MHC_MODULE`, none from DeepSeek-V4's
//!   `hyper_connection` module.
//! - `glm_hc_pre` refuses more tokens than `mhc_mix_max_tokens()`, the bound the loader sizes
//!   each site's `mix` scratch from.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-09-25: Every kernel the GLM-5.3 hyper-connection launches. `Copy`: the loader resolves
/// them once and every layer holds a copy.
#[derive(Clone, Copy)]
pub struct Glm5NextMhcKernels {
    /// 2026-09-25: Broadcast the embedding into the `hc_mult` highway streams; first text layer
    /// only. A GLM target cannot use `hyper_connection::hc_expand`: that kernel is in the
    /// DeepSeek-V4 model directory, and a target compiles only `common/` and its own model
    /// directory.
    pub hc_expand: KernelHandle,
    /// 2026-09-25: The fused `glm5next_hc_pre`, one block per token. `glm_hc_pre` does not launch
    /// it; `examples/glm5next_hc_split_gate.rs` uses it as the oracle for `hc_mix` + `hc_finish`.
    pub hc_pre: KernelHandle,
    /// 2026-09-25: `glm5next_hc_mix`: one block per (token, mixing row), grid `(T, mix_hc)`,
    /// reading an FP32 `hc_fn`.
    pub hc_mix: KernelHandle,
    /// 2026-09-25: `glm5next_hc_mix` reading a BF16 `hc_fn`. Widening BF16 to FP32 is exact, so it
    /// computes the FP32 kernel's result on the widened weights. Optional (`try_kernel`, 0 when
    /// absent); `glm_hc_pre` uses it when the site's `hc_fn_bf16` is set and the handle is not 0.
    pub hc_mix_bf16: KernelHandle,
    /// 2026-10-09: `glm5next_hc_mix_bf16_rows`: `hc_mix_bf16` with one block per (token,
    /// [`HC_MIX_ROWS`] mixing rows), each mix's bits unchanged; `glm_hc_pre` takes it from
    /// [`HC_MIX_ROWS_MIN_TOKENS`] tokens. Optional (`try_kernel`, 0 when absent).
    pub hc_mix_bf16_rows: KernelHandle,
    /// 2026-09-25: `glm5next_hc_finish`: from the mixes, `post`, `comb` (Sinkhorn) and the
    /// collapsed row `y`.
    pub hc_finish: KernelHandle,
    /// 2026-10-09: `glm5next_hc_pre_norm` (module [`GLM5NEXT_MHC_FUSED_MODULE`]): `hc_mix_bf16`,
    /// `hc_finish` and the following `rms_norm_vanilla` in one launch, bit for bit; launched by
    /// `glm5next_mhc_fused::glm_hc_pre_norm`. Optional (`try_kernel`, 0 when absent).
    pub hc_pre_norm: KernelHandle,
    /// 2026-10-09: `glm5next_hc_pre_norm_rows`: `hc_pre_norm` with `hc_mix_bf16_rows`'s grid.
    /// Optional (`try_kernel`, 0 when absent).
    pub hc_pre_norm_rows: KernelHandle,
    /// 2026-10-11: `glm5next_hc_pre_norm_pf64`: `hc_pre_norm` with every per-thread load in
    /// flight at once, same bits, at `hc_mult * H == 64 * 256` (GLM-5.3); taken under
    /// `METRALE_GLM_MHC_PRE_PREFETCH=1` (`glm5next_mhc_fused::pre_norm_kernel`). Optional
    /// (`try_kernel`, 0 when absent).
    pub hc_pre_norm_pf: KernelHandle,
    /// 2026-10-11: `glm5next_hc_pre_norm_sk64`: `hc_pre_norm` with the mix split along k over 64
    /// blocks per token, at `hc_mult * H == 64 * 256` (GLM-5.3); taken under
    /// `METRALE_GLM_MHC_PRE_SPLITK=1`. Reorders the mix sums (accuracy class RO). Optional
    /// (`try_kernel`, 0 when absent).
    pub hc_pre_norm_sk: KernelHandle,
    pub hc_post: KernelHandle,
    /// 2026-10-10: `glm5next_hc_post_rank_sum` (module [`GLM5NEXT_HC_POST_SUM_MODULE`]): the
    /// one-shot all-reduce's sum fused into `hc_post`, same bits. Optional (`try_kernel`, 0 when
    /// absent); taken under `METRALE_GLM_AR_FUSED_POST=1`.
    pub hc_post_rank_sum: KernelHandle,
    /// 2026-10-10: `glm5next_hc_post_pre_norm`: one token's post-mix (with the one-shot sum in
    /// front) and the next site's pre-norm in one launch, same bits; taken by the decode under
    /// `METRALE_GLM_MHC_POST_FOLD=1`. Optional (`try_kernel`, 0 when absent).
    pub hc_post_pre_norm: KernelHandle,
    pub hc_head: KernelHandle,
}

/// 2026-09-25: The kernel module every GLM mHC kernel resolves from.
pub const GLM5NEXT_MHC_MODULE: &str = "glm5next_mhc";

/// 2026-10-09: The kernel module of the fused pre-mix + norm (`glm5next_mhc_fused.cu`).
pub const GLM5NEXT_MHC_FUSED_MODULE: &str = "glm5next_mhc_fused";

/// 2026-10-10: The kernel module of the folded post-mix + pre-norm (`glm5next_mhc_post_fold.cu`).
pub const GLM5NEXT_MHC_POST_FOLD_MODULE: &str = "glm5next_mhc_post_fold";

/// 2026-10-10: The kernel module of the fused all-reduce sum + post-mix (`glm5next_hc_post_sum.cu`).
pub const GLM5NEXT_HC_POST_SUM_MODULE: &str = "glm5next_hc_post_sum";

impl Glm5NextMhcKernels {
    /// 2026-09-25: Resolve the kernels. A missing one is an error, except `hc_mix_bf16`, which is
    /// 0 when absent.
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            hc_expand: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_expand")?,
            hc_pre: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_pre")?,
            hc_mix: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_mix")?,
            hc_mix_bf16: metrale_model_layers::layers::try_kernel(
                gpu,
                GLM5NEXT_MHC_MODULE,
                "glm5next_hc_mix_bf16",
            ),
            hc_mix_bf16_rows: metrale_model_layers::layers::try_kernel(
                gpu,
                GLM5NEXT_MHC_MODULE,
                "glm5next_hc_mix_bf16_rows",
            ),
            hc_finish: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_finish")?,
            hc_pre_norm: metrale_model_layers::layers::try_kernel(
                gpu,
                GLM5NEXT_MHC_FUSED_MODULE,
                "glm5next_hc_pre_norm",
            ),
            hc_pre_norm_rows: metrale_model_layers::layers::try_kernel(
                gpu,
                GLM5NEXT_MHC_FUSED_MODULE,
                "glm5next_hc_pre_norm_rows",
            ),
            hc_pre_norm_pf: metrale_model_layers::layers::try_kernel(
                gpu,
                GLM5NEXT_MHC_FUSED_MODULE,
                "glm5next_hc_pre_norm_pf64",
            ),
            hc_pre_norm_sk: metrale_model_layers::layers::try_kernel(
                gpu,
                GLM5NEXT_MHC_FUSED_MODULE,
                "glm5next_hc_pre_norm_sk64",
            ),
            hc_post: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_post")?,
            hc_post_pre_norm: metrale_model_layers::layers::try_kernel(
                gpu,
                GLM5NEXT_MHC_POST_FOLD_MODULE,
                "glm5next_hc_post_pre_norm",
            ),
            hc_post_rank_sum: metrale_model_layers::layers::try_kernel(
                gpu,
                GLM5NEXT_HC_POST_SUM_MODULE,
                "glm5next_hc_post_rank_sum",
            ),
            hc_head: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_head")?,
        })
    }
}

/// 2026-09-25: Broadcast each of `num_tokens` BF16 hidden rows into its `hc_mult` FP32 highway
/// streams.
pub fn glm_hc_expand(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    streams: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(streams)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

/// 2026-09-25: Collapse the highway to one BF16 row per token: the unweighted mean of the
/// `hc_mult` streams. It takes no weights, unlike DeepSeek-V4's `ops::hc_head`.
pub fn hc_head_mean(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    y_out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(y_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

/// 2026-09-25: One mHC site's weights; a layer has one set for the attention site and one for
/// the FFN site.
#[derive(Debug, Clone, Copy)]
pub struct Glm5NextMhcSiteWeights {
    /// 2026-09-25: `[mix_hc, hc_mult * hidden]` (see `mix_hc`): BF16 when `hc_fn_bf16` is set,
    /// FP32 otherwise. The loader uploads it as BF16.
    pub hc_fn: DevicePtr,
    /// 2026-09-25: Whether `hc_fn` is BF16; the loader sets it.
    pub hc_fn_bf16: bool,
    /// 2026-09-25: `[3]` FP32: the logit scales of pre, post and comb, in that order.
    pub hc_scale: DevicePtr,
    /// 2026-09-25: `[mix_hc]` FP32.
    pub hc_base: DevicePtr,
    /// 2026-09-25: `[mhc_mix_max_tokens(), mix_hc]` FP32 scratch that `hc_mix` writes and
    /// `hc_finish` reads; the loader allocates one per site.
    pub mix: DevicePtr,
    /// 2026-10-09: `[mhc_mix_max_tokens()]` u32, zeroed at allocation: the per-token arrival
    /// counters of `glm5next_hc_pre_norm`, whose last block for a token zeroes its counter
    /// again. One per site, like `mix`.
    pub done: DevicePtr,
}

/// 2026-10-09: `#define GLM_HC_MIX_ROWS` in `glm5next_mhc.cu`: mixing rows per block of
/// `glm5next_hc_mix_bf16_rows`.
pub const HC_MIX_ROWS: usize = 8;

/// 2026-10-09: Fewest tokens `glm_hc_pre` sends to `glm5next_hc_mix_bf16_rows`: 32 tokens give
/// 96 blocks at GLM-5.3's 24 mixing rows, two per GB10 SM. Below it the per-row kernel's
/// T x 24 blocks keep the SMs busier. Not yet measured; the C16 bench times both.
pub const HC_MIX_ROWS_MIN_TOKENS: usize = 32;

/// 2026-09-25: The lower bound of `mhc_mix_max_tokens()`.
pub const MHC_MIX_MAX_TOKENS: usize = 256;

/// 2026-09-25: Token bound of each site's `mix` scratch, `max(prefill_rows(), MHC_MIX_MAX_TOKENS)`,
/// read once. The loader allocates the scratch from it and `glm_hc_pre` refuses more tokens
/// than it, so the allocation and the check read one value.
pub fn mhc_mix_max_tokens() -> usize {
    static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        let t = crate::glm5next_layer::prefill_rows().max(MHC_MIX_MAX_TOKENS);
        if t != MHC_MIX_MAX_TOKENS {
            tracing::warn!(
                "GLM mHC `mix` scratch widened to {t} tokens (floor {MHC_MIX_MAX_TOKENS}) to \
                 match METRALE_GLM_PREFILL_ROWS"
            );
        }
        t
    })
}

/// 2026-09-25: Collapse each token's `hc_mult` FP32 streams to one BF16 row in `y_out`, and write
/// this site's `post` and `comb`, by launching `hc_mix` (or `hc_mix_bf16`) then `hc_finish`.
/// `streams` is read, not written, so `glm_hc_post` can take it as its residual. Errors when
/// `num_tokens` exceeds `mhc_mix_max_tokens()`.
#[allow(clippy::too_many_arguments)]
pub fn glm_hc_pre(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextMhcKernels,
    streams: DevicePtr,
    w: &Glm5NextMhcSiteWeights,
    y_out: DevicePtr,
    post_out: DevicePtr,
    comb_out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    sinkhorn_iters: u32,
    norm_eps: f32,
    hc_eps: f32,
    stream: u64,
) -> Result<()> {
    let mix_hc = (2 + hc_mult) * hc_mult;
    let mix_cap = mhc_mix_max_tokens();
    if num_tokens as usize > mix_cap {
        anyhow::bail!(
            "glm_hc_pre: {num_tokens} tokens exceeds the {mix_cap}-token `mix` scratch \
             (floor {MHC_MIX_MAX_TOKENS}, widened to METRALE_GLM_PREFILL_ROWS). Do not launch \
             past the allocation."
        );
    }
    // 2026-09-25: Both mix kernels take the same arguments; only the element width of `hc_fn`
    // differs, so `hc_fn_bf16` must describe the pointer.
    let mix_kernel = if w.hc_fn_bf16 && kernels.hc_mix_bf16.0 != 0 {
        kernels.hc_mix_bf16
    } else {
        kernels.hc_mix
    };
    // 2026-10-09: From `HC_MIX_ROWS_MIN_TOKENS` tokens, one block per (token, `HC_MIX_ROWS`
    // rows) reads each token's streams once per group instead of once per row; each mix keeps
    // its bits. Fewer tokens keep one block per (token, row) for the parallelism.
    let (mix_kernel, mix_grid_y) = if w.hc_fn_bf16
        && kernels.hc_mix_bf16_rows.0 != 0
        && num_tokens as usize >= HC_MIX_ROWS_MIN_TOKENS
    {
        (
            kernels.hc_mix_bf16_rows,
            mix_hc.div_ceil(HC_MIX_ROWS as u32),
        )
    } else {
        (mix_kernel, mix_hc)
    };
    KernelLaunch::new(gpu, mix_kernel)
        .grid([num_tokens, mix_grid_y, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(w.hc_fn)
        .arg_ptr(w.mix)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_f32(norm_eps)
        .launch(stream)?;
    KernelLaunch::new(gpu, kernels.hc_finish)
        // 2026-09-25: Block `y == 0` computes post, comb and the Sinkhorn; blocks `1..` split the
        // collapse.
        .grid([num_tokens, 1 + collapse_blocks(hidden_size), 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(w.mix)
        .arg_ptr(w.hc_scale)
        .arg_ptr(w.hc_base)
        .arg_ptr(y_out)
        .arg_ptr(post_out)
        .arg_ptr(comb_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(sinkhorn_iters)
        .arg_f32(hc_eps)
        .launch(stream)
}

/// 2026-09-25: `out[j] = post[j] * block_out + Σ_i comb[i][j] * residual[i]` for each token.
/// `out` may alias `residual`: the kernel reads every residual stream of an element before it
/// writes that element.
#[allow(clippy::too_many_arguments)]
pub fn glm_hc_post(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    block_out: DevicePtr,
    residual: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, collapse_blocks(hidden_size), 1])
        .block([256, 1, 1])
        .arg_ptr(block_out)
        .arg_ptr(residual)
        .arg_ptr(post)
        .arg_ptr(comb)
        .arg_ptr(out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

/// 2026-09-25: Blocks along grid y for `hc_finish`'s collapse and for `hc_post`: `ceil(H / 256)`,
/// at least 1, where 256 is the block width both launch at. Each output element is computed on
/// its own, so the block count does not change the result.
pub(crate) const fn collapse_blocks(hidden_size: u32) -> u32 {
    if hidden_size < 256 {
        1
    } else {
        hidden_size.div_ceil(256)
    }
}

/// 2026-09-25: Row count of `hc_fn` and `hc_base`, `(2 + hc_mult) * hc_mult`: `hc_mult` rows
/// each for pre and post, then `hc_mult * hc_mult` for comb.
pub fn mix_hc(hc_mult: usize) -> usize {
    (2 + hc_mult) * hc_mult
}

#[cfg(test)]
mod mhc_shape_tests {
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

    /// 2026-09-25: Without `METRALE_GLM_PREFILL_ROWS`, `prefill_rows()` is `PREFILL_ROWS` (16), so
    /// the bound is the 256-token floor.
    #[test]
    fn the_mix_bound_never_goes_below_the_shipped_floor() {
        assert_eq!(mhc_mix_max_tokens(), MHC_MIX_MAX_TOKENS);
        assert!(mhc_mix_max_tokens() >= MHC_MIX_MAX_TOKENS);
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
        let k = |h| KernelHandle(h);
        let kernels = Glm5NextMhcKernels {
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
            hc_post: k(7),
            hc_post_rank_sum: k(11),
            hc_post_pre_norm: k(12),
            hc_head: k(8),
        };
        let site = |bf16| Glm5NextMhcSiteWeights {
            hc_fn: DevicePtr(0x100),
            hc_fn_bf16: bf16,
            hc_scale: DevicePtr(0x200),
            hc_base: DevicePtr(0x300),
            mix: DevicePtr(0x400),
            done: DevicePtr(0x480),
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
}
