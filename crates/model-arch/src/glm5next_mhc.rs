// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host launchers for the GLM-5.3 mHC kernels: expand, pre (the `hc_mix` +
//! `hc_finish` pair), post, and the head mean.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Every kernel resolves from `GLM5NEXT_MHC_MODULE`, none from DeepSeek-V4's
//!   `hyper_connection` module.
//! - `glm_hc_pre` refuses more tokens than the site's `mix_tokens`, the bound the loader sized
//!   the site's `mix` scratch from.

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
    /// 2026-10-12: `glm5next_hc_pre_norm_tok4` (module [`GLM5NEXT_MHC_TOK_MODULE`]):
    /// `hc_pre_norm` with four tokens and every mix row per block, same bits, for wide prefill
    /// sub-chunks; taken under `METRALE_GLM_MHC_PRE_TOK=1` (`glm5next_mhc_tok`). Optional
    /// (`try_kernel`, 0 when absent).
    pub hc_pre_norm_tok: KernelHandle,
    /// 2026-10-15: `glm5next_hc_pre_norm_tile4` (`glm5next_mhc_tile`): tok4's bits; optional.
    pub hc_pre_norm_tile: KernelHandle,
    /// 2026-10-21: `glm5next_hc_pre_norm_c1` (`glm5next_mhc_c1`): `hc_pre_norm`'s bits with the
    /// last block's tail shortened, for decode steps; optional.
    pub hc_pre_norm_c1: KernelHandle,
    pub hc_post: KernelHandle,
    /// 2026-10-10: `glm5next_hc_post_rank_sum` (module [`GLM5NEXT_HC_POST_SUM_MODULE`]): the
    /// one-shot all-reduce's sum fused into `hc_post`, same bits. Optional (`try_kernel`, 0 when
    /// absent); taken under `METRALE_GLM_AR_FUSED_POST=1`.
    pub hc_post_rank_sum: KernelHandle,
    /// 2026-10-15: Its 16-byte-access twin, same bits (`glm5next_hc_post_vec`; 0 when absent).
    pub hc_post_rank_sum_vec8: KernelHandle,
    /// 2026-10-10: `glm5next_hc_post_pre_norm`: one token's post-mix (with the one-shot sum in
    /// front) and the next site's pre-norm in one launch, same bits; taken by the decode under
    /// `METRALE_GLM_MHC_POST_FOLD=1`. Optional (`try_kernel`, 0 when absent).
    pub hc_post_pre_norm: KernelHandle,
    /// 2026-10-14: `glm5next_hc_post_pre_norm_tok4` (module [`GLM5NEXT_MHC_TOK_MODULE`]): a
    /// prefill piece's `hc_post_rank_sum` and the next site's `hc_pre_norm_tok` in one launch,
    /// same bits; taken by SP-mHC under `METRALE_GLM_MHC_POST_PRE_FUSED=1`
    /// (`glm5next_mhc_post_pre_tok`). Optional (`try_kernel`, 0 when absent).
    pub hc_post_pre_norm_tok: KernelHandle,
    pub hc_head: KernelHandle,
    /// 2026-10-17: The mHC launch levers, resolved once at startup (`GlmLevers::mhc`).
    pub levers: crate::glm5next_mhc_levers::MhcLevers,
}

/// 2026-09-25: The kernel module every GLM mHC kernel resolves from.
pub const GLM5NEXT_MHC_MODULE: &str = "glm5next_mhc";

/// 2026-10-09: The kernel module of the fused pre-mix + norm (`glm5next_mhc_fused.cu`).
pub const GLM5NEXT_MHC_FUSED_MODULE: &str = "glm5next_mhc_fused";
/// 2026-10-12: The token-tiled pre-norm kernels (`glm5next_mhc_tok.cu`).
pub const GLM5NEXT_MHC_TOK_MODULE: &str = "glm5next_mhc_tok";
/// 2026-10-15: The cp.async-ring pre-norm kernels (`glm5next_mhc_tile.cu`).
pub const GLM5NEXT_MHC_TILE_MODULE: &str = "glm5next_mhc_tile";

/// 2026-10-10: The kernel module of the folded post-mix + pre-norm (`glm5next_mhc_post_fold.cu`).
pub const GLM5NEXT_MHC_POST_FOLD_MODULE: &str = "glm5next_mhc_post_fold";

/// 2026-10-10: The kernel module of the fused all-reduce sum + post-mix (`glm5next_hc_post_sum.cu`).
pub const GLM5NEXT_HC_POST_SUM_MODULE: &str = "glm5next_hc_post_sum";

impl Glm5NextMhcKernels {
    /// 2026-09-25: Resolve the kernels. A missing one is an error, except `hc_mix_bf16`, which is
    /// 0 when absent.
    /// 2026-10-17: `levers` rides along for the launchers.
    pub fn resolve(
        gpu: &dyn GpuBackend,
        levers: crate::glm5next_mhc_levers::MhcLevers,
    ) -> Result<Self> {
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
            hc_pre_norm_tok: metrale_model_layers::layers::try_kernel(
                gpu,
                GLM5NEXT_MHC_TOK_MODULE,
                "glm5next_hc_pre_norm_tok4",
            ),
            hc_pre_norm_tile: metrale_model_layers::layers::try_kernel(
                gpu,
                GLM5NEXT_MHC_TILE_MODULE,
                crate::glm5next_mhc_tile::HC_PRE_TILE_KERNEL,
            ),
            hc_pre_norm_c1: metrale_model_layers::layers::try_kernel(
                gpu,
                crate::glm5next_mhc_c1::HC_PRE_C1_MODULE,
                crate::glm5next_mhc_c1::HC_PRE_C1_KERNEL,
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
            hc_post_rank_sum_vec8: crate::glm5next_hc_post_vec::resolve(gpu),
            hc_post_pre_norm_tok: metrale_model_layers::layers::try_kernel(
                gpu,
                GLM5NEXT_MHC_TOK_MODULE,
                "glm5next_hc_post_pre_norm_tok4",
            ),
            hc_head: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_head")?,
            levers,
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
    /// 2026-09-25: `[mix_tokens, mix_hc]` FP32 scratch that `hc_mix` writes and
    /// `hc_finish` reads; the loader allocates one per site.
    pub mix: DevicePtr,
    /// 2026-10-09: `[mix_tokens]` u32, zeroed at allocation: the per-token arrival
    /// counters of `glm5next_hc_pre_norm`, whose last block for a token zeroes its counter
    /// again. One per site, like `mix`.
    pub done: DevicePtr,
    /// 2026-10-17: Token rows of `mix` and `done`, [`mhc_mix_max_tokens`] of the prefill width
    /// the loader allocated them for; every launcher refuses more tokens.
    pub mix_tokens: usize,
}

/// 2026-10-09: `#define GLM_HC_MIX_ROWS` in `glm5next_mhc.cu`: mixing rows per block of
/// `glm5next_hc_mix_bf16_rows`.
pub const HC_MIX_ROWS: usize = 8;

/// 2026-10-09: Fewest tokens `glm_hc_pre` sends to `glm5next_hc_mix_bf16_rows`: 32 tokens give
/// 96 blocks at GLM-5.3's 24 mixing rows, two per GB10 SM. Below it the per-row kernel's
/// T x 24 blocks keep the SMs busier. Not yet measured; the C16 bench times both.
pub const HC_MIX_ROWS_MIN_TOKENS: usize = 32;

/// 2026-09-25: The lower bound of [`mhc_mix_max_tokens`].
pub const MHC_MIX_MAX_TOKENS: usize = 256;

/// 2026-09-25: Token bound of each site's `mix` scratch, `max(prefill_rows, MHC_MIX_MAX_TOKENS)`.
/// 2026-10-17: The loader allocates the scratch from it and stores it as the site's
/// `mix_tokens`, which `glm_hc_pre` checks, so the allocation and the check read one value.
pub fn mhc_mix_max_tokens(prefill_rows: usize) -> usize {
    prefill_rows.max(MHC_MIX_MAX_TOKENS)
}

/// 2026-09-25: Collapse each token's `hc_mult` FP32 streams to one BF16 row in `y_out`, and write
/// this site's `post` and `comb`, by launching `hc_mix` (or `hc_mix_bf16`) then `hc_finish`.
/// `streams` is read, not written, so `glm_hc_post` can take it as its residual. Errors when
/// `num_tokens` exceeds the site's `mix_tokens`.
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
    let mix_cap = w.mix_tokens;
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
#[path = "glm5next_mhc_tests.rs"]
pub(crate) mod mhc_shape_tests;
