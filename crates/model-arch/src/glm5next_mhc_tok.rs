// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: The token-tiled mHC pre-norm grid (`glm5next_hc_pre_norm_tok4` in
//! `glm5next_mhc_tok.cu`), `METRALE_GLM_MHC_PRE_TOK=1`: one block per four tokens with every mix
//! row, instead of the 8-row grid's three blocks per token. Each block reads a token's FP32
//! streams once (the 8-row grid reads them three times) and serves four tokens from each BF16
//! `hc_fn` load (the 8-row grid one). Measured on GB10 (H 4096, hc 4): 1.36 vs 2.08 ms at 2048
//! tokens, 0.36 vs 0.55 ms at 512; slower below about 128 tokens.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - The kernel writes what `glm5next_hc_pre_norm` writes, byte for byte: every mix is the same
//!   per-thread chain and the same 256-wide tree, and each token's tail is the same code
//!   (`examples/glm5next_mhc_slice_bench.rs` checks it against the 8-row grid).
//! - Taken only with the lever on, the kernel in the build, and at least
//!   `HC_PRE_NORM_ROWS_MIN_TOKENS` tokens, where the 8-row grid runs otherwise; the caller has
//!   already refused the shapes the fused kernels do not cover (`hc_pre_norm_refusal`).

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::glm5next_mhc::{Glm5NextMhcKernels, Glm5NextMhcSiteWeights};
use crate::glm5next_mhc_fused::HC_PRE_NORM_ROWS_MIN_TOKENS;

/// 2026-10-12: `#define GLM_HCF_TOK` in `glm5next_mhc_tok.cu`: tokens per block.
pub const HC_PRE_NORM_TOK: u32 = 4;

/// 2026-10-12: The token-tiled kernel for `num_tokens` tokens when `tok` is asked, the tokens
/// reach `HC_PRE_NORM_ROWS_MIN_TOKENS` and the build has it; `None` otherwise.
pub fn tok_kernel(
    kernels: &Glm5NextMhcKernels,
    num_tokens: u32,
    tok: bool,
) -> Option<KernelHandle> {
    (tok && num_tokens as usize >= HC_PRE_NORM_ROWS_MIN_TOKENS && kernels.hc_pre_norm_tok.0 != 0)
        .then_some(kernels.hc_pre_norm_tok)
}

/// 2026-10-12: One launch of `kernel` (`glm5next_hc_pre_norm_tok4`) over `num_tokens` tokens:
/// `glm5next_hc_pre_norm`'s arguments, then the token count, on `ceil(num_tokens / 4)` blocks.
/// `outs` is `[y, post, comb, normed]`, `dims` `[hidden, hc_mult, sinkhorn_iters]`, `eps`
/// `[norm_eps, hc_eps, rms_eps]`.
#[allow(clippy::too_many_arguments)]
pub fn launch_tok(
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
        .grid([num_tokens.div_ceil(HC_PRE_NORM_TOK), 1, 1])
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
        .arg_u32(num_tokens)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glm5next_mhc_fused::glm_hc_pre_norm_with;
    use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};

    fn site() -> Glm5NextMhcSiteWeights {
        Glm5NextMhcSiteWeights {
            hc_fn: DevicePtr(0x100),
            hc_fn_bf16: true,
            hc_scale: DevicePtr(0x200),
            hc_base: DevicePtr(0x300),
            mix: DevicePtr(0x400),
            done: DevicePtr(0x500),
            mix_tokens: crate::glm5next_mhc::MHC_MIX_MAX_TOKENS,
        }
    }

    fn run(gpu: &MockGpuBackend, k: &Glm5NextMhcKernels, t: u32, tok: bool) -> (u64, [u32; 3]) {
        let from = gpu.launch_count();
        glm_hc_pre_norm_with(
            gpu,
            k,
            DevicePtr(0x1000),
            &site(),
            DevicePtr(0x2000),
            DevicePtr(0x3000),
            DevicePtr(0x4000),
            DevicePtr(0x5000),
            DevicePtr(0x6000),
            t,
            4096,
            4,
            20,
            1e-6,
            1e-6,
            1e-5,
            crate::glm5next_mhc_fused::PreNormChoice {
                prefetch: false,
                splitk: false,
                tok,
                c1: false,
            },
            7,
        )
        .unwrap();
        assert_eq!(gpu.launch_count(), from + 1, "one launch per site");
        let l = gpu.launches_snapshot()[from].clone();
        (l.func, l.grid)
    }

    /// 2026-10-12: The token-tiled grid needs the lever, the kernel and the 8-row grid's token
    /// count; anything short of that keeps the 8-row or per-row grid unchanged.
    #[test]
    fn the_token_tiled_grid_needs_the_lever_the_kernel_and_wide_rows() {
        let gpu = MockGpuBackend::new();
        let k =
            Glm5NextMhcKernels::resolve(&gpu, crate::glm5next_mhc_levers::MhcLevers::declared())
                .expect("mock resolves every kernel");
        let min = HC_PRE_NORM_ROWS_MIN_TOKENS as u32;
        let rows = k.hc_pre_norm_rows.0;
        assert_eq!(run(&gpu, &k, 250, true), (k.hc_pre_norm_tok.0, [63, 1, 1]));
        assert_eq!(
            run(&gpu, &k, min, true),
            (k.hc_pre_norm_tok.0, [min / 4, 1, 1])
        );
        assert_eq!(run(&gpu, &k, 250, false), (rows, [250, 3, 1]));
        assert_eq!(run(&gpu, &k, min - 1, true).0, k.hc_pre_norm.0);
        let mut absent = k;
        absent.hc_pre_norm_tok = KernelHandle(0);
        assert_eq!(run(&gpu, &absent, 250, true), (rows, [250, 3, 1]));
    }

    /// 2026-10-12: The kernel takes `glm5next_hc_pre_norm`'s arguments in order, then the token
    /// count; a dropped or swapped pointer would read the wrong buffer with no error.
    #[test]
    fn the_token_tiled_launch_passes_the_pre_norm_arguments_then_the_tokens() {
        let gpu = MockGpuBackend::new();
        let k =
            Glm5NextMhcKernels::resolve(&gpu, crate::glm5next_mhc_levers::MhcLevers::declared())
                .expect("mock resolves every kernel");
        let from = gpu.launch_count();
        run(&gpu, &k, 130, true);
        let l = gpu.launches_snapshot()[from].clone();
        let ptrs = [
            0x1000u64, 0x100, 0x200, 0x300, 0x400, 0x500, 0x2000, 0x3000, 0x5000, 0x6000, 0x4000,
        ];
        for (i, p) in ptrs.iter().enumerate() {
            assert_eq!(
                l.args[i],
                MockArg::Buffer(DevicePtr(*p)),
                "pointer argument {i}"
            );
        }
        let u32s = |v: u32| MockArg::Bytes(v.to_le_bytes().to_vec());
        assert_eq!(l.args[11], u32s(4096));
        assert_eq!(l.args[12], u32s(4));
        assert_eq!(l.args[13], u32s(20));
        assert_eq!(l.args[17], u32s(130), "the token count closes the list");
        assert_eq!(l.args.len(), 18);
    }
}
