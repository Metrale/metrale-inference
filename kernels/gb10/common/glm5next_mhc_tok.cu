// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-12: GLM-5.3 mHC pre-norm for wide prefill sub-chunks, token-tiled
// (METRALE_GLM_MHC_PRE_TOK, host side in model-arch glm5next_mhc_tok.rs): four tokens per block,
// the mix and then each token's tail.
//
// Owner: gb10 kernels.
// Invariants:
// - Every output is byte-identical to glm5next_hc_pre_norm (glm5next_mhc_fused.cu): the same
//   per-thread chains, trees, rsq and tail code (examples/glm5next_mhc_slice_bench.rs checks it
//   against the 8-row grid, which glm5next_hc_fused_gate ties to the unfused kernels).
// - blockDim 256; the shapes glm5next_mhc_fused.cu admits; grid (ceil(T / 4), 1, 1).

#include "glm5next_mhc_fused_parts.cuh"

// 2026-10-12: glm5next_hc_pre_norm for wide prefill sub-chunks: one block per GLM_HCF_TOK tokens
// and all (2 + hc) * hc mix rows, grid (ceil(T / GLM_HCF_TOK), 1, 1), `num_tokens` = T. The 8-row
// grid reads each token's FP32 streams once per 8 mix rows and the BF16 hc_fn once per token;
// here each stream value is read once and every hc_fn load serves GLM_HCF_TOK tokens.
// Byte-identical to glm5next_hc_pre_norm: thread tid forms, for each token and mix row, the same
// chain (k = tid, tid + 256, .. ascending; ss += v * v, acc += w * v) and the RMS and mix sums go
// through the same 256-wide shared-memory tree, token by token; the mix is red * rsq as there,
// and each token's tail (pre, Sinkhorn on warp 0, collapse + norm on warps 1..7) is
// glm_hcf_body's, run in this block token by token. `done` is not used; `mix` gets the same values.
// 2026-10-12, GB10 (H 4096, hc 4): 1.36 ms vs the 8-row grid's 2.08 ms at 2048 tokens (1.5x),
// 0.36 vs 0.55 ms at 512; slower below ~128 tokens (66 vs 47 us at 64), where the four tails of a
// block run one after another. Eight tokens per 512-thread block (two groups of 12 mix rows)
// measured 0.8x of the 8-row grid and was dropped.
#define GLM_HCF_TOK 4
struct GlmHcfTokSmem {
    float red[GLM_HCF_MAX_MIX + 1][GLM_HCF_BLOCK];
    float mix[GLM_HCF_TOK][GLM_HCF_MAX_MIX];
    float pre[GLM_HCF_MAX_MULT];
    float ws[33];
    __nv_bfloat16 y[GLM_HCF_MAX_H];
};

template <int HC>
__device__ __forceinline__ void glm_hcf_tok_body(GLM_HCF_ARGS, unsigned int num_tokens,
                                                 GlmHcfTokSmem& sm) {
    constexpr unsigned int MIX = (2 + HC) * HC;
    const unsigned int H = hidden_size;
    const unsigned int tid = threadIdx.x;
    const unsigned int t0 = blockIdx.x * GLM_HCF_TOK;
    if (t0 >= num_tokens) return;
    const unsigned int nt = min((unsigned int)GLM_HCF_TOK, num_tokens - t0);
    const unsigned int hc_dim = HC * H;
    const float* xb = streams + (size_t)t0 * hc_dim;

    float ss[GLM_HCF_TOK];
    float acc[GLM_HCF_TOK][MIX];
#pragma unroll
    for (int q = 0; q < GLM_HCF_TOK; ++q) {
        ss[q] = 0.f;
#pragma unroll
        for (int r = 0; r < (int)MIX; ++r) acc[q][r] = 0.f;
    }
#pragma unroll 2
    for (unsigned int k = tid; k < hc_dim; k += GLM_HCF_BLOCK) {
        float v[GLM_HCF_TOK];
#pragma unroll
        for (int q = 0; q < GLM_HCF_TOK; ++q)
            v[q] = ((unsigned int)q < nt) ? xb[(size_t)q * hc_dim + k] : 0.f;
#pragma unroll
        for (int q = 0; q < GLM_HCF_TOK; ++q) ss[q] += v[q] * v[q];
#pragma unroll
        for (int r = 0; r < (int)MIX; ++r) {
            const float w = __bfloat162float(hc_fn[(size_t)r * hc_dim + k]);
#pragma unroll
            for (int q = 0; q < GLM_HCF_TOK; ++q) acc[q][r] += w * v[q];
        }
    }

    // 2026-10-12: Token by token: the RMS partial in row 0 and the mixes in rows 1.. of `red`,
    // then glm_hcf_body's tree (s = 128 .. 1, `red[tid] += red[tid + s]`) on every row.
#pragma unroll
    for (int q = 0; q < GLM_HCF_TOK; ++q) {
        sm.red[0][tid] = ss[q];
#pragma unroll
        for (int r = 0; r < (int)MIX; ++r) sm.red[1 + r][tid] = acc[q][r];
        __syncthreads();
        for (unsigned int s = GLM_HCF_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) {
#pragma unroll
                for (int r = 0; r <= (int)MIX; ++r) sm.red[r][tid] += sm.red[r][tid + s];
            }
            __syncthreads();
        }
        if (tid < MIX) {
            const float rsq = rsqrtf(sm.red[0][0] / (float)hc_dim + norm_eps);
            const float m = sm.red[1 + tid][0] * rsq;
            sm.mix[q][tid] = m;
            if ((unsigned int)q < nt) mix[(size_t)(t0 + q) * MIX + tid] = m;
        }
        __syncthreads();
    }

    // 2026-10-12: Each token's tail in this block, right after the mix read its streams, so the
    // collapse finds them in L2 (a separate one-block-per-token tail kernel measured 1.93 ms
    // against 1.36 ms here at 2048 tokens on GB10).
    for (unsigned int q = 0; q < nt; ++q) {
        const unsigned int t = t0 + q;
        if (tid < HC) {
            float pr = sm.mix[q][tid] * hc_scale[0] + hc_base[tid];
            sm.pre[tid] = 1.f / (1.f + expf(-pr)) + hc_eps;
        }
        __syncthreads();
        if (tid < 32) {
            glm_hcf_sinkhorn<HC>(sm.mix[q], hc_scale, hc_base, post_out, comb_out, t,
                                 sinkhorn_iters, hc_eps);
        } else {
            glm_hcf_collapse_norm<HC>(xb + (size_t)q * hc_dim, sm.pre, norm_w,
                                      y_out + (size_t)t * H, normed_out + (size_t)t * H, sm.y,
                                      sm.ws, H, rms_eps);
        }
        __syncthreads();
    }
}

#define GLM_HCF_TOK_PASS                                                                      \
    streams, hc_fn, hc_scale, hc_base, mix, done, norm_w, y_out, post_out, comb_out,          \
        normed_out, hidden_size, hc_mult, sinkhorn_iters, norm_eps, hc_eps, rms_eps,          \
        num_tokens, sm

extern "C" __global__ void __launch_bounds__(GLM_HCF_BLOCK)
glm5next_hc_pre_norm_tok4(GLM_HCF_ARGS, const unsigned int num_tokens) {
    __shared__ GlmHcfTokSmem sm;
    switch (hc_mult) {
    case 1: glm_hcf_tok_body<1>(GLM_HCF_TOK_PASS); break;
    case 2: glm_hcf_tok_body<2>(GLM_HCF_TOK_PASS); break;
    case 3: glm_hcf_tok_body<3>(GLM_HCF_TOK_PASS); break;
    case 4: glm_hcf_tok_body<4>(GLM_HCF_TOK_PASS); break;
    default: break;
    }
}
