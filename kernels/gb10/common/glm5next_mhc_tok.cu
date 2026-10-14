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

// 2026-10-14: CG reads the streams through L2 (__ldcg) in the mix and the collapse, for
// glm5next_hc_post_pre_norm_tok4, whose post-mix wrote them earlier in the same launch; the values
// read, and so every output, are the same.
template <int HC, bool CG = false>
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
            v[q] = ((unsigned int)q < nt)
                       ? (CG ? __ldcg(&xb[(size_t)q * hc_dim + k]) : xb[(size_t)q * hc_dim + k])
                       : 0.f;
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
            glm_hcf_collapse_norm<HC, CG>(xb + (size_t)q * hc_dim, sm.pre, norm_w,
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

// 2026-10-14: glm5next_hc_post_pre_norm_tok4 (METRALE_GLM_MHC_POST_PRE_FUSED, host side in
// model-arch glm5next_mhc_post_pre_tok.rs): glm5next_hc_post_rank_sum (glm5next_hc_post_sum.cu)
// over a block's GLM_HCF_TOK tokens, then glm5next_hc_pre_norm_tok4's body on the streams it
// wrote, in one launch. Unfused, the SP-mHC layer runs the two launches back to back between the
// mixer and the MLP.
//
// Byte-identical to that pair: thread tid forms column d = tid + 256 c of each token with
// glm5next_hc_post_rank_sum's expressions in its order (x the FP32 rank-order sum of the BF16
// partials from 0.0f, rounded to BF16 once and stored to `block_out`; stream j's new value
// post[j] * x, then + comb[i][j] * residual[i][d] for i ascending) and stores the new streams.
// This directory compiles with --fmad=false (KERNEL.toml), so neither kernel contracts a product
// into an add: each product and sum rounds as written, in the same order;
// after the block barrier the tok4 body reads them back through L2 and runs its chains, trees and
// tails unchanged. The streams cannot stay in registers (64 floats per token and thread, four
// tokens, beside the 96 mix accumulators), and summing them as they are formed would visit k
// out of the mix's stream-major order, so they go through memory; the read-back mostly hits L2.
//
// Invariants: blockDim 256, grid (ceil(T / 4), 1, 1), the shapes glm5next_hc_pre_norm_tok4 takes;
// `peers` as for glm5next_hc_post_rank_sum (peer r in slot (r < my_rank ? r : r - 1) of
// `slot_elems` elements, element (t, d) at t * H + d); `n_ranks` >= 1. `streams` may be
// `residual` (column d of a token is read, then written, by one thread), and `post`/`comb` may be
// `post_out`/`comb_out` (a block reads its tokens' into shared memory before any tail writes
// them; no other block touches those tokens), so none of those is __restrict__.
struct GlmHcfPostTokSmem {
    float p[GLM_HCF_TOK][GLM_HCF_MAX_MULT];
    float c[GLM_HCF_TOK][GLM_HCF_MAX_MULT * GLM_HCF_MAX_MULT];
};

template <int HC>
__device__ __forceinline__ void glm_hcf_post_tok(
    __nv_bfloat16* __restrict__ block_out, const __nv_bfloat16* __restrict__ peers,
    long long slot_elems, int n_ranks, int my_rank, const float* residual, const float* p_post,
    const float* p_comb, float* streams, unsigned int H, unsigned int num_tokens,
    GlmHcfPostTokSmem& ps) {
    const unsigned int tid = threadIdx.x;
    const unsigned int t0 = blockIdx.x * GLM_HCF_TOK;
    if (t0 >= num_tokens) return;
    const unsigned int nt = min((unsigned int)GLM_HCF_TOK, num_tokens - t0);
    if (tid < nt * HC) ps.p[tid / HC][tid % HC] = p_post[(size_t)t0 * HC + tid];
    if (tid < nt * HC * HC)
        ps.c[tid / (HC * HC)][tid % (HC * HC)] = p_comb[(size_t)t0 * HC * HC + tid];
    __syncthreads();

    for (unsigned int q = 0; q < nt; ++q) {
        const unsigned int t = t0 + q;
        __nv_bfloat16* x = block_out + (size_t)t * H;
        const float* res = residual + (size_t)t * HC * H;
        float* o = streams + (size_t)t * HC * H;
#pragma unroll 4
        for (unsigned int d = tid; d < H; d += GLM_HCF_BLOCK) {
            const long long e = (long long)t * H + d;
            // glm5next_hc_post_rank_sum's arithmetic for element e and column d.
            float acc = 0.0f;
            for (int r = 0; r < n_ranks; ++r) {
                __nv_bfloat16 v =
                    (r == my_rank) ? x[d]
                                   : peers[(long long)(r < my_rank ? r : r - 1) * slot_elems + e];
                acc += __bfloat162float(v);
            }
            const __nv_bfloat16 xb = __float2bfloat16_rn(acc);
            x[d] = xb;
            float xd = (float)xb;
            float rv[HC];
#pragma unroll
            for (int i = 0; i < HC; ++i) rv[i] = res[i * H + d];
#pragma unroll
            for (int j = 0; j < HC; ++j) {
                float a = ps.p[q][j] * xd;
#pragma unroll
                for (int i = 0; i < HC; ++i) a += ps.c[q][i * HC + j] * rv[i];
                o[j * H + d] = a;
            }
        }
    }
    // 2026-10-14: Every new stream of the block's tokens is stored before any thread reads one
    // back (the tail's collapse reads columns other threads wrote).
    __syncthreads();
}

extern "C" __global__ void __launch_bounds__(GLM_HCF_BLOCK) glm5next_hc_post_pre_norm_tok4(
    __nv_bfloat16* __restrict__ block_out, const __nv_bfloat16* __restrict__ peers,
    long long slot_elems, int n_ranks, int my_rank, const float* residual, const float* p_post,
    const float* p_comb, float* streams, const __nv_bfloat16* __restrict__ hc_fn,
    const float* __restrict__ hc_scale, const float* __restrict__ hc_base,
    float* __restrict__ mix, unsigned int* __restrict__ done,
    const __nv_bfloat16* __restrict__ norm_w, __nv_bfloat16* __restrict__ y_out,
    float* post_out, float* comb_out, __nv_bfloat16* __restrict__ normed_out,
    const unsigned int hidden_size, const unsigned int hc_mult, const unsigned int sinkhorn_iters,
    const float norm_eps, const float hc_eps, const float rms_eps, const unsigned int num_tokens) {
    __shared__ GlmHcfTokSmem sm;
    __shared__ GlmHcfPostTokSmem ps;
#define GLM_HCF_POST_TOK_PASS                                                                 \
    block_out, peers, slot_elems, n_ranks, my_rank, residual, p_post, p_comb, streams,        \
        hidden_size, num_tokens, ps
    switch (hc_mult) {
    case 1:
        glm_hcf_post_tok<1>(GLM_HCF_POST_TOK_PASS);
        glm_hcf_tok_body<1, true>(GLM_HCF_TOK_PASS);
        break;
    case 2:
        glm_hcf_post_tok<2>(GLM_HCF_POST_TOK_PASS);
        glm_hcf_tok_body<2, true>(GLM_HCF_TOK_PASS);
        break;
    case 3:
        glm_hcf_post_tok<3>(GLM_HCF_POST_TOK_PASS);
        glm_hcf_tok_body<3, true>(GLM_HCF_TOK_PASS);
        break;
    case 4:
        glm_hcf_post_tok<4>(GLM_HCF_POST_TOK_PASS);
        glm_hcf_tok_body<4, true>(GLM_HCF_TOK_PASS);
        break;
    default: break;
    }
#undef GLM_HCF_POST_TOK_PASS
}
