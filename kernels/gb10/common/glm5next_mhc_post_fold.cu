// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-10: glm5next_hc_post_pre_norm: one token's mHC post-mix of the previous site (with the
// one-shot all-reduce's sum in front when `n_ranks` > 0) and the next site's glm5next_hc_pre_norm
// (glm5next_mhc_fused.cu), in one launch. Unfused, the decode runs bf16_add_rank_sum (or nothing, when the sum is already
// in `own`), glm5next_hc_post (streams updated in place) and glm5next_hc_pre_norm.
//
// Each block (row m) computes the post-mixed streams its 256 threads sum, in registers: thread
// tid owns k = tid + 256 jj, i.e. column d = tid + 256 c of every stream, and for each such d
// forms x[d] with bf16_add_rank_sum's arithmetic (FP32 sum in rank order from 0.0f, one BF16
// rounding; or `own` as it is) and the hc new values with glm5next_hc_post's (post[j] * x then
// + comb[i][j] * residual[i][d], i ascending). It then runs glm5next_hc_pre_norm's mix on them in
// its order (k ascending: stream-major, column-minor) and its trees, tail and norm, the tail
// reading the new streams back through L2. Block row 0 also stores the new streams to
// `streams_out`, which must not alias `residual` (every block reads `residual` while row 0 writes),
// before its arrival, so the tail sees them. `post`/`comb` may be the buffers the tail writes:
// every block reads them before it arrives. The summed x is not stored back to `own`.
//
// Invariants: one token, grid (1, mix_hc, 1), block GLM_HCF_BLOCK; H a multiple of 256 and at
// most GLM_HCF_POST_FOLD_MAX_H (otherwise nothing is written); hc_mult <= 4; `peers` holds peer r
// in slot (r < my_rank ? r : r - 1) of `slot_elems` elements; `done` as for glm5next_hc_pre_norm.
//
// Owner: gb10 kernels.

#include "glm5next_mhc_fused_parts.cuh"

#define GLM_HCF_POST_FOLD_MAX_H 4096

template <int HC>
__device__ __forceinline__ void glm_hcf_post_pre_body(
    const __nv_bfloat16* __restrict__ own,
    const __nv_bfloat16* __restrict__ peers,
    long long slot_elems,
    int n_ranks,
    int my_rank,
    const float* __restrict__ residual,
    const float* __restrict__ p_post,
    const float* __restrict__ p_comb,
    float* __restrict__ streams_out,
    const __nv_bfloat16* __restrict__ hc_fn,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    float* __restrict__ mix,
    unsigned int* __restrict__ done,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    __nv_bfloat16* __restrict__ normed_out,
    unsigned int H,
    unsigned int sinkhorn_iters,
    float norm_eps,
    float hc_eps,
    float rms_eps,
    GlmHcfSmem<1>& sm
) {
    constexpr unsigned int MIX = (2 + HC) * HC;
    constexpr unsigned int MAXC = GLM_HCF_POST_FOLD_MAX_H / GLM_HCF_BLOCK;
    const unsigned int m = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int hc_dim = HC * H;
    const unsigned int cols = H / GLM_HCF_BLOCK;
    __shared__ float s_p[GLM_HCF_MAX_MULT];
    __shared__ float s_c[GLM_HCF_MAX_MULT * GLM_HCF_MAX_MULT];
    if (tid < HC) s_p[tid] = p_post[tid];
    if (tid < HC * HC) s_c[tid] = p_comb[tid];
    __syncthreads();

    float nv[HC][MAXC];
#pragma unroll
    for (unsigned int c = 0; c < MAXC; ++c) {
        if (c >= cols) break;
        const unsigned int d = tid + GLM_HCF_BLOCK * c;
        float xd;
        if (n_ranks > 0) {
            float acc = 0.0f;
            for (int r = 0; r < n_ranks; ++r) {
                const __nv_bfloat16 v =
                    (r == my_rank) ? own[d]
                                   : peers[(long long)(r < my_rank ? r : r - 1) * slot_elems + d];
                acc += __bfloat162float(v);
            }
            xd = (float)__float2bfloat16_rn(acc);
        } else {
            xd = (float)own[d];
        }
        float rv[HC];
#pragma unroll
        for (int i = 0; i < HC; ++i) rv[i] = residual[i * H + d];
#pragma unroll
        for (int j = 0; j < HC; ++j) {
            float a = s_p[j] * xd;
#pragma unroll
            for (int i = 0; i < HC; ++i) a += s_c[i * HC + j] * rv[i];
            nv[j][c] = a;
            if (m == 0) streams_out[j * H + d] = a;
        }
    }

    float ss = 0.f;
    float acc = 0.f;
    const __nv_bfloat16* fn = hc_fn + (size_t)m * hc_dim;
#pragma unroll
    for (int j = 0; j < HC; ++j) {
#pragma unroll
        for (unsigned int c = 0; c < MAXC; ++c) {
            if (c >= cols) break;
            const float v = nv[j][c];
            ss += v * v;
            acc += __bfloat162float(fn[j * H + tid + GLM_HCF_BLOCK * c]) * v;
        }
    }
    sm.red_ss[tid] = ss;
    sm.red[0][tid] = acc;
    __syncthreads();
    for (unsigned int s = GLM_HCF_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sm.red_ss[tid] += sm.red_ss[tid + s];
            sm.red[0][tid] += sm.red[0][tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const float rsq = rsqrtf(sm.red_ss[0] / (float)hc_dim + norm_eps);
        mix[m] = sm.red[0][0] * rsq;
        __threadfence();
        const unsigned int prev = atomicAdd(&done[0], 1u);
        sm.last = (prev == gridDim.y - 1);
    }
    __syncthreads();
    if (!sm.last) return;

    __threadfence();
    if (tid == 0) done[0] = 0;
    if (tid < MIX) sm.mix[tid] = __ldcg(&mix[tid]);
    __syncthreads();
    if (tid < HC) {
        float pr = sm.mix[tid] * hc_scale[0] + hc_base[tid];
        sm.pre[tid] = 1.f / (1.f + expf(-pr)) + hc_eps;
    }
    __syncthreads();
    if (tid < 32) {
        glm_hcf_sinkhorn<HC>(sm.mix, hc_scale, hc_base, post_out, comb_out, 0, sinkhorn_iters,
                             hc_eps);
    } else {
        glm_hcf_collapse_norm<HC, true>(streams_out, sm.pre, norm_w, y_out, normed_out, sm.y,
                                        sm.ws, H, rms_eps);
    }
}

extern "C" __global__ void __launch_bounds__(GLM_HCF_BLOCK) glm5next_hc_post_pre_norm(
    const __nv_bfloat16* __restrict__ own, const __nv_bfloat16* __restrict__ peers,
    long long slot_elems, int n_ranks, int my_rank, const float* __restrict__ residual,
    const float* __restrict__ p_post, const float* __restrict__ p_comb,
    float* __restrict__ streams_out, const __nv_bfloat16* __restrict__ hc_fn,
    const float* __restrict__ hc_scale, const float* __restrict__ hc_base,
    float* __restrict__ mix, unsigned int* __restrict__ done,
    const __nv_bfloat16* __restrict__ norm_w, __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ post_out, float* __restrict__ comb_out,
    __nv_bfloat16* __restrict__ normed_out, const unsigned int hidden_size,
    const unsigned int hc_mult, const unsigned int sinkhorn_iters, const float norm_eps,
    const float hc_eps, const float rms_eps) {
    __shared__ GlmHcfSmem<1> sm;
    if (hidden_size % GLM_HCF_BLOCK != 0 || hidden_size > GLM_HCF_POST_FOLD_MAX_H) return;
#define GLM_HCF_PP_PASS                                                                       \
    own, peers, slot_elems, n_ranks, my_rank, residual, p_post, p_comb, streams_out, hc_fn,   \
        hc_scale, hc_base, mix, done, norm_w, y_out, post_out, comb_out, normed_out,          \
        hidden_size, sinkhorn_iters, norm_eps, hc_eps, rms_eps, sm
    switch (hc_mult) {
    case 1: glm_hcf_post_pre_body<1>(GLM_HCF_PP_PASS); break;
    case 2: glm_hcf_post_pre_body<2>(GLM_HCF_PP_PASS); break;
    case 3: glm_hcf_post_pre_body<3>(GLM_HCF_PP_PASS); break;
    case 4: glm_hcf_post_pre_body<4>(GLM_HCF_PP_PASS); break;
    default: break;
    }
#undef GLM_HCF_PP_PASS
}
