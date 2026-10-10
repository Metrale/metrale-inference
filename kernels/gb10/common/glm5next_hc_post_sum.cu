// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-10: glm5next_hc_post_rank_sum: the reduce step of the one-shot all-reduce
// (bf16_add_rank_sum) and the mHC post-mix after it (glm5next_hc_post) in one launch.
//
// Owner: gb10 kernels.
// Invariants:
// - Byte-identical to bf16_add_rank_sum (bf16_add.cu) followed by glm5next_hc_post
//   (glm5next_mhc.cu) on the same inputs: x is the FP32 sum of the ranks' partials in rank
//   order 0..n_ranks-1 from 0.0f, rounded to BF16 once; the post-mix then widens that BF16 value
//   and runs glm5next_hc_post's loops and order. Both outputs are written: `block_out` gets x (what
//   bf16_add_rank_sum leaves there) and `out` the post-mix. glm5next_hc_post_sum_gate (model-arch
//   examples) checks both byte for byte.
// - `block_out` holds this rank's partial, `peers` the others', one slot of `slot_elems` elements
//   per peer, peer r in slot (r < my_rank ? r : r - 1); element (t, d) is at t * H + d in both.
// - blockDim == 256, hc_mult <= 4, grid (T, NB) as glm5next_hc_post; `out` may alias `residual`
//   (same reasoning as glm5next_hc_post: column d is read and written by one thread only).
// Callers: model-arch glm5next_mhc.rs glm_hc_post_rank_sum, opt-in METRALE_GLM_AR_FUSED_POST.

#include <cuda_bf16.h>

#define GLM_HCPS_BLOCK 256
#define GLM_HCPS_MAX_MULT 4

extern "C" __global__ void glm5next_hc_post_rank_sum(
    __nv_bfloat16* __restrict__ block_out,
    const __nv_bfloat16* __restrict__ peers,
    long long slot_elems,
    int n_ranks,
    int my_rank,
    const float* __restrict__ residual,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    float* __restrict__ out,
    const unsigned int hidden_size,
    const unsigned int hc_mult
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;

    __nv_bfloat16* x = block_out + (size_t)t * H;
    const float* res = residual + (size_t)t * hc * H;
    const float* p = post + (size_t)t * hc;
    const float* c = comb + (size_t)t * hc * hc;
    float* o = out + (size_t)t * hc * H;

    __shared__ float s_p[GLM_HCPS_MAX_MULT];
    __shared__ float s_c[GLM_HCPS_MAX_MULT * GLM_HCPS_MAX_MULT];
    if (tid < hc) s_p[tid] = p[tid];
    if (tid < hc * hc) s_c[tid] = c[tid];
    __syncthreads();

    const unsigned int stride = gridDim.y * GLM_HCPS_BLOCK;
    for (unsigned int d = blockIdx.y * GLM_HCPS_BLOCK + tid; d < H; d += stride) {
        const long long e = (long long)t * H + d;
        // bf16_add_rank_sum's arithmetic for element e.
        float acc = 0.0f;
        for (int r = 0; r < n_ranks; ++r) {
            __nv_bfloat16 v = (r == my_rank) ? x[d]
                : peers[(long long)(r < my_rank ? r : r - 1) * slot_elems + e];
            acc += __bfloat162float(v);
        }
        const __nv_bfloat16 xb = __float2bfloat16_rn(acc);
        x[d] = xb;
        // glm5next_hc_post's arithmetic for column d.
        float xd = (float)xb;
        float rv[GLM_HCPS_MAX_MULT];
#pragma unroll
        for (unsigned int i = 0; i < GLM_HCPS_MAX_MULT; ++i)
            if (i < hc) rv[i] = res[i * H + d];
#pragma unroll
        for (unsigned int j = 0; j < GLM_HCPS_MAX_MULT; ++j) {
            if (j < hc) {
                float a = s_p[j] * xd;
#pragma unroll
                for (unsigned int i = 0; i < GLM_HCPS_MAX_MULT; ++i)
                    if (i < hc) a += s_c[i * hc + j] * rv[i];
                o[j * H + d] = a;
            }
        }
    }
}
