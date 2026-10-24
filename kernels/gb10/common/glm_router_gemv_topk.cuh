// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-24: R2, the GLM-5.3 MoE router of 2..8 rows and its top-k in one launch: the few-row
// BF16 GEMV (dense_gemv_bf16_batchm_few.cuh, FP32 logits, NPB outputs a block) and, in the last
// block to finish, glm5next_router_topk_rank's body (glm_router_topk_rank.cuh) over every row, one
// named-barrier thread group a row. Replaces dense_gemv_bf16_batchm_few_fp32out followed by
// glm5next_router_topk_rank: one launch and one stream edge fewer per MoE layer.
//
// Owner: gb10 kernels (GLM-5.3).
// Invariants:
// - The logits are the GEMV's bytes (written as before; few_body is unchanged) and the ids and
//   weights are glm5next_router_topk_rank's (the body's group size only strides the experts).
// - Completion: each block fences its stores, then takes a ticket from `ctr` with atomicInc
//   wrapping at gridDim.x - 1; the block that takes the last ticket runs the top-k after a
//   fence, and the wrap leaves `ctr` at 0 for the next launch (graph replay needs no reset). No
//   block waits on another, so any residency is safe. `ctr` starts at 0 and one launch at a time
//   uses it (the stream orders the layers' launches).
// - Grid (ceil(N / NPB), 1, 1), block (64 * NPB, 1, 1) with 64 * NPB / M >= 32; 2 <= M <= 8,
//   N <= GLM_RGT_MAX_EXPERTS, top_k <= 16.
#pragma once

#include "dense_gemv_bf16_batchm_few.cuh"
#include "glm_router_topk_rank.cuh"

#define GLM_RGT_MAX_EXPERTS 512

namespace glm_router_fused {

template <unsigned int NPB>
__device__ __forceinline__ void gemv_topk(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ logits,
    const float* __restrict__ bias,
    int* __restrict__ ids,
    float* __restrict__ wts,
    unsigned int* __restrict__ ctr,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int top_k,
    float routed_scale,
    unsigned int renormalize,
    unsigned int bf16_ladder
) {
    __shared__ glm_router_rank::Smem<GLM_RGT_MAX_EXPERTS> sm[bmfew::FEW_MAX_M];
    __shared__ bool last;
    bmfew::few_dispatch<NPB, float>(A, B, logits, M, N, K, N, blockIdx.x);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicInc(ctr, gridDim.x - 1u) == gridDim.x - 1u;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const unsigned int gsize = (blockDim.x / M) & ~31u;
    const unsigned int r = threadIdx.x / gsize;
    if (r >= M) return;
    const glm_router_rank::BarGroup g{r * gsize, gsize, 1u + r};
    glm_router_rank::topk_rank_body<GLM_RGT_MAX_EXPERTS>(
        g, glm_router_rank::LdCg{}, sm[r], logits + (size_t)r * N, bias, ids + (size_t)r * top_k,
        wts + (size_t)r * top_k, N, top_k, 1u, routed_scale, renormalize, bf16_ladder);
}

}  // namespace glm_router_fused
