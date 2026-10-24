// SPDX-License-Identifier: MIT OR Apache-2.0
// metrale: instantiates dense_bf16 (glm_router_gemv_topk.cuh)

// 2026-10-24: glm5next_router_gemv_topk_few (R2, opt-in --glm-router-fused): the GLM-5.3 MoE
// router GEMV of 2..8 rows (FP32 logits, four outputs a block as dense_gemv_bf16_batchm_few_fp32out)
// and glm5next_router_topk_rank's ids and weights in one launch (glm_router_gemv_topk.cuh).
//
// Owner: gb10 kernels (GLM-5.3).
// Invariants: as glm_router_gemv_topk.cuh's; grid (ceil(N / 4), 1, 1), block (256, 1, 1).
#include "glm_router_gemv_topk.cuh"

extern "C" __global__ void __launch_bounds__(bmfew::BLOCK) glm5next_router_gemv_topk_few(
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
    glm_router_fused::gemv_topk<bmfew::N_PER_BLOCK>(
        A, B, logits, bias, ids, wts, ctr, M, N, K, top_k, routed_scale, renormalize, bf16_ladder);
}
