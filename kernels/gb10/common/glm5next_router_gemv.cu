// SPDX-License-Identifier: MIT OR Apache-2.0
// metrale: instantiates dense_bf16 (dense_gemv_bf16_batchm_few.cuh)

// 2026-10-23: The GLM-5.3 MoE router's 2..8-row GEMV (288 x 4096, FP32 logits) as the few-row
// template at one output per block (288 blocks of 64 threads instead of 72 of 256 on GB10's 48
// SMs). The bits are dense_gemv_bf16_batchm_few_fp32out's (NPB places outputs only;
// glm5next_batchm_few_gate checks them byte for byte). Kept as the paired A/B instrument's
// known-small validation lever `router_npb1` (crates/model-arch/src/glm5next_layer/ab_levers.rs):
// 0.13 us per call in isolation.
//
// Owner: gb10 kernels (GLM-5.3).
// Invariants: launch grid (N, 1, 1), block (64, 1, 1), 2 <= M <= 8.
#include "dense_gemv_bf16_batchm_few.cuh"

extern "C" __global__ void __launch_bounds__(bmfew::TPO) glm5next_router_gemv_npb1(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride
) {
    bmfew::few_dispatch<1u, float>(A, B, C, M, N, K, out_stride, blockIdx.x);
}
