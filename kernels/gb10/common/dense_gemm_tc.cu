// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: BF16 tensor-core GEMM, C[M, N] = A[M, K] * B[N, K]^T with FP32 accumulation, on m16n8k16 MMA.
//
// `dense_gemm_tc` stores C; `dense_gemm_tc_scaled_acc` folds C += scale * bf16(A * B^T) for the LoRA path
// (crates/model-layers/src/layers/ops/lora_delta.rs). A block computes a 16x64 tile of C with 4 warps, each 16x16 as
// two 16x8 MMAs, and walks K in steps of 16. Grid (ceil(N/64), ceil(M/16)), block 128.
//
// 2026-10-17: The mainloop lives in dense_gemm_tc_body.cuh, shared with the FP32-output entry
// (dense_gemm_tc_f32out.cu).
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.

#include "dense_gemm_tc_body.cuh"

extern "C" __global__ void dense_gemm_tc(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    dense_gemm_tc_body<false>(A, B, C, M, N, K, 0.0f);
}

// 2026-09-25: C[M, N] += scale * bf16(A[M, K] * B[N, K]^T): the LoRA expand and fold in one pass.
extern "C" __global__ void dense_gemm_tc_scaled_acc(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    float scale
) {
    dense_gemm_tc_body<true>(A, B, C, M, N, K, scale);
}
