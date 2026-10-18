// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-17: The dense_gemm_tc mainloop, parameterized by the output type: C[M, N] = A[M, K] * B[N, K]^T with
// FP32 accumulation on m16n8k16 MMA, a 16x64 tile of C per block, 4 warps each 16x16 as two 16x8 MMAs, K in steps of
// 16. `dense_gemm_tc_body<ACC>` has a BF16-output overload (dense_gemm_tc.cu) and an FP32-output overload
// (dense_gemm_tc_f32out.cu), both from the one body in dense_gemm_tc_mainloop.cuh.
//
// Owner: gb10 kernels.
// Invariants:
// - A is [M, K] contiguous, B is [N, K] row-major, C is [M, N] contiguous; any M, N, K (a tile edge past M, N or K
//   loads zeros and stores nothing).
// - Grid (ceil(N / TC_TN), ceil(M / TC_TM)), block TC_BLOCK, static shared memory only.
// - The accumulation order depends on K alone, never on the output type: the FP32 overload stores the accumulator the
//   BF16 overload rounds, so bf16(C_f32) == C_bf16 bit for bit.
// - ACC (the scaled fold) exists for BF16 output only; the FP32 store rejects it at compile time.

#pragma once

#include <cuda_bf16.h>

#define TC_TM 16
#define TC_TN 64
#define TC_TK 16
#define TC_PAD 8
#define TC_BLOCK 128



template <bool ACC>
__device__ __forceinline__ void store(
    __nv_bfloat16* __restrict__ C, unsigned int idx, float a, float scale
) {
    if (ACC) {
        float d = __bfloat162float(__float2bfloat16(a));
        C[idx] = __float2bfloat16(__bfloat162float(C[idx]) + scale * d);
    } else {
        C[idx] = __float2bfloat16(a);
    }
}

// 2026-10-17: FP32 output stores the accumulator as it is.
template <bool ACC>
__device__ __forceinline__ void store(
    float* __restrict__ C, unsigned int idx, float a, float scale
) {
    static_assert(!ACC, "the scaled fold is defined for BF16 output only");
    (void)scale;
    C[idx] = a;
}

#define TC_OUT_T __nv_bfloat16
#include "dense_gemm_tc_mainloop.cuh"
#undef TC_OUT_T
#define TC_OUT_T float
#include "dense_gemm_tc_mainloop.cuh"
#undef TC_OUT_T
