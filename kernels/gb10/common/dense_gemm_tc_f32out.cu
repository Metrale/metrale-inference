// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-17: BF16 x BF16 -> FP32 dense projection on m16n8k16 tensor cores:
// C[M, N] (FP32) = A[M, K] (BF16) * B[N, K]^T (BF16), FP32 accumulation, no output rounding.
//
// The in-tree replacement for the cuBLASLt FP32-output projection
// (model-layers ops::cublas_bf16_proj_dense_f32_out). Two instantiations of existing mainloops,
// no copy of either:
// - `dense_gemm_tc_f32out`: the dense_gemm_tc tile (dense_gemm_tc_body.cuh), FP32-output overload.
//   Any M, N, K. Grid (ceil(N / 64), ceil(M / 16)), block 128.
// - `dense_gemv_bf16_tc{8,16,32}_f32out`: the small-M tensor-core GEMV
//   (dense_gemv_bf16_tc_body.cuh), FP32-output overload, for M <= 8 / 16 / 32 decode rows.
//   K % 64 == 0, any N. Grid (ceil(N / 16)), block 256.
//
// Owner: gb10 kernels.
// Invariants:
// - A is [M, K] contiguous, B is [N, K] row-major. The tile entry writes C [M, N] contiguous;
//   the GEMV entries write row t at C + t * out_stride.
// - Each entry's sum order is its BF16 sibling's (dense_gemm_tc, dense_gemv_bf16_tc*), so
//   rounding its output to BF16 reproduces the sibling's bits. The tile and GEMV paths sum in
//   different orders, so their FP32 outputs differ in the last bits for the same inputs.
// - The GEMV tiers keep the geometry of dense_gemv_bf16_tc.cu's DTC_ENTRY list (its comment
//   holds the measurements); FP32 output adds no registers to the mainloop.

#include "dense_gemm_tc_body.cuh"
#include "dense_gemv_bf16_tc_body.cuh"

extern "C" __global__ void dense_gemm_tc_f32out(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    dense_gemm_tc_body<false>(A, B, C, M, N, K, 0.0f);
}

#define DTC_F32_ENTRY(NAME, NB, NT, KU)                                                 \
    extern "C" __global__ __launch_bounds__(DTC_WARPS * 32) void NAME(                   \
        const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ W,         \
        float* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,            \
        unsigned int out_stride) {                                                        \
        dense_gemv_bf16_tc_impl<NB, NT, KU>(A, W, C, M, N, K, out_stride);               \
    }

DTC_F32_ENTRY(dense_gemv_bf16_tc8_f32out, 1, 1, 8)
DTC_F32_ENTRY(dense_gemv_bf16_tc16_f32out, 2, 1, 8)
DTC_F32_ENTRY(dense_gemv_bf16_tc32_f32out, 4, 1, 4)
