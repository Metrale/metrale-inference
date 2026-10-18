// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-17: The dense_gemv_bf16_tc small-M tensor-core GEMV, parameterized by the output type:
// C[t, n] = sum_k A[t, k] * W[n, k] with FP32 accumulation, then one store per output (BF16
// round to nearest, or the FP32 sum as it is). `dense_gemv_bf16_tc_impl<NB, NT, KU>` has a
// BF16-output overload (dense_gemv_bf16_tc.cu) and an FP32-output overload
// (dense_gemm_tc_f32out.cu), both from the one body in dense_gemv_bf16_tc_mainloop.cuh.
//
// Owner: gb10 kernels.
// Invariants:
// - A is [M, K] contiguous, W is [N, K] row-major, row t of C starts at C + t * out_stride.
// - K % DTC_KB == 0 (the host declines any other K); any N; M <= 8 * NB.
// - Grid (ceil(N / (16 * NT)), 1, 1), block (DTC_WARPS * 32, 1, 1).
// - The sum order depends on K and DTC_WARPS alone, never on the output type, so the FP32
//   overload stores the sum the BF16 overload rounds: bf16_rn(C_f32) == C_bf16 bit for bit.
//
// Operand placement: weights on the 16-row A side of mma.m16n8k16, tokens on the 8-wide
// B side, so one MMA covers a 16x16 weight block for up to 8 tokens and the raw BF16 words
// go into the fragments with no per-weight conversion. NB token tiles reuse each A
// fragment; a tile whose first row is at or past M is skipped (warp-uniform).
//
// K permutation (why there is no repack): thread (g = lane / 4, t = lane % 4) loads two
// 16-byte pieces of weight rows g and g + 8 of its tile, at k offsets 8t and 32 + 8t of a
// 64-k block, straight from the [N, K] layout. Its eight BF16x2 words w0..w7 (piece 0,
// then piece 1) feed four MMAs: MMA j takes w(2j) in k slots {2t, 2t+1} and w(2j+1) in
// {2t+8, 2t+9}. The B fragment of token g is built from the same 16 k of activation row g
// in the same order, so both operands carry the same permutation and the dot product is
// unchanged.

#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

#define DTC_WARPS 8
#define DTC_KB 64

__device__ __forceinline__ void dtc_mma(float (&d)[4], uint32_t a0, uint32_t a1,
                                        uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// 2026-09-25: Streamed weight load: each weight is read once per launch, so it bypasses L1
// and leaves L1 to the activation rows every CTA re-reads.
__device__ __forceinline__ uint4 dtc_ld_stream(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t dtc_word(const uint4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// 2026-10-17: The per-output value: BF16 rounds to nearest, FP32 keeps the sum. A conversion, not a
// store, so the call site keeps the original `C[idx] = value` order (the PTX stays byte-identical).
template <class T> __device__ __forceinline__ T dtc_out(float v);
template <> __device__ __forceinline__ __nv_bfloat16 dtc_out<__nv_bfloat16>(float v) {
    return __float2bfloat16_rn(v);
}
template <> __device__ __forceinline__ float dtc_out<float>(float v) { return v; }

#define DTC_OUT_T __nv_bfloat16
#include "dense_gemv_bf16_tc_mainloop.cuh"
#undef DTC_OUT_T
#define DTC_OUT_T float
#include "dense_gemv_bf16_tc_mainloop.cuh"
#undef DTC_OUT_T
