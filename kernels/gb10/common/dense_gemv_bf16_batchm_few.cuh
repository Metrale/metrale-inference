// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-23: The body of the few-row BF16 GEMV (dense_gemv_bf16_batchm_few.cu, which documents
// the method and its bit identity with dense_gemv_bf16_batchm), templated on NPB, the outputs one
// block of TPO * NPB threads holds. Instantiated by dense_gemv_bf16_batchm_few.cu (NPB = 4) and
// the GLM router entries (glm5next_router_gemv.cu).
//
// Owner: gb10 kernels.
// Invariants: an output's bits do not depend on NPB (module comment of the .cu); launch grid
// (ceil(N / NPB), 1, 1), block (TPO * NPB, 1, 1).
#pragma once

#include <cuda_bf16.h>
#include "pdl.cuh"

namespace bmfew {

// 2026-10-23: TPO threads per output fix every output's arithmetic (its kv sequence, the warp
// shuffle and the two-warp sum); NPB, the outputs one block holds, only places outputs on
// blocks, so every NPB gives the same bits. The common entries are NPB = 4 (256 threads).
constexpr unsigned int TPO = 64;
constexpr unsigned int N_PER_BLOCK = 4;
constexpr unsigned int BLOCK = TPO * N_PER_BLOCK;
constexpr unsigned int WARP = 32;
constexpr unsigned int VEC = 8;
// 2026-10-21: Slabs whose B vectors are in flight at once: K up to FEW_PF * 512 in one trip
// (GLM-5.3's widest such K is 4096).
constexpr int FEW_PF = 8;
constexpr int FEW_MAX_M = 8;

__device__ __forceinline__ void store_out(__nv_bfloat16* C, unsigned long long i, float r) {
    C[i] = __float2bfloat16(r);
}
__device__ __forceinline__ void store_out(float* C, unsigned long long i, float r) {
    C[i] = r;
}

__device__ __forceinline__ float lo(unsigned int v) {
    __nv_bfloat16 h;
    *(unsigned short*)&h = (unsigned short)(v & 0xFFFF);
    return __bfloat162float(h);
}
__device__ __forceinline__ float hi(unsigned int v) {
    __nv_bfloat16 h;
    *(unsigned short*)&h = (unsigned short)(v >> 16);
    return __bfloat162float(h);
}

template <int MM, unsigned int NPB, typename OutT>
__device__ __forceinline__ void few_body(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    OutT* __restrict__ C,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride,
    float (*smem)[NPB * 2],
    unsigned int blk
) {
    const unsigned int local_out = threadIdx.x / TPO;
    const unsigned int lane = threadIdx.x % TPO;
    const unsigned int n = blk * NPB + local_out;
    const bool active = (n < N);

    float acc[MM];
#pragma unroll
    for (int t = 0; t < MM; t++) acc[t] = 0.0f;

    const unsigned int K_VEC = K / VEC;
    // 2026-10-21: PDL (pdl.cuh): the first FEW_PF slabs of B (the weight) are loaded before the
    // wait; A and every store after it. Same loads, same values.
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)(active ? n : 0u) * K);
    uint4 b[FEW_PF];
#pragma unroll
    for (int p = 0; p < FEW_PF; p++) {
        const unsigned int kv = (unsigned int)p * TPO + lane;
        b[p] = active && kv < K_VEC ? B_vec[kv] : make_uint4(0u, 0u, 0u, 0u);
    }
    pdl_enter();
    if (active) {
        for (unsigned int s0 = 0; s0 * TPO < K_VEC; s0 += FEW_PF) {
            if (s0 != 0) {
#pragma unroll
                for (int p = 0; p < FEW_PF; p++) {
                    const unsigned int kv = (s0 + p) * TPO + lane;
                    b[p] = kv < K_VEC ? B_vec[kv] : make_uint4(0u, 0u, 0u, 0u);
                }
            }
#pragma unroll
            for (int p = 0; p < FEW_PF; p++) {
                const unsigned int kv = (s0 + p) * TPO + lane;
                if (kv < K_VEC) {
                    const unsigned int b_raw[4] = {b[p].x, b[p].y, b[p].z, b[p].w};
                    float bf[8];
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        bf[2 * i] = lo(b_raw[i]);
                        bf[2 * i + 1] = hi(b_raw[i]);
                    }
#pragma unroll
                    for (int t = 0; t < MM; t++) {
                        const uint4 a_data = ((const uint4*)(A + (unsigned long long)t * K))[kv];
                        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
                        float a = acc[t];
#pragma unroll
                        for (int i = 0; i < 4; i++) {
                            a += lo(a_raw[i]) * bf[2 * i];
                            a += hi(a_raw[i]) * bf[2 * i + 1];
                        }
                        acc[t] = a;
                    }
                }
            }
        }
        const unsigned int tail_start = K_VEC * VEC;
        const __nv_bfloat16* B_row = B + (unsigned long long)n * K;
        for (unsigned int k = tail_start + lane; k < K; k += TPO) {
            const float bfv = __bfloat162float(B_row[k]);
#pragma unroll
            for (int t = 0; t < MM; t++) acc[t] += __bfloat162float(A[(unsigned long long)t * K + k]) * bfv;
        }
#pragma unroll
        for (int t = 0; t < MM; t++) {
            float a = acc[t];
#pragma unroll
            for (int offset = WARP / 2; offset > 0; offset >>= 1) a += __shfl_down_sync(0xFFFFFFFF, a, offset);
            acc[t] = a;
        }
        if (threadIdx.x % WARP == 0) {
            const unsigned int idx = local_out * 2 + (lane / WARP);
#pragma unroll
            for (int t = 0; t < MM; t++) smem[t][idx] = acc[t];
        }
    }
    __syncthreads();
    if (active && lane == 0) {
#pragma unroll
        for (int t = 0; t < MM; t++) {
            const float r = smem[t][local_out * 2] + smem[t][local_out * 2 + 1];
            store_out(C, (unsigned long long)t * out_stride + n, r);
        }
    }
}

template <unsigned int NPB, typename OutT>
__device__ __forceinline__ void few_dispatch(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    OutT* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride,
    unsigned int blk
) {
    __shared__ float smem[FEW_MAX_M][NPB * 2];
    switch (M) {
        case 2: few_body<2, NPB, OutT>(A, B, C, N, K, out_stride, smem, blk); break;
        case 3: few_body<3, NPB, OutT>(A, B, C, N, K, out_stride, smem, blk); break;
        case 4: few_body<4, NPB, OutT>(A, B, C, N, K, out_stride, smem, blk); break;
        case 5: few_body<5, NPB, OutT>(A, B, C, N, K, out_stride, smem, blk); break;
        case 6: few_body<6, NPB, OutT>(A, B, C, N, K, out_stride, smem, blk); break;
        case 7: few_body<7, NPB, OutT>(A, B, C, N, K, out_stride, smem, blk); break;
        case 8: few_body<8, NPB, OutT>(A, B, C, N, K, out_stride, smem, blk); break;
        default: break;
    }
}

}  // namespace bmfew
