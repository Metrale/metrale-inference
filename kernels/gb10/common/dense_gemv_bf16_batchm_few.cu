// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-21: dense_gemv_bf16_batchm (dense_gemv_bf16_batchm.cu) for 2..FEW_MAX_M rows with the
// latency taken out. The runtime-M body walks K in 64-vector slabs: each slab stages every A row
// in shared memory, waits at a barrier, loads one B vector per thread, accumulates (acc[] in local
// memory, as M is not a compile-time count) and waits at a second barrier. At GLM-5.3's decode
// widths (two verify rows) a call is a chain of K / 512 such rounds, each a DRAM trip for B; on
// GB10 the router (288 x 4096) took 16.6 us a call against a 10 us DRAM floor.
// Here, per thread:
//   - every B vector of up to FEW_PF slabs is loaded before any is used (one DRAM trip, not one
//     per slab);
//   - the A vectors are read from global memory (L1/L2: the four output groups of a block and
//     every block read the same rows), so no slab barrier;
//   - the row count is a template parameter, so acc[] is in registers.
//
// Owner: gb10 kernels.
// Invariants:
// - Each row's result is bit-identical to dense_gemv_bf16_batchm's row (and so dense_gemv_bf16's):
//   the thread's kv sequence (lane, lane + 64, ...), the lo-then-hi adds per row, the scalar tail
//   over K % 8, the warp shuffle and the two-warp sum are the same operations in the same order;
//   the common build passes --fmad=false. Only where the operands are read from changes.
//   glm5next_batchm_few_gate (gpu examples) checks it byte for byte.
// - A is [M, K] contiguous, B is [N, K] row-major, row t of C starts at C + t * out_stride. K % 8 == 0
//   (16-byte rows, as dense_gemv_bf16_batchm).
// - Launch: grid (ceil(N / 4), 1, 1), block (256, 1, 1). M outside 2..FEW_MAX_M writes nothing
//   (the host sends only those).

#include <cuda_bf16.h>

namespace bmfew {

constexpr unsigned int BLOCK = 256;
constexpr unsigned int N_PER_BLOCK = 4;
constexpr unsigned int TPO = BLOCK / N_PER_BLOCK;
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

template <int MM, typename OutT>
__device__ __forceinline__ void few_body(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    OutT* __restrict__ C,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride,
    float (*smem)[N_PER_BLOCK * 2]
) {
    const unsigned int local_out = threadIdx.x / TPO;
    const unsigned int lane = threadIdx.x % TPO;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool active = (n < N);

    float acc[MM];
#pragma unroll
    for (int t = 0; t < MM; t++) acc[t] = 0.0f;

    const unsigned int K_VEC = K / VEC;
    if (active) {
        const uint4* B_vec = (const uint4*)(B + (unsigned long long)n * K);
        for (unsigned int s0 = 0; s0 * TPO < K_VEC; s0 += FEW_PF) {
            uint4 b[FEW_PF];
#pragma unroll
            for (int p = 0; p < FEW_PF; p++) {
                const unsigned int kv = (s0 + p) * TPO + lane;
                b[p] = kv < K_VEC ? B_vec[kv] : make_uint4(0u, 0u, 0u, 0u);
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

template <typename OutT>
__device__ __forceinline__ void few_dispatch(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    OutT* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride
) {
    __shared__ float smem[FEW_MAX_M][N_PER_BLOCK * 2];
    switch (M) {
        case 2: few_body<2, OutT>(A, B, C, N, K, out_stride, smem); break;
        case 3: few_body<3, OutT>(A, B, C, N, K, out_stride, smem); break;
        case 4: few_body<4, OutT>(A, B, C, N, K, out_stride, smem); break;
        case 5: few_body<5, OutT>(A, B, C, N, K, out_stride, smem); break;
        case 6: few_body<6, OutT>(A, B, C, N, K, out_stride, smem); break;
        case 7: few_body<7, OutT>(A, B, C, N, K, out_stride, smem); break;
        case 8: few_body<8, OutT>(A, B, C, N, K, out_stride, smem); break;
        default: break;
    }
}

}  // namespace bmfew

extern "C" __global__ void __launch_bounds__(bmfew::BLOCK) dense_gemv_bf16_batchm_few(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride
) {
    bmfew::few_dispatch<__nv_bfloat16>(A, B, C, M, N, K, out_stride);
}

extern "C" __global__ void __launch_bounds__(bmfew::BLOCK) dense_gemv_bf16_batchm_few_fp32out(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride
) {
    bmfew::few_dispatch<float>(A, B, C, M, N, K, out_stride);
}
