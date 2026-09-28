// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-28: `moe_router_gemm_rt`: the MoE router gate GEMM C[M, N] = A[M, K] * B[N, K]^T (BF16 in, BF16 out) with
// the bits of `dense_gemm_bf16_router` (dense_gemm_bf16.cu): each C[m, n] is one FP32 accumulator updated as
// acc = acc + a * b in ascending k (this directory builds with --fmad=false, so the multiply and the add round
// separately), from BF16 values converted exactly to FP32, rounded to BF16 once at the end. Only the blocking differs:
// a 64 x 128 tile per CTA, 4 rows x 8 columns per thread (columns tx*4..+3 and 64+tx*4..+3, so each warp's float4
// reads of the B slice are contiguous), 16-wide K slices staged in shared memory as FP32 and double-buffered through
// registers. About 5x fewer shared-memory reads per multiply-add than the 4-column router kernel.
//
// Owner: gb10 kernels.
// Invariants:
// - K % 16 == 0 (the caller falls back to dense_gemm_bf16_router otherwise); any M, N.
// - Grid (ceil(N / 128), ceil(M / 64)), block 256.

#include <cuda_bf16.h>

#define RT_BM 64
#define RT_BN 128
#define RT_BK 16
#define RT_PA (RT_BM + 4)
#define RT_PB (RT_BN + 4)

extern "C" __global__ void __launch_bounds__(256) moe_router_gemm_rt(
    const __nv_bfloat16* __restrict__ A,   // 2026-09-28: [M, K]
    const __nv_bfloat16* __restrict__ B,   // 2026-09-28: [N, K]
    __nv_bfloat16* __restrict__ C,         // 2026-09-28: [M, N]
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    __shared__ __align__(16) float sA[2][RT_BK][RT_PA];
    __shared__ __align__(16) float sB[2][RT_BK][RT_PB];

    const unsigned tid = threadIdx.x;
    const unsigned tx = tid & 15u, ty = tid >> 4;
    const unsigned m0 = blockIdx.y * RT_BM, n0 = blockIdx.x * RT_BN;

    // 2026-09-28: Load roles: A row tid / 4, k (tid % 4) * 4 .. +3; B row tid / 2, k (tid % 2) * 8 .. +7.
    const unsigned a_row = tid >> 2, a_k = (tid & 3u) * 4;
    const unsigned b_row = tid >> 1, b_k = (tid & 1u) * 8;
    const bool a_ok = m0 + a_row < M, b_ok = n0 + b_row < N;
    const unsigned short* a_src = (const unsigned short*)A + (unsigned long long)(a_ok ? m0 + a_row : 0) * K + a_k;
    const unsigned short* b_src = (const unsigned short*)B + (unsigned long long)(b_ok ? n0 + b_row : 0) * K + b_k;

    ushort4 ra;
    uint4 rb;
    auto fetch = [&](unsigned kb) {
        ra = a_ok ? *(const ushort4*)(a_src + kb) : make_ushort4(0, 0, 0, 0);
        rb = b_ok ? *(const uint4*)(b_src + kb) : make_uint4(0, 0, 0, 0);
    };
    auto stash = [&](unsigned buf) {
        sA[buf][a_k + 0][a_row] = __bfloat162float(__ushort_as_bfloat16(ra.x));
        sA[buf][a_k + 1][a_row] = __bfloat162float(__ushort_as_bfloat16(ra.y));
        sA[buf][a_k + 2][a_row] = __bfloat162float(__ushort_as_bfloat16(ra.z));
        sA[buf][a_k + 3][a_row] = __bfloat162float(__ushort_as_bfloat16(ra.w));
        const unsigned short* u = (const unsigned short*)&rb;
        #pragma unroll
        for (int j = 0; j < 8; j++) sB[buf][b_k + j][b_row] = __bfloat162float(__ushort_as_bfloat16(u[j]));
    };

    float acc[4][8];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;

    fetch(0);
    stash(0);
    __syncthreads();
    unsigned buf = 0;
    for (unsigned kb = 0; kb < K; kb += RT_BK) {
        const bool more = kb + RT_BK < K;
        if (more) fetch(kb + RT_BK);
        #pragma unroll
        for (int kk = 0; kk < RT_BK; kk++) {
            const float4 a = *(const float4*)&sA[buf][kk][ty * 4];
            const float4 b0 = *(const float4*)&sB[buf][kk][tx * 4];
            const float4 b1 = *(const float4*)&sB[buf][kk][64 + tx * 4];
            const float av[4] = {a.x, a.y, a.z, a.w};
            const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] += av[i] * bv[j];
        }
        if (more) {
            stash(buf ^ 1u);
            __syncthreads();
            buf ^= 1u;
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const unsigned m = m0 + ty * 4 + i;
        if (m >= M) continue;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const unsigned n = n0 + (j < 4 ? tx * 4 + j : 64 + tx * 4 + (j - 4));
            if (n < N) C[(unsigned long long)m * N + n] = __float2bfloat16(acc[i][j]);
        }
    }
}
