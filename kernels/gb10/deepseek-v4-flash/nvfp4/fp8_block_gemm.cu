// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

// 2026-10-19: GEMV and GEMM against a weight held in E4M3 with one E8M0 scale per 32 x 32 tile,
// decoded in the kernel. Each is the twin of a bf16 kernel with the weight load replaced:
//   fp8_block_gemv_bf16 = common/dense_gemv_bf16 (grid ceil(N / 4), block 256)
//   fp8_block_gemm_bf16 = common/dense_gemm_bf16 (grid (ceil(N / 16), ceil(M / 16)), block 16 x 16)
// A weight decodes to code * 2^(scale - 127), rounded to bf16 nearest-even. That product has at
// most four significant bits, so for a scale code of 3 or more the rounding is exact and every
// value equals the reference decode (`Mat::decode_row`); below 3 the smallest steps fall under
// the bf16 subnormal step and round. The accumulation is the twin's own, in the same order, so
// each output is bit-identical to the twin over the decoded weight
// (`the_direct_gemv_is_bitwise_...`, `the_direct_gemm_is_bitwise_...`).
//
// A: [M, K] bf16. B: [N, K] E4M3 codes, row-major. S: [N / 32, K / 32] E8M0 bytes, row-major.
// C: [M, N] bf16.
// Owner: gb10 kernels (deepseek-v4-flash); launched by model-arch `fp8_block_v41`.
// Invariants:
// - N and K are whole 32-wide tiles; K % 8 == 0 for the GEMV's 8-code loads (the launcher
//   checks both).

#include <cuda_bf16.h>

#define FP8B_TILE 32

// 2026-10-19: e4m3fn: sign, 4-bit exponent with bias 7, 3-bit mantissa; exponent 0 is
// subnormal (mantissa / 512) and S.1111.111 is NaN. E8M0 byte b is 2^(b - 127), and 0 is the
// subnormal 2^-127.
__device__ __forceinline__ __nv_bfloat16 fp8b_decode(unsigned int code, unsigned int sbyte) {
    const unsigned int e = (code >> 3) & 0xFu, m = code & 7u;
    float mag;
    if (e == 0xFu && m == 7u) {
        mag = __int_as_float(0x7FC00000);
    } else if (e == 0u) {
        mag = (float)m * (1.0f / 512.0f);
    } else {
        mag = __uint_as_float(((e + 120u) << 23) | (m << 20));
    }
    const float v = (code & 0x80u) ? -mag : mag;
    const float s = sbyte == 0u ? __uint_as_float(0x00400000u) : __uint_as_float(sbyte << 23);
    return __float2bfloat16(v * s);
}

extern "C" __global__ void fp8_block_gemv_bf16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const unsigned char* __restrict__ S,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = 64;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * 4 + local_out;
    if (n >= N) return;

    float acc = 0.0f;
    const unsigned int K_VEC = K / 8;
    const uint4* A_vec = (const uint4*)A;
    const uint2* B_vec = (const uint2*)(B + (unsigned long long)n * K);
    const unsigned char* srow = S + (unsigned long long)(n / FP8B_TILE) * (K / FP8B_TILE);

    for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
        const uint4 a_data = A_vec[kv];
        const uint2 b_data = B_vec[kv];
        // 2026-10-19: Eight codes share one tile: 8 * kv .. 8 * kv + 7 lie in tile kv / 4.
        const unsigned int sb = srow[kv / 4];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int b_raw[2] = {b_data.x, b_data.y};

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
            const unsigned int pair = b_raw[i / 2] >> (16 * (i % 2));
            const __nv_bfloat16 b_lo = fp8b_decode(pair & 0xFFu, sb);
            const __nv_bfloat16 b_hi = fp8b_decode((pair >> 8) & 0xFFu, sb);
            acc += __bfloat162float(a_lo) * __bfloat162float(b_lo);
            acc += __bfloat162float(a_hi) * __bfloat162float(b_hi);
        }
    }

    const unsigned int warp_lane = threadIdx.x % 32;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }
    __shared__ float smem[4 * 2];
    if (warp_lane == 0) smem[local_out * 2 + (lane / 32)] = acc;
    __syncthreads();
    if (lane == 0) {
        const float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = __float2bfloat16(result);
    }
}

extern "C" __global__ void fp8_block_gemm_bf16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const unsigned char* __restrict__ S,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int row = blockIdx.y * 16 + threadIdx.y;
    const unsigned int col = blockIdx.x * 16 + threadIdx.x;
    const unsigned int k_tiles = K / FP8B_TILE;

    __shared__ __nv_bfloat16 smem_A[16][16];
    __shared__ __nv_bfloat16 smem_B[16][16];

    float acc = 0.0f;

    for (unsigned int k_base = 0; k_base < K; k_base += 16) {
        if (row < M && (k_base + threadIdx.x) < K) {
            smem_A[threadIdx.y][threadIdx.x] = A[row * K + k_base + threadIdx.x];
        } else {
            smem_A[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }

        const unsigned int k = k_base + threadIdx.y;
        if (k < K && col < N) {
            smem_B[threadIdx.y][threadIdx.x] = fp8b_decode(
                B[(unsigned long long)col * K + k],
                S[(unsigned long long)(col / FP8B_TILE) * k_tiles + k / FP8B_TILE]);
        } else {
            smem_B[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }

        __syncthreads();

        for (unsigned int kk = 0; kk < 16; kk++) {
            acc += __bfloat162float(smem_A[threadIdx.y][kk])
                 * __bfloat162float(smem_B[kk][threadIdx.x]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2bfloat16(acc);
    }
}
