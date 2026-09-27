// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-26: moe_router_gemm_bf16: the MoE router gate GEMM, C[M, N] = A[M, K] * B[N, K]^T with
// BF16 in and out, bit-identical to dense_gemm_bf16 (dense_gemm_bf16.cu) at small M.
//
// dense_gemm_bf16 gives each output one FP32 accumulator that adds float(a) * float(b) for
// k = 0 .. K - 1 in order, and waits on a global load for every 16-wide K tile, so at the
// router's shape (N = 256 experts, K = 2048, M <= 16) it runs 16 blocks for ~70 us. Here a block
// stages a 512-wide K slice of RC_COLS weight rows and of its (at most 16) activation rows in
// shared memory with all 256 threads, and each of 16 * RC_COLS threads then runs its output's
// chain over the slice in the same k order. The build passes --fmad=false (KERNEL.toml), so the
// product is rounded before the add, as in dense_gemm_bf16.
//
// Owner: gb10 kernels.
// Invariants:
// - K % 16 == 0 (the host checks it): dense_gemm_bf16 pads a partial K tile with zero terms, which
//   would turn a -0 sum into +0.
// - Launch: grid (ceil(N / RC_COLS), ceil(M / RC_ROWS)), block 256, no dynamic shared memory.
//   RC_COLS and RC_ROWS must equal MOE_ROUTER_GEMM_COLS and MOE_ROUTER_GEMM_ROWS in
//   gemm_dense_bf16.rs.

#include <cuda_bf16.h>

#define RC_THREADS 256
#define RC_COLS 4
#define RC_ROWS 16
#define RC_KC 512
// 2026-09-26: Row pitch of the staged slices: 8 BF16 of padding shifts consecutive rows by four
// banks, so the RC_COLS weight rows a warp reads in one step do not share a bank.
#define RC_PITCH (RC_KC + 8)

extern "C" __global__ void moe_router_gemm_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    __shared__ __align__(16) __nv_bfloat16 s_a[RC_ROWS][RC_PITCH];
    __shared__ __align__(16) __nv_bfloat16 s_b[RC_COLS][RC_PITCH];

    const unsigned int col0 = blockIdx.x * RC_COLS;
    const unsigned int row0 = blockIdx.y * RC_ROWS;
    const unsigned int rows = min((unsigned int)RC_ROWS, M - row0);
    const unsigned int cols = min((unsigned int)RC_COLS, N - col0);
    // 2026-09-26: Thread t < RC_ROWS * RC_COLS owns output (row0 + t / RC_COLS, col0 + t % RC_COLS).
    const unsigned int r = threadIdx.x / RC_COLS;
    const unsigned int c = threadIdx.x % RC_COLS;
    const bool owns = threadIdx.x < RC_ROWS * RC_COLS && r < rows && c < cols;
    float acc = 0.0f;

    const unsigned int vec_per_row = RC_KC / 8;
    for (unsigned int k0 = 0; k0 < K; k0 += RC_KC) {
        const unsigned int kc = min((unsigned int)RC_KC, K - k0);
        const unsigned int kv = kc / 8;
        for (unsigned int i = threadIdx.x; i < (rows + cols) * vec_per_row; i += RC_THREADS) {
            const unsigned int line = i / vec_per_row;
            const unsigned int v = i % vec_per_row;
            if (v >= kv) continue;
            if (line < rows) {
                const uint4 x = *(const uint4*)(A + (unsigned long long)(row0 + line) * K + k0 + v * 8);
                *(uint4*)&s_a[line][v * 8] = x;
            } else {
                const unsigned int bl = line - rows;
                const uint4 x = *(const uint4*)(B + (unsigned long long)(col0 + bl) * K + k0 + v * 8);
                *(uint4*)&s_b[bl][v * 8] = x;
            }
        }
        __syncthreads();
        if (owns) {
            // 2026-09-26: Eight k at a time from one uint4 of each operand, still added one by one
            // in ascending k.
            for (unsigned int k8 = 0; k8 < kv; k8++) {
                const uint4 va = *(const uint4*)&s_a[r][k8 * 8];
                const uint4 vb = *(const uint4*)&s_b[c][k8 * 8];
                const unsigned int wa[4] = {va.x, va.y, va.z, va.w};
                const unsigned int wb[4] = {vb.x, vb.y, vb.z, vb.w};
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    __nv_bfloat16 a_lo, a_hi, b_lo, b_hi;
                    *(unsigned short*)&a_lo = (unsigned short)(wa[i] & 0xFFFF);
                    *(unsigned short*)&a_hi = (unsigned short)(wa[i] >> 16);
                    *(unsigned short*)&b_lo = (unsigned short)(wb[i] & 0xFFFF);
                    *(unsigned short*)&b_hi = (unsigned short)(wb[i] >> 16);
                    acc += __bfloat162float(a_lo) * __bfloat162float(b_lo);
                    acc += __bfloat162float(a_hi) * __bfloat162float(b_hi);
                }
            }
        }
        __syncthreads();
    }
    if (owns) {
        C[(unsigned long long)(row0 + r) * N + col0 + c] = __float2bfloat16(acc);
    }
}
