// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-27: `fp8_gemm_blockscaled_pipe_*`: the dense W8A8 block-scaled GEMM of `fp8_gemm_t_blockscaled` (same
// operands and output) on the pipelined tile of e4m3_mma_pipe.cuh:
//   C[m, n] = bf16( sum_g ( sum_{k in g} A[m, k] * B[n, k] ) * a_scale[m, g] * b_scale[n / 128, g] ),  g = 128-K groups
// Per output element the arithmetic is fp8_gemm_t_blockscaled's: one m16n8k32 e4m3 MMA per 32 K values in ascending
// K into an inner F32 accumulator that is folded into the outer one with a_scale * b_scale at every 128-K boundary.
// The output is therefore bit-identical to it; only the tiling, staging and fragment loads differ.
//
// Owner: gb10 kernels.
// Invariants:
// - A is [M, K] FP8 E4M3 with a_scale [M, K/128] F32; B is [N, K] FP8 E4M3 with b_scale [ceil(N/128), K/128] F32; C
//   is [M, N] BF16. K % 128 == 0 and N % BN == 0 (checked by the caller).
// - Grid (N / BN, ceil(M / BM), 1), block 256, dynamic shared memory e4m3g::SmemBytes<BM, BN, STAGES>.

#include "e4m3_mma_pipe.cuh"

namespace e4m3g {

template <int BM, int BN, int WARPS_M, int WARPS_N, int STAGES>
__device__ __forceinline__ void dense_body(
    const unsigned char* __restrict__ A_fp8,
    const float* __restrict__ a_scale,
    const unsigned char* __restrict__ B_fp8,
    const float* __restrict__ b_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    constexpr int THREADS = WARPS_M * WARPS_N * 32;
    extern __shared__ __align__(128) unsigned char smem[];
    int* sTok = tile_rows<BM, BN, STAGES>(smem);
    // 2026-09-27: Grouped rasterization: consecutive CTAs walk GROUP_M M tiles down one N column before moving to
    // the next, so a wave reuses a few B tiles and a few A rows from L2 instead of streaming all of B once per M row
    // (B is 25 MB at N 12288, K 2048, more than L2). It changes only which CTA computes which tile.
    constexpr unsigned GROUP_M = 8;
    const unsigned num_n = gridDim.x, num_m = gridDim.y;
    const unsigned pid = blockIdx.y * num_n + blockIdx.x;
    const unsigned first_m = (pid / (GROUP_M * num_n)) * GROUP_M;
    const unsigned group_rows = min(num_m - first_m, GROUP_M);
    const unsigned cta_m = (first_m + pid % group_rows) * BM;
    const unsigned cta_n = ((pid % (GROUP_M * num_n)) / group_rows) * BN;
    for (unsigned i = threadIdx.x; i < (unsigned)BM; i += THREADS) sTok[i] = cta_m + i < M ? (int)(cta_m + i) : -1;
    __syncthreads();
    const int rows_valid = (int)(M - cta_m);
    float outer[BM / WARPS_M / 16][BN / WARPS_N / 8][4];
    tile_mma<BM, BN, WARPS_M, WARPS_N, STAGES, false, 2>(
        smem, A_fp8, a_scale, B_fp8 + (unsigned long long)cta_n * K,
        b_scale + (cta_n / SCALE_BLOCK) * (K / SCALE_BLOCK), K, rows_valid, outer);
    tile_store<BM, BN, WARPS_M, WARPS_N>(C, N, cta_m, rows_valid, cta_n, outer);
}

}
#define FP8P_PARAMS                                                                                    \
    const unsigned char* __restrict__ A_fp8, const float* __restrict__ a_scale,                        \
        const unsigned char* __restrict__ B_fp8, const float* __restrict__ b_scale,                    \
        __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K
#define FP8P_ARGS A_fp8, a_scale, B_fp8, b_scale, C, M, N, K

// 2026-09-27: 128 x 64 tiles, 8 warps as 4 x 2 (warp tile 32 x 32), 3 stages: 38 KiB, two CTAs per SM. Measured
// on GB10 against 128x128 (4 stages), 256x128 and 64x128 (3 stages): 97-112 TFLOPS at the 35B's projection shapes.
extern "C" __global__ void __launch_bounds__(256, 2) fp8_gemm_blockscaled_pipe_128x64(FP8P_PARAMS) {
    e4m3g::dense_body<128, 64, 4, 2, 3>(FP8P_ARGS);
}
