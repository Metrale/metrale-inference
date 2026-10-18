// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-12: NVFP4 W4A16 GEMM over the full M in one launch, on the row-tile template of
// tc_rows.cuh: block (blockIdx.x, blockIdx.y) computes the 128 output columns of column block
// blockIdx.x for rows [64 blockIdx.y, 64 blockIdx.y + 64) with tr_block_split<Nvfp4G16, NT 8, G 1,
// RAGGED, S 1, TrWhole, NW 8>, i.e. w4a16_tc_rows_64's body on 8 warps, offset by the row tile.
//   C[r, n] = s2 * sum_k A[r, k] * BF16(E2M1(W[n, k]) * E4M3(scale[n, k / 16])),  r < M, n < N
//
// Why: the prefill arm of the W4A16 tier ran dequant_nvfp4_to_bf16_g16 (a 2-byte copy of every
// weight, 22 ms per cold 1k request on GLM-5.3) and a cuBLASLt BF16 GEMM, whose bits differ from
// the row tiles decode runs. This launch reads the NVFP4 weight directly (from L2 per row tile)
// and, by tc_rows.cuh's invariant that a row's output bits depend on neither M, NT, G nor NW,
// writes EVERY row's bytes exactly as w4a16_tc_rows_64 does: prefill bits == decode bits.
//
// Owner: gb10 kernels.
// Invariants:
// - Contract of w4a16_tc_rows (packed E2M1 [N, K / 2], E4M3 scales [N, K / 16], FP32 s2; any
//   M >= 1, any N > 0, K a positive multiple of 256, lda a multiple of 8 and >= K, ldc >= N).
// - Grid (ceil(N / 128), ceil(M / 64), 1), block 256, static shared memory only (35 KB).
// - A row tile past M exits; a partial last tile passes its live rows as M to the template.
#include "tc_rows.cuh"

extern "C" __global__ void __launch_bounds__(256) w4a16_tc_gemm_64x128(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ packed,
    const unsigned char* __restrict__ scale, float s2, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int lda, unsigned int ldc
) {
    const unsigned int r0 = blockIdx.y * 64u;
    if (r0 >= M) return;
    const unsigned int rows = min(64u, M - r0);
    tr_block_split<Nvfp4G16, 8, 1, true, 1, TrWhole, 8>(
        A + (unsigned long long)r0 * lda, {packed, scale, s2}, C + (unsigned long long)r0 * ldc,
        rows, N, K, lda, ldc, blockIdx.x, 0u);
}
