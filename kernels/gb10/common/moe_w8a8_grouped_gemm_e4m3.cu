// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-27: `moe_w8a8_grouped_gemm_e4m3_*`: the routed-expert grouped GEMM of `moe_w8a8_grouped_gemm_pm4`
// (same inputs, same work-list, same output) on the native FP8 tensor-core MMA
// `mma.sync.m16n8k32.row.col.f32.e4m3.e4m3.f32` instead of software E4M3 -> BF16 decode plus BF16 MMAs:
//   C[m, n] = bf16( sum_c ( sum_(k in c) A[token(m), k] * B[n, k] ) * (a_scale[token(m), c / 2] * b_scale[n / 128, c / 2]) )
// with c over 64-value chunks of K, as in the PM4 kernel. Every FP8 x FP8 product is exact in F32 on both paths, and
// each MMA sums the same 16 products in the same K order as PM4's BF16 m16n8k16 MMAs (SPLIT16 in the header), so the
// output equals PM4's bit for bit on every input tested. It is deterministic: a row's result depends only on its own
// inputs (fixed tile and fold order, no atomics, no split-K).
//
// Work-list: item w is (expert_id, mt << 6 | nt) from `moe_build_tile_worklist` with m_tile = BM and
// n_tiles = N / BN, written earlier on the same stream. Expert e owns rows expert_offsets[e] .. expert_offsets[e+1];
// token(m) is sorted_token_ids[m], or m when that pointer is NULL. An expert with a NULL weight pointer is skipped.
//
// The tile pipeline is e4m3_mma_pipe.cuh (SPLIT16, FOLD_STEPS 1). Requirements (checked by the caller): K % 128 == 0,
// N % BN == 0. Dynamic shared memory: e4m3g::SmemBytes<BM, BN, 3>.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.

#include "e4m3_mma_pipe.cuh"

namespace e4m3g {

template <int BM, int BN, int WARPS_M, int WARPS_N, int STAGES>
__device__ __forceinline__ void grouped_body(
    const unsigned char* __restrict__ A_fp8,
    const float* __restrict__ a_scale,
    const unsigned long long* __restrict__ B_weight_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int N,
    unsigned int K,
    const unsigned int* __restrict__ worklist,
    const int* __restrict__ total_tiles
) {
    constexpr int THREADS = WARPS_M * WARPS_N * 32;
    extern __shared__ __align__(128) unsigned char smem[];
    int* sTok = tile_rows<BM, BN, STAGES>(smem);
    const unsigned k_blocks = K / SCALE_BLOCK;
    const int total = *total_tiles;

    for (int wid = blockIdx.x; wid < total; wid += (int)gridDim.x) {
        __syncthreads();  // 2026-09-27: the previous item is done with shared memory

        const unsigned expert_id = worklist[wid * 2 + 0];
        const unsigned packed = worklist[wid * 2 + 1];
        const unsigned cta_m = (packed >> 6) * BM;
        const unsigned cta_n = (packed & 0x3Fu) * BN;
        const int m_start = expert_offsets[expert_id];
        const int M_expert = expert_offsets[expert_id + 1] - m_start;
        const unsigned char* B_exp = (const unsigned char*)B_weight_ptrs[expert_id];
        const float* S_exp = (const float*)B_scale_ptrs[expert_id];
        if (B_exp == 0) continue;

        for (unsigned i = threadIdx.x; i < (unsigned)BM; i += THREADS) {
            const unsigned m = cta_m + i;
            int t = -1;
            if (m < (unsigned)M_expert) {
                const int s = m_start + (int)m;
                t = sorted_token_ids ? sorted_token_ids[s] : s;
            }
            sTok[i] = t;
        }
        __syncthreads();

        const int rows_valid = M_expert - (int)cta_m;
        float outer[BM / WARPS_M / 16][BN / WARPS_N / 8][4];
        tile_mma<BM, BN, WARPS_M, WARPS_N, STAGES, true, 1>(
            smem, A_fp8, a_scale, B_exp + (unsigned long long)cta_n * K,
            S_exp + (cta_n / SCALE_BLOCK) * k_blocks, K, rows_valid, outer);
        tile_store<BM, BN, WARPS_M, WARPS_N>(C, N, (unsigned long long)(m_start + (int)cta_m), rows_valid, cta_n, outer);
    }
}

}  // namespace e4m3g

// 2026-09-27: The entry points' shared parameter list and forwarding list.
#define E4M3G_PARAMS                                                                                   \
    const unsigned char* __restrict__ A_fp8,              /* [total_tokens, K] FP8 E4M3 */             \
    const float* __restrict__ a_scale,                    /* [total_tokens, K / 128] F32 */            \
    const unsigned long long* __restrict__ B_weight_ptrs, /* [num_experts] -> [N, K] FP8 */            \
    const unsigned long long* __restrict__ B_scale_ptrs,  /* [num_experts] -> [N / 128, K / 128] F32 */\
    __nv_bfloat16* __restrict__ C,                        /* [total_expanded, N] BF16 */               \
    const int* __restrict__ expert_offsets,               /* [num_experts + 1] */                      \
    const int* __restrict__ sorted_token_ids,             /* [total_expanded] or NULL */               \
    unsigned int N, unsigned int K,                                                                    \
    const unsigned int* __restrict__ worklist,            /* [*total_tiles * 2] */                     \
    const int* __restrict__ total_tiles
#define E4M3G_ARGS A_fp8, a_scale, B_weight_ptrs, B_scale_ptrs, C, expert_offsets, sorted_token_ids, N, K, worklist, total_tiles

// 2026-09-27: Gate/up shape (N = inter, K = hidden): 64 x 128 items, 8 warps as 2 x 4 (warp tile 32 x 32), 3
// stages, 37.8 KiB of dynamic shared memory, 2 CTAs per SM. Work-list m_tile 64, n_tiles N / 128.
extern "C" __global__ void __launch_bounds__(256, 2) moe_w8a8_grouped_gemm_e4m3_gu(E4M3G_PARAMS) {
    e4m3g::grouped_body<64, 128, 2, 4, 3>(E4M3G_ARGS);
}

// 2026-09-27: Down shape (N = hidden, K = inter): 128 x 64 items, 8 warps as 4 x 2 (warp tile 32 x 32), 3 stages,
// 38.8 KiB, 2 CTAs per SM. Work-list m_tile 128, n_tiles N / 64.
extern "C" __global__ void __launch_bounds__(256, 2) moe_w8a8_grouped_gemm_e4m3_dn(E4M3G_PARAMS) {
    e4m3g::grouped_body<128, 64, 4, 2, 3>(E4M3G_ARGS);
}
