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
#include <cuda_fp8.h>

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

}
// 2026-09-27: The entry points' shared parameter list and forwarding list. Shapes: A_fp8
// [total_tokens, K] FP8 E4M3; a_scale [total_tokens, K / 128] F32; B_weight_ptrs [num_experts] ->
// [N, K] FP8; B_scale_ptrs [num_experts] -> [N / 128, K / 128] F32; C [total_expanded, N] BF16;
// expert_offsets [num_experts + 1]; sorted_token_ids [total_expanded] or NULL; worklist
// [*total_tiles * 2].
#define E4M3G_PARAMS                                                                               \
    const unsigned char* __restrict__ A_fp8,                                                       \
    const float* __restrict__ a_scale,                                                             \
    const unsigned long long* __restrict__ B_weight_ptrs,                                          \
    const unsigned long long* __restrict__ B_scale_ptrs,                                           \
    __nv_bfloat16* __restrict__ C,                                                                 \
    const int* __restrict__ expert_offsets,                                                        \
    const int* __restrict__ sorted_token_ids,                                                      \
    unsigned int N, unsigned int K,                                                                \
    const unsigned int* __restrict__ worklist,                                                     \
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

namespace e4m3g {

// 2026-09-28: Gate and up of the same 64 x 128 item, then SiLU(gate) * up quantized to E4M3 per 128-column group, in
// place of the two `_gu` GEMMs, their BF16 stores and `silu_mul_quant_fp8` (moe_silu_mul.cu). Per element the math is
// the unfused chain's: gate and up rounded to BF16 as the GEMMs store them, r = bf16(g * (1 / (1 + __expf(-g))) * u),
// the row's max |r| over the group (an exact max, so its order is free), scale = max / 448 floored at 1e-12, then
// r / scale clamped to +-448 and encoded SATFINITE. Writes out_fp8 [total_expanded, N] and a_scale_out
// [total_expanded, N / 128] at the sorted rows. N % 128 == 0 (one group per item).
__device__ __forceinline__ void gateup_silu_body(
    const unsigned char* __restrict__ A_fp8, const float* __restrict__ a_scale,
    const unsigned long long* __restrict__ G_weight_ptrs, const unsigned long long* __restrict__ G_scale_ptrs,
    const unsigned long long* __restrict__ U_weight_ptrs, const unsigned long long* __restrict__ U_scale_ptrs,
    unsigned char* __restrict__ out_fp8, float* __restrict__ a_scale_out, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, unsigned int N, unsigned int K, const unsigned int* __restrict__ worklist,
    const int* __restrict__ total_tiles
) {
    constexpr int BM = 64, BN = 128, WARPS_M = 2, WARPS_N = 4, STAGES = 3, THREADS = 256;
    constexpr int MI = BM / WARPS_M / 16, NI = BN / WARPS_N / 8;
    extern __shared__ __align__(128) unsigned char smem[];
    int* sTok = tile_rows<BM, BN, STAGES>(smem);
    __shared__ float row_max[BM][WARPS_N];
    const unsigned k_blocks = K / SCALE_BLOCK;
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u, gid = lane >> 2, tig = lane & 3u;
    const unsigned wm0 = (warp / WARPS_N) * (BM / WARPS_M), wn = warp % WARPS_N, wn0 = wn * (BN / WARPS_N);
    const int total = *total_tiles;

    for (int wid = blockIdx.x; wid < total; wid += (int)gridDim.x) {
        __syncthreads();  // 2026-09-28: the previous item is done with shared memory
        const unsigned expert_id = worklist[wid * 2 + 0];
        const unsigned packed = worklist[wid * 2 + 1];
        const unsigned cta_m = (packed >> 6) * BM;
        const unsigned cta_n = (packed & 0x3Fu) * BN;
        const int m_start = expert_offsets[expert_id];
        const int M_expert = expert_offsets[expert_id + 1] - m_start;
        const unsigned char* G = (const unsigned char*)G_weight_ptrs[expert_id];
        const unsigned char* U = (const unsigned char*)U_weight_ptrs[expert_id];
        if (G == 0 || U == 0) continue;
        for (unsigned i = threadIdx.x; i < (unsigned)BM; i += THREADS) {
            const unsigned m = cta_m + i;
            int t = -1;
            if (m < (unsigned)M_expert) {
                const int s2 = m_start + (int)m;
                t = sorted_token_ids ? sorted_token_ids[s2] : s2;
            }
            sTok[i] = t;
        }
        __syncthreads();
        const int rows_valid = M_expert - (int)cta_m;
        const unsigned long long w_off = (unsigned long long)cta_n * K;
        const unsigned s_off = (cta_n / SCALE_BLOCK) * k_blocks;
        // 2026-09-28: The gate tile is kept only as its BF16 values (all the epilogue reads), packed two per register,
        // so the up tile's accumulators fit beside it at two CTAs per SM.
        unsigned gpk[MI][NI][2];
        {
            float og[MI][NI][4];
            tile_mma<BM, BN, WARPS_M, WARPS_N, STAGES, true, 1>(
                smem, A_fp8, a_scale, G + w_off, (const float*)G_scale_ptrs[expert_id] + s_off, K, rows_valid, og);
            #pragma unroll
            for (int mi = 0; mi < MI; mi++)
                #pragma unroll
                for (int ni = 0; ni < NI; ni++)
                    #pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const __nv_bfloat162 v = __floats2bfloat162_rn(og[mi][ni][2 * h], og[mi][ni][2 * h + 1]);
                        gpk[mi][ni][h] = *reinterpret_cast<const unsigned*>(&v);
                    }
        }
        __syncthreads();  // 2026-09-28: every warp is done with the gate tile's stages before the up tile refills them
        float og[MI][NI][4];
        tile_mma<BM, BN, WARPS_M, WARPS_N, STAGES, true, 1>(
            smem, A_fp8, a_scale, U + w_off, (const float*)U_scale_ptrs[expert_id] + s_off, K, rows_valid, og);

        // 2026-09-28: r in place of the up accumulator og, and each thread's max |r| per row, reduced over the quad then the warps.
        float rmax[MI][2];
        #pragma unroll
        for (int mi = 0; mi < MI; mi++) {
            rmax[mi][0] = 0.0f; rmax[mi][1] = 0.0f;
            #pragma unroll
            for (int ni = 0; ni < NI; ni++)
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const __nv_bfloat162 gb = *reinterpret_cast<const __nv_bfloat162*>(&gpk[mi][ni][e >> 1]);
                    const float g = __bfloat162float((e & 1) ? gb.y : gb.x);
                    const float u = __bfloat162float(__float2bfloat16(og[mi][ni][e]));
                    const float sigmoid_g = 1.0f / (1.0f + __expf(-g));
                    const float r = __bfloat162float(__float2bfloat16(g * sigmoid_g * u));
                    og[mi][ni][e] = r;
                    rmax[mi][e >> 1] = fmaxf(rmax[mi][e >> 1], fabsf(r));
                }
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                rmax[mi][h] = fmaxf(rmax[mi][h], __shfl_xor_sync(0xffffffffu, rmax[mi][h], 1));
                rmax[mi][h] = fmaxf(rmax[mi][h], __shfl_xor_sync(0xffffffffu, rmax[mi][h], 2));
                if (tig == 0) row_max[wm0 + mi * 16 + gid + h * 8][wn] = rmax[mi][h];
            }
        }
        __syncthreads();
        #pragma unroll
        for (int mi = 0; mi < MI; mi++)
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const unsigned r = wm0 + mi * 16 + gid + h * 8;
                if ((int)r >= rows_valid) continue;
                float gmax = 0.0f;
                #pragma unroll
                for (int w = 0; w < WARPS_N; w++) gmax = fmaxf(gmax, row_max[r][w]);
                float scale = gmax / 448.0f;
                if (scale < 1e-12f) scale = 1e-12f;
                const unsigned long long orow = (unsigned long long)(m_start + (int)(cta_m + r));
                if (wn == 0 && tig == 0) a_scale_out[orow * (N / SCALE_BLOCK) + cta_n / SCALE_BLOCK] = scale;
                #pragma unroll
                for (int ni = 0; ni < NI; ni++) {
                    float v0 = og[mi][ni][h * 2] / scale, v1 = og[mi][ni][h * 2 + 1] / scale;
                    v0 = fmaxf(fminf(v0, 448.0f), -448.0f);
                    v1 = fmaxf(fminf(v1, 448.0f), -448.0f);
                    const unsigned short q = (unsigned short)__nv_cvt_float_to_fp8(v0, __NV_SATFINITE, __NV_E4M3) |
                        ((unsigned short)__nv_cvt_float_to_fp8(v1, __NV_SATFINITE, __NV_E4M3) << 8);
                    *(unsigned short*)&out_fp8[orow * N + cta_n + wn0 + ni * 8 + tig * 2] = q;
                }
            }
    }
}

}
// 2026-09-28: Gate/up + SiLU + E4M3 quant (see gateup_silu_body): 64 x 128 items (work-list m_tile 64, n_tiles
// N / 128), 256 threads, dynamic shared memory SmemBytes<64, 128, 3> (37.8 KiB) plus 1 KiB static. `_w2` runs two CTAs
// per SM (128 registers), faster from a few thousand tokens; `_w1` one CTA per SM, faster below.
#define GATEUP_SILU_PARAMS                                                                                          \
    const unsigned char* __restrict__ A_fp8, const float* __restrict__ a_scale,                                     \
        const unsigned long long* __restrict__ G_weight_ptrs, const unsigned long long* __restrict__ G_scale_ptrs,   \
        const unsigned long long* __restrict__ U_weight_ptrs, const unsigned long long* __restrict__ U_scale_ptrs,   \
        unsigned char* __restrict__ out_fp8, float* __restrict__ a_scale_out, const int* __restrict__ expert_offsets, \
        const int* __restrict__ sorted_token_ids, unsigned int N, unsigned int K,                                    \
        const unsigned int* __restrict__ worklist, const int* __restrict__ total_tiles
#define GATEUP_SILU_ARGS A_fp8, a_scale, G_weight_ptrs, G_scale_ptrs, U_weight_ptrs, U_scale_ptrs, out_fp8, a_scale_out, \
                         expert_offsets, sorted_token_ids, N, K, worklist, total_tiles
extern "C" __global__ void __launch_bounds__(256, 2) moe_w8a8_gateup_silu_e4m3_w2(GATEUP_SILU_PARAMS) {
    e4m3g::gateup_silu_body(GATEUP_SILU_ARGS);
}
extern "C" __global__ void __launch_bounds__(256, 1) moe_w8a8_gateup_silu_e4m3_w1(GATEUP_SILU_PARAMS) {
    e4m3g::gateup_silu_body(GATEUP_SILU_ARGS);
}
