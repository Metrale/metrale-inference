// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-27: The block-scaled FP8 tile pipeline shared by `moe_w8a8_grouped_gemm_e4m3.cu` (routed experts, work-list
// items) and `fp8_gemm_blockscaled_pipe.cu` (dense projections, one tile per CTA). For one BM x BN output tile:
//   C[m, n] = bf16( sum_f ( sum_(k in f) A[row(m), k] * B[n, k] ) * (a_scale[row(m), f'] * b_scale[n / 128, f']) )
// where f runs over FOLD_STEPS * 64-value chunks of K in ascending order and f' is the 128-K scale block of f. The
// inner sums are native `mma.sync.m16n8k32.e4m3` F32 accumulations in ascending K: one k32 MMA per 32 values, or with
// SPLIT16 two MMAs whose other K half is zeroed, so each MMA sums the same 16 products as a BF16 m16n8k16 MMA over
// the decoded bytes would. Folding: outer += inner * (a_scale * b_scale), then inner = 0.
//
// Pipeline: STAGES-deep cp.async ring of 64-byte K slices of A (rows through sTok) and B with the slice's per-row
// a_scale, XOR-swizzled 64-byte smem rows read with ldmatrix. 2026-10-05: a point of the tile mainloop family
// (gemm_mainloop.cuh): atom bundle `mma16832_e4m3_cpasync_ldsm` (KERNEL_FAMILIES.toml), the multistage schedule,
// and this file's body (the row gather, the a_scale ring and the fold). Rows whose sTok is -1 are zero-filled (a_scale 0) and
// never stored; m16 sub-tiles that lie wholly past `rows_valid` skip their MMAs.
//
// Owner: gb10 kernels.
// Invariants:
// - K % 128 == 0; the B tile's rows are all in range (the caller checks N % BN == 0); BN divides 128.
// - Shared memory: SmemBytes<BM, BN, STAGES>::value bytes of dynamic shared memory, 128-byte aligned.
// - sTok (in that shared memory) holds the tile's A row indices, visible to all threads before `tile_mma`.

#pragma once
#include <cuda_bf16.h>
#include "gemm_mainloop.cuh"

namespace e4m3g {

constexpr int BK = 64;
constexpr int SCALE_BLOCK = 128;

// 2026-10-05: The atom bundle: m16n8k32 E4M3 MMA, 16- and 4-byte async copies, ldmatrix x4, 64-byte rows under
// sigma(2, 4, 3) (the XOR spreads the 8 rows one ldmatrix phase reads over all 8 16-byte bank groups).
using Atoms = gml::Bundle<gml::MmaE4m3M16N8K32, gml::CopyAsync16, gml::CopyAsync4, gml::CopyAsyncGroups,
                          gml::LoadMatrixX4, gml::SwizzledRows<BK, 2, 4, 3>>;

template <int BM, int BN, int STAGES>
struct SmemBytes {
    static constexpr int value = STAGES * (BM + BN) * BK + STAGES * BM * 4 + BM * 4;
};

// 2026-09-27: The shared-memory carve of SmemBytes: A ring, B ring, staged a_scale ring, then the row ids.
template <int BM, int BN, int STAGES>
__device__ __forceinline__ int* tile_rows(unsigned char* smem) {
    return (int*)(smem + STAGES * (BM + BN) * BK + STAGES * BM * 4);
}

// 2026-09-27: One tile into outer[MI][NI][4] (MI = BM / WARPS_M / 16, NI = BN / WARPS_N / 8) in the m16n8 accumulator
// layout. `B_tile` points at the tile's first weight row, `S_row` at its b_scale row; `rows_valid` bounds the MMAs.
template <int BM, int BN, int WARPS_M, int WARPS_N, int STAGES, bool SPLIT16, int FOLD_STEPS>
__device__ __forceinline__ void tile_mma(
    unsigned char* smem,
    const unsigned char* __restrict__ A_fp8,
    const float* __restrict__ a_scale,
    const unsigned char* __restrict__ B_tile,
    const float* __restrict__ S_row,
    unsigned int K,
    int rows_valid,
    float (&outer)[BM / WARPS_M / 16][BN / WARPS_N / 8][4]
) {
    using Rows = Atoms::Rows;
    constexpr int THREADS = WARPS_M * WARPS_N * 32;
    constexpr int WM = BM / WARPS_M;
    constexpr int WN = BN / WARPS_N;
    constexpr int MI = WM / 16;
    constexpr int NI = WN / 8;
    static_assert(MI >= 1 && NI >= 2 && NI % 2 == 0, "warp tile");
    static_assert(128 % BN == 0, "BN divides 128");
    static_assert(FOLD_STEPS >= 1 && SCALE_BLOCK % (FOLD_STEPS * BK) == 0, "a fold stays inside one scale block");

    unsigned char* sA = smem;
    unsigned char* sB = sA + STAGES * BM * BK;
    float* sAS = (float*)(sB + STAGES * BN * BK);
    const int* sTok = tile_rows<BM, BN, STAGES>(smem);

    const unsigned tid = threadIdx.x;
    const unsigned warp = tid >> 5;
    const unsigned lane = tid & 31u;
    const unsigned gid = lane >> 2;
    const unsigned wm0 = (warp / WARPS_N) * WM;
    const unsigned wn0 = (warp % WARPS_N) * WN;
    const unsigned lmat = lane >> 3;
    const unsigned lrow = lane & 7u;
    const unsigned k_blocks = K / SCALE_BLOCK;
    const unsigned n_steps = K / BK;

    auto load_stage = [&](unsigned step, unsigned stage) {
        const unsigned k0 = step * BK;
        unsigned char* a_dst = sA + stage * BM * BK;
        unsigned char* b_dst = sB + stage * BN * BK;
        #pragma unroll
        for (unsigned c = tid; c < (unsigned)BM * 4; c += THREADS) {
            const unsigned row = c >> 2, ch = c & 3u;
            const int t = sTok[row];
            const unsigned char* src = A_fp8 + (unsigned long long)(t < 0 ? 0 : t) * K + k0 + ch * 16;
            Atoms::Copy::copy(gml::smem_addr(a_dst + Rows::offset(row, ch)), src, t >= 0);
        }
        #pragma unroll
        for (unsigned c = tid; c < (unsigned)BN * 4; c += THREADS) {
            const unsigned row = c >> 2, ch = c & 3u;
            Atoms::Copy::copy(gml::smem_addr(b_dst + Rows::offset(row, ch)), B_tile + (unsigned long long)row * K + k0 + ch * 16, true);
        }
        const unsigned kb = k0 / SCALE_BLOCK;
        for (unsigned i = tid; i < (unsigned)BM; i += THREADS) {
            const int t = sTok[i];
            Atoms::SmallCopy::copy(gml::smem_addr(sAS + stage * BM + i), a_scale + (unsigned long long)(t < 0 ? 0 : t) * k_blocks + kb, t >= 0);
        }
    };

    // 2026-09-27: The m16 sub-tiles of this warp that hold at least one live row.
    const int rows_left = rows_valid - (int)wm0;
    const int mi_valid = rows_left <= 0 ? 0 : min(MI, (rows_left + 15) / 16);

    float inner[MI][NI][4];
    #pragma unroll
    for (int mi = 0; mi < MI; mi++)
        #pragma unroll
        for (int ni = 0; ni < NI; ni++)
            #pragma unroll
            for (int r = 0; r < 4; r++) { inner[mi][ni][r] = 0.0f; outer[mi][ni][r] = 0.0f; }

    auto compute_stage = [&](unsigned step, unsigned stage) {
        const unsigned a_base = gml::smem_addr(sA + stage * BM * BK);
        const unsigned b_base = gml::smem_addr(sB + stage * BN * BK);

        #pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            unsigned bf[NI][2];
            #pragma unroll
            for (int j = 0; j < NI / 2; j++) {
                const unsigned nrow = wn0 + j * 16 + lrow + ((lmat >> 1) << 3);
                const unsigned ch = 2 * ks + (lmat & 1u);
                Atoms::Load::load(b_base + Rows::offset(nrow, ch), bf[2 * j][0], bf[2 * j][1], bf[2 * j + 1][0], bf[2 * j + 1][1]);
            }
            #pragma unroll
            for (int mi = 0; mi < MI; mi++) {
                if (mi < mi_valid) {
                    unsigned af[4];
                    const unsigned arow = wm0 + mi * 16 + lrow + ((lmat & 1u) << 3);
                    const unsigned ch = 2 * ks + (lmat >> 1);
                    Atoms::Load::load(a_base + Rows::offset(arow, ch), af[0], af[1], af[2], af[3]);
                    if (SPLIT16) {
                        const unsigned a_lo[4] = {af[0], af[1], 0u, 0u}, a_hi[4] = {0u, 0u, af[2], af[3]};
                        #pragma unroll
                        for (int ni = 0; ni < NI; ni++) {
                            Atoms::Mma::mma(inner[mi][ni], a_lo, bf[ni][0], 0u);
                            Atoms::Mma::mma(inner[mi][ni], a_hi, 0u, bf[ni][1]);
                        }
                    } else {
                        #pragma unroll
                        for (int ni = 0; ni < NI; ni++) Atoms::Mma::mma(inner[mi][ni], af, bf[ni][0], bf[ni][1]);
                    }
                }
            }
        }

        if ((step + 1) % FOLD_STEPS == 0 || step + 1 == n_steps) {
            const float bs = S_row[(step * BK) / SCALE_BLOCK];
            const float* as = sAS + stage * BM;
            #pragma unroll
            for (int mi = 0; mi < MI; mi++) {
                const float s0 = as[wm0 + mi * 16 + gid] * bs;
                const float s1 = as[wm0 + mi * 16 + gid + 8] * bs;
                #pragma unroll
                for (int ni = 0; ni < NI; ni++) {
                    outer[mi][ni][0] += inner[mi][ni][0] * s0;
                    outer[mi][ni][1] += inner[mi][ni][1] * s0;
                    outer[mi][ni][2] += inner[mi][ni][2] * s1;
                    outer[mi][ni][3] += inner[mi][ni][3] * s1;
                    inner[mi][ni][0] = 0.0f; inner[mi][ni][1] = 0.0f;
                    inner[mi][ni][2] = 0.0f; inner[mi][ni][3] = 0.0f;
                }
            }
        }
    };

    gml::multistage<Atoms, STAGES>(n_steps, load_stage, compute_stage);
}

// 2026-09-27: Stores the tile's live rows as BF16: tile row r goes to C row `row_base + r` for r < rows_valid.
template <int BM, int BN, int WARPS_M, int WARPS_N>
__device__ __forceinline__ void tile_store(
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned long long row_base,
    int rows_valid,
    unsigned int cta_n,
    const float (&outer)[BM / WARPS_M / 16][BN / WARPS_N / 8][4]
) {
    gml::store_bf16_rows<BM, BN, WARPS_M, WARPS_N>(C, N, row_base, rows_valid, cta_n, outer);
}

}
