// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-14: The arithmetic of the DSA indexer scores on BF16 tensor cores, shared by
// dsa_index_scores_mma (dsa_indexer_rows.cu) and dsa_index_scores_mma_pipe
// (dsa_index_scores_pipe.cu), so the two entries compute every score with the same
// instructions in the same order and differ only in how the keys and the candidacy reach them.
//
// Owner: gb10 kernels.
// Invariants:
// - A score of query row r and pool p depends only on row r's query and head weights and pool
//   p's key: the A fragments are row r's 32 heads rounded to BF16 (RN) in register order
//   a[mt][ks]; each accumulator acc[mt][nt] chains k-steps 0..7 in order (dsa_sm_tile_mma);
//   dsa_sm_fold adds heads (g, g + 8) of m-tile 0, then of m-tile 1, then three xor-shuffles
//   over g (4, 8, 16). FP32 addition is commutative, so after the shuffles every lane of a
//   column holds the same bits.
// - The B tile is [pool][k] BF16 in shared memory at row stride DSA_SM_LD halves.

#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

#define DSA_SM_QPC 8u
#define DSA_SM_PT 64u
#define DSA_SM_D 128u
#define DSA_SM_H 32u
#define DSA_SM_LD (DSA_SM_D + 8u)

__device__ __forceinline__ uint32_t dsa_sm_pack(float2 v) {
    __nv_bfloat162 b = __floats2bfloat162_rn(v.x, v.y);
    return *reinterpret_cast<uint32_t*>(&b);
}

__device__ __forceinline__ void dsa_sm_mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                           uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void dsa_sm_ldsm_x4(uint32_t addr, uint32_t& d0, uint32_t& d1,
                                               uint32_t& d2, uint32_t& d3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(addr));
}

// 2026-10-12: Row r's 32 heads as BF16 A fragments: tile mt holds heads 16 mt .. 16 mt + 15;
// a[mt][ks] = rows (g, g + 8) x cols (2 tig, 2 tig + 1) and (2 tig + 8, 2 tig + 9) of k-step ks.
// A row past Q (live false) loads zeros.
__device__ __forceinline__ void dsa_sm_load_a(const float* __restrict__ q,
                                              const float* __restrict__ weights, unsigned int r,
                                              bool live, unsigned int g, unsigned int tig,
                                              uint32_t (&a)[2][8][4], float (&wv)[2][2]) {
    const float* qr = q + (size_t)(live ? r : 0u) * DSA_SM_H * DSA_SM_D;
    const float2 z = make_float2(0.0f, 0.0f);
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        const unsigned int h0 = (unsigned int)mt * 16u + g, h1 = h0 + 8u;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const unsigned int k0 = (unsigned int)ks * 16u + 2u * tig;
            const float2 x00 = live ? *(const float2*)(qr + h0 * DSA_SM_D + k0) : z;
            const float2 x10 = live ? *(const float2*)(qr + h1 * DSA_SM_D + k0) : z;
            const float2 x01 = live ? *(const float2*)(qr + h0 * DSA_SM_D + k0 + 8u) : z;
            const float2 x11 = live ? *(const float2*)(qr + h1 * DSA_SM_D + k0 + 8u) : z;
            a[mt][ks][0] = dsa_sm_pack(x00);
            a[mt][ks][1] = dsa_sm_pack(x10);
            a[mt][ks][2] = dsa_sm_pack(x01);
            a[mt][ks][3] = dsa_sm_pack(x11);
        }
        wv[mt][0] = live ? weights[(size_t)r * DSA_SM_H + h0] : 0.0f;
        wv[mt][1] = live ? weights[(size_t)r * DSA_SM_H + h1] : 0.0f;
    }
}

__device__ __forceinline__ void dsa_sm_zero(float (&acc)[2][8][4]) {
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[mt][nt][c] = 0.0f;
}

// 2026-10-12: One 64-pool tile: 2 x 8 x 8 MMAs into acc (zeroed by the caller) from the B tile
// at shared address sbase; accumulator acc[mt][nt] takes k-steps 0..7 in order.
__device__ __forceinline__ void dsa_sm_tile_mma(float (&acc)[2][8][4],
                                                const uint32_t (&a)[2][8][4], uint32_t sbase,
                                                unsigned int lane) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const unsigned int row = (unsigned int)nt * 8u + (lane & 7u);
            const unsigned int kof = (unsigned int)kk * 32u + (lane >> 3) * 8u;
            uint32_t b0, b1, b2, b3;
            dsa_sm_ldsm_x4(sbase + (row * DSA_SM_LD + kof) * 2u, b0, b1, b2, b3);
            dsa_sm_mma(acc[0][nt], a[0][2 * kk], b0, b1);
            dsa_sm_mma(acc[1][nt], a[1][2 * kk], b0, b1);
            dsa_sm_mma(acc[0][nt], a[0][2 * kk + 1], b2, b3);
            dsa_sm_mma(acc[1][nt], a[1][2 * kk + 1], b2, b3);
        }
    }
}

// 2026-10-12: The score of column 8 nt + 2 tig + j of the tile: c0/c1 are rows g of columns
// 2 tig, 2 tig + 1; c2/c3 rows g + 8. Each lane folds its four heads, then three xor-shuffles
// over g add the 8 lane groups; every lane of the column ends with the same bits.
__device__ __forceinline__ float dsa_sm_fold(const float (&acc)[2][8][4], const float (&wv)[2][2],
                                             float scale, int nt, int j) {
    float s = wv[0][0] * fmaxf(scale * acc[0][nt][j], 0.0f);
    s += wv[0][1] * fmaxf(scale * acc[0][nt][2 + j], 0.0f);
    s += wv[1][0] * fmaxf(scale * acc[1][nt][j], 0.0f);
    s += wv[1][1] * fmaxf(scale * acc[1][nt][2 + j], 0.0f);
    s += __shfl_xor_sync(0xFFFFFFFFu, s, 4);
    s += __shfl_xor_sync(0xFFFFFFFFu, s, 8);
    s += __shfl_xor_sync(0xFFFFFFFFu, s, 16);
    return s;
}
