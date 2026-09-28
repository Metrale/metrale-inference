// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-28: The skinny tensor-core decode engine of the WxAy design (one engine for every decode projection with
// M <= 128, format-specific work in a policy). This file holds the engine; each `*.cu` that includes it supplies its
// format policies and the `extern "C"` entries. First user: w8a8_gemv.cu.
//
//   mma_gemv<F, MB, KU>: 8 warps per CTA, F::ROWS weight rows per CTA (the MMA's M side), MB token tiles of 8 (the
//   MMA's N side). K is cut into 128-value chunks; chunk c belongs to warp c mod 8 for every M and is multiplied in
//   ascending c order; KU chunks' weights are loaded before any of them is multiplied; the 8 warp partials are summed
//   in warp order. So a token's result never depends on MB, KU or how many tokens share the launch (row invariance by
//   construction: the canonical row tier).
//
// A policy F supplies:
//   Args, ROWS, LAZY_A (activations loaded with the weights, or inside the multiply loop)
//   Tile tile(a, n0, L, N, K) / Tok tok(a, j, L, M, K)       operand pointers and liveness
//   WReg load_w(tile, c, live) / AReg load_a(tok, c, live)    the loads of chunk c
//   WFrag prep_w(w, L); mma_frag(acc, wf, b, L)               fragment build and the MMA atom (LAZY_A == false)
//   mma_chunk(acc, w, b, L)                                   the same in one step (LAZY_A == true)
//   store(a, C, r, j, n0, L, M, N)                            scale fold and store of token tile j
//
// Owner: gb10 kernels.
// Invariants:
// - K % 128 == 0. Grid (ceil(N / F::ROWS), 1, 1), block (256, 1, 1), no dynamic shared memory.
// - Static shared memory: 8 * min(MB, 8) * 4 * 32 * 4 bytes (at most 32 KiB).

#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

namespace wxay {

constexpr unsigned WARPS = 8;

__device__ __forceinline__ void mma_e4m3_16832(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                               uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

struct Lane {
    unsigned warp, lane, g, t;
};

__device__ __forceinline__ Lane lane_ids() {
    Lane l;
    l.warp = threadIdx.x >> 5;
    l.lane = threadIdx.x & 31u;
    l.g = l.lane >> 2;
    l.t = l.lane & 3u;
    return l;
}

template <class F, int MB, int KU>
__device__ __forceinline__ void mma_gemv(const typename F::Args& a, __nv_bfloat16* __restrict__ C, unsigned M,
                                         unsigned N, unsigned K) {
    const Lane L = lane_ids();
    const unsigned n0 = blockIdx.x * F::ROWS;
    const unsigned num_c = K >> 7;

    const typename F::Tile tile = F::tile(a, n0, L, N, K);
    // 2026-09-28: With lazy activations the token pointers are recomputed where they are used rather than held for every
    // tile (at MB 16 the held array alone would spill).
    typename F::Tok tok[F::LAZY_A ? 1 : MB];
    if constexpr (!F::LAZY_A) {
#pragma unroll
        for (int j = 0; j < MB; j++) tok[j] = F::tok(a, (unsigned)j, L, M, K);
    }

    float acc[MB][4];
#pragma unroll
    for (int j = 0; j < MB; j++)
#pragma unroll
        for (int c = 0; c < 4; c++) acc[j][c] = 0.0f;

    for (unsigned c0 = L.warp; c0 < num_c; c0 += WARPS * KU) {
        typename F::WReg w[KU];
        typename F::AReg b[F::LAZY_A ? 1 : KU][MB];
#pragma unroll
        for (int u = 0; u < KU; u++) {
            const unsigned c = c0 + (unsigned)u * WARPS;
            const bool live = c < num_c;
            w[u] = F::load_w(tile, c, live);
            if constexpr (!F::LAZY_A) {
#pragma unroll
                for (int j = 0; j < MB; j++) b[u][j] = F::load_a(tok[j], c, live);
            }
        }
#pragma unroll
        for (int u = 0; u < KU; u++) {
            const unsigned c = c0 + (unsigned)u * WARPS;
            if (c >= num_c) break;
            if constexpr (F::LAZY_A) {
#pragma unroll
                for (int j = 0; j < MB; j++) {
                    const typename F::AReg bj = F::load_a(F::tok(a, (unsigned)j, L, M, K), c, true);
                    F::mma_chunk(acc[j], w[u], bj, L);
                }
            } else {
                const typename F::WFrag wf = F::prep_w(w[u], L);
#pragma unroll
                for (int j = 0; j < MB; j++) F::mma_frag(acc[j], wf, b[u][j], L);
            }
        }
    }

    // 2026-09-28: The warp partials are reduced JR token tiles at a time (JR = min(MB, 8)), so the shared buffer stays
    // at most 32 KiB for any MB; each tile's sum is still warp 0 + warp 1 + ... + warp 7.
    constexpr int JR = MB < 8 ? MB : 8;
    __shared__ float red[WARPS][JR][4][32];
#pragma unroll
    for (int j0 = 0; j0 < MB; j0 += JR) {
        if (j0 > 0) __syncthreads();
#pragma unroll
        for (int jj = 0; jj < JR; jj++)
#pragma unroll
            for (int c = 0; c < 4; c++) red[L.warp][jj][c][L.lane] = acc[j0 + jj][c];
        __syncthreads();
        for (unsigned jj = L.warp; jj < (unsigned)JR; jj += WARPS) {
            float r[4];
#pragma unroll
            for (int c = 0; c < 4; c++) {
                float v = red[0][jj][c][L.lane];
#pragma unroll
                for (unsigned ww = 1; ww < WARPS; ww++) v += red[ww][jj][c][L.lane];
                r[c] = v;
            }
            F::store(a, C, r, (unsigned)j0 + jj, n0, L, M, N);
        }
    }
}

}  // namespace wxay
