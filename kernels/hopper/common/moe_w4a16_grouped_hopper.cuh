// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-14: Pieces the grouped W4A16 MoE prefill twins share (moe_w4a16_grouped_pipe_hopper.cu,
// moe_w4a16_grouped_wgmma_hopper.cu), kept in one place so their weights round alike and their
// persistent grids count tiles alike.
//
// Owner: hopper kernels.
// Invariants: `pipe_scale_pair` is the one rounding of a staged weight for both twins.

#pragma once

#include <cuda_bf16.h>

// 2026-10-13: One BF16 pair p (each E2M1 * E4M3, exact) times scale2 in FP32, rounded to BF16 as
// the base kernel rounds `lut * fp8 * scale2`.
__device__ __forceinline__ unsigned int pipe_scale_pair(unsigned int p, float s2) {
    const float lo = __uint_as_float(p << 16) * s2;
    const float hi = __uint_as_float(p & 0xFFFF0000u) * s2;
    const __nv_bfloat162 r = __floats2bfloat162_rn(lo, hi);
    return *(const unsigned int*)&r;
}

// 2026-10-14: Exclusive scan of ceil(rows_e / TILE_ROWS) over the experts into first_tile[0 .. E]
// (first_tile[E] = the total), by a block of THREADS threads; thread t owns a contiguous run.
// `warp_tiles` is THREADS / 32 ints of shared memory. Ends with a block barrier; returns the total.
template <int TILE_ROWS, int THREADS>
__device__ __forceinline__ int pipe_scan_tiles(int* first_tile, int* warp_tiles,
                                               const int* __restrict__ expert_offsets,
                                               unsigned int num_experts) {
    const unsigned int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const unsigned int per = (num_experts + THREADS - 1) / THREADS;
    const unsigned int e0 = t * per;
    int mine = 0;
    for (unsigned int i = 0; i < per && e0 + i < num_experts; i++) {
        const int rows = expert_offsets[e0 + i + 1] - expert_offsets[e0 + i];
        mine += rows > 0 ? (rows + TILE_ROWS - 1) / TILE_ROWS : 0;
    }
    int incl = mine;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, incl, d);
        if ((int)lane >= d) incl += v;
    }
    if (lane == 31) warp_tiles[warp] = incl;
    __syncthreads();
    int before = 0, total = 0;
#pragma unroll
    for (int w = 0; w < THREADS / 32; w++) {
        before += w < (int)warp ? warp_tiles[w] : 0;
        total += warp_tiles[w];
    }
    int run = before + incl - mine;
    for (unsigned int i = 0; i < per && e0 + i < num_experts; i++) {
        first_tile[e0 + i] = run;
        const int rows = expert_offsets[e0 + i + 1] - expert_offsets[e0 + i];
        run += rows > 0 ? (rows + TILE_ROWS - 1) / TILE_ROWS : 0;
    }
    if (t == 0) first_tile[num_experts] = total;
    __syncthreads();
    return total;
}

// 2026-10-14: The expert of persistent tile `tile`: the last one whose first tile is at or before
// it (first_tile from pipe_scan_tiles).
__device__ __forceinline__ unsigned int pipe_tile_expert(const int* first_tile,
                                                         unsigned int num_experts,
                                                         unsigned int tile) {
    unsigned int lo = 0, hi = num_experts;
    while (hi - lo > 1) {
        const unsigned int mid = (lo + hi) >> 1;
        if ((unsigned int)first_tile[mid] <= tile) lo = mid; else hi = mid;
    }
    return lo;
}
