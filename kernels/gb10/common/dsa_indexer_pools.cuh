// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-16: The two DSA indexer bodies that read pool keys from the cache, parameterized over
// how a pool's key is produced (`Src`): dsa_kpool_compress_body (dsa_kpool_compress) and
// dsa_pool_scores_rows_body (dsa_pool_scores_rows). Moved from dsa_indexer.cu and
// dsa_indexer_rows.cu; their BF16 entries instantiate them with DsaBf16Pools, the FP8 entries
// (dsa_indexer_fp8.cu) with the FP8 cache's source.
//
// Owner: gb10 kernels.
// Invariants:
// - With DsaBf16Pools each body computes exactly the expressions, in exactly the order, of the
//   entry it was moved from: Src::key_block runs that entry's channel loop over dsa_pool_key.
// - Src::key_block(p, S, bt, out, base, tid) writes channels [0, D) of pool p's key to
//   out[base + d]. Every thread of the block calls it with the same p and S (the pool loops are
//   uniform); it may synchronise the block.

#pragma once

#include "dsa_indexer_body.cuh"

// 2026-10-16: Pool keys computed from a BF16 cache's rows by dsa_pool_key, one channel per
// thread stride.
struct DsaBf16Pools {
    const __nv_bfloat16* __restrict__ k;
    const __nv_bfloat16* __restrict__ gate;
    const unsigned char* __restrict__ valid;
    const float* __restrict__ ape;
    unsigned int D;
    unsigned int KP;
    int first_key;
    unsigned int bs;
    unsigned int blk_elems;
    __device__ __forceinline__ void key_block(unsigned int p, unsigned int S, const int* bt,
                                              float* out, unsigned int base,
                                              unsigned int tid) const {
        for (unsigned int d = tid; d < D; d += blockDim.x)
            out[base + d] = dsa_pool_key(k, gate, valid, ape, S, D, KP, first_key, bt, bs,
                                         blk_elems, p, d);
    }
};

// 2026-09-25: 1. kpool compression. One block per pool; threads stride over channels. The
// softmax runs over the pool-slot axis, independently per channel. Pool p covers tokens
// first_key + p*KP + s, so left padding before first_key is skipped. A pool is valid only
// if every slot is in range and valid, so a trailing partial pool is never valid.
template <class Src>
__device__ __forceinline__ void dsa_kpool_compress_body(
    const Src& src,
    const unsigned char* __restrict__ valid,
    float* __restrict__ pool_keys,
    int* __restrict__ pool_indices,
    unsigned char* __restrict__ pool_valid,
    unsigned int S,
    unsigned int D,
    unsigned int KP,
    int first_key,
    const int* __restrict__ geom,
    const int* __restrict__ bt
) {
    const unsigned int tid = threadIdx.x;
    // 2026-10-14: Pools [0, n) are written, block x taking x, x + gridDim.x, ... With no geom
    // n is the grid (one pool per block, as before). Under a ceiling launch (geom set, S the
    // ceiling's max_pools * KP) n is the live pool count clamped to the ceiling's
    // max_pools + 1, the old one-block-per-pool grid, so the host may cap the grid without
    // changing which pools are written or how: each pool's writes come from one block running
    // the code below on the same inputs.
    unsigned int n = gridDim.x;
    if (geom) {
        n = min((unsigned int)geom[DSA_GEOM_NPOOLS_F], S / KP + 1u);
        S = (unsigned int)geom[DSA_GEOM_S];
    }

    for (unsigned int p = blockIdx.x; p < n; p += gridDim.x) {
        // 2026-09-25: Slot bookkeeping is the same for every channel, so thread 0 writes it.
        bool all_valid = true;
        for (unsigned int s = 0; s < KP; ++s) {
            const int idx = dsa_pool_slot(first_key, p, KP, s, S, valid);
            all_valid &= idx != DSA_INVALID;
            if (tid == 0) pool_indices[p * KP + s] = idx;
        }
        if (tid == 0) pool_valid[p] = all_valid ? 1 : 0;

        src.key_block(p, S, bt, pool_keys, p * D, tid);
    }
}

// 2026-10-09: The score and candidacy of pool blockIdx.x (and every gridDim.x-th pool after it)
// for row blockIdx.z; see dsa_pool_scores_rows in dsa_indexer_rows.cu. `key` and `heads` are
// shared memory of D and H floats.
template <class Src>
__device__ __forceinline__ void dsa_pool_scores_rows_body(
    const Src& src,
    const float* __restrict__ q,
    const float* __restrict__ weights,
    const int* __restrict__ q_pos,
    float* __restrict__ out,
    unsigned char* __restrict__ valid_cand,
    unsigned int H,
    unsigned int D,
    unsigned int KP,
    int first_key,
    float scale,
    const int* __restrict__ geom,
    const int* __restrict__ bt,
    unsigned int bt_stride,
    unsigned int sc_stride,
    float* key,
    float* heads
) {
    const unsigned int row = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const int* g = geom + (size_t)row * DSA_GEOM_SLOTS;
    const unsigned int S = (unsigned int)g[DSA_GEOM_S];
    const unsigned int P = (unsigned int)g[DSA_GEOM_NPOOLS];
    const int* btr = bt + (size_t)row * bt_stride;
    float* out_r = out + (size_t)row * sc_stride;

    // 2026-10-13: p, and so cand, is the same for every thread of the block: the `continue`
    // is uniform and every thread reaches each __syncthreads.
    for (unsigned int p = blockIdx.x; p < P; p += gridDim.x) {
        // 2026-10-09: dsa_kpool_compress's slot bookkeeping, then dsa_index_scores' candidacy.
        bool all_valid = true;
        int end = DSA_INVALID;
        for (unsigned int s = 0; s < KP; ++s) {
            end = dsa_pool_slot(first_key, p, KP, s, S, nullptr);
            all_valid &= end != DSA_INVALID;
        }
        int end_c = end < 0 ? 0 : (end >= (int)S ? (int)S - 1 : end);
        bool vis = (end_c <= q_pos[row]) && dsa_row_valid(nullptr, end_c);
        bool cand = all_valid && vis;
        if (tid == 0) valid_cand[(size_t)row * sc_stride + p] = cand ? 1 : 0;
        if (!cand) {
            if (tid == 0) out_r[p] = -FLT_MAX;
            continue;
        }

        src.key_block(p, S, btr, key, 0u, tid);
        __syncthreads();
        const float acc = dsa_score_heads(q + (size_t)row * H * D, key,
                                          weights + (size_t)row * H, H, D, scale, heads, tid);
        if (tid == 0) out_r[p] = acc;
        // 2026-10-13: The next pool overwrites `key` and `heads`.
        __syncthreads();
    }
}
