// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-21: The digit pick and the final sort of the DSA radix top-k, shared by the one-CTA
// dsa_topk_pools_radix (dsa_indexer_rows.cu, where they were written on 2026-10-12) and the
// multi-CTA dsa_topk_mc_* kernels (dsa_topk_mc.cu), so both take the same digits and emit the
// same slots.
//
// Owner: gb10 kernels.
// Invariants:
// - The order is dsa_topk_pools's total order: score descending, then pool index ascending.
//   Keys are dsa_tk (dsa_topk_stream.cuh): key equality is value equality.
// - dsa_tr_sort_emit's output depends only on the multiset of its `take` (score, pool) pairs, not
//   on their order in cv / ci, for NaN-free scores (the order is then strict and total).

#pragma once

#include <float.h>
#include <limits.h>

#define DSA_TR_MAX 512u
#define DSA_TR_WARPS 8u

// 2026-10-21: One radix digit of the select: count(d) is how many of the still-matching keys
// carry digit d (0..255) at this pass; k (>= 1, <= the total) is how many of the select's
// winners are still to be found among them. Warp 0 calls it; the lane that finds the digit d
// holding the k-th largest writes *s_digit = d, *s_need = k minus the keys above d, and
// *s_cnt = count(d). Lane l scans digits 255 - 8 l .. 248 - 8 l (descending).
template <class Count>
__device__ __forceinline__ void dsa_tr_pick(const Count& count, unsigned int k, unsigned int lane,
                                            unsigned int* s_digit, unsigned int* s_need,
                                            unsigned int* s_cnt) {
    unsigned int c[8], sum = 0u;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        c[j] = count(255u - 8u * lane - (unsigned int)j);
        sum += c[j];
    }
    unsigned int incl = sum;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const unsigned int n = __shfl_up_sync(0xFFFFFFFFu, incl, o);
        if (lane >= (unsigned int)o) incl += n;
    }
    const unsigned int excl = incl - sum;
    const unsigned int hit = __ballot_sync(0xFFFFFFFFu, excl < k && incl >= k);
    if (lane == (unsigned int)(__ffs(hit) - 1)) {
        unsigned int cum = excl;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (cum + c[j] >= k) {
                *s_digit = 255u - 8u * lane - (unsigned int)j;
                *s_need = k - cum;
                *s_cnt = c[j];
                break;
            }
            cum += c[j];
        }
    }
}

// 2026-10-21: The final step of the radix top-k, all threads of the block: a bitonic sort of the
// `take` (cv, ci) pairs in shared memory (padded to a power of two with entries that sort last on
// both keys; cv / ci hold at least that many) descending in the total order, then slot i of
// `out` is the i-th pool, DSA_INVALID past `take`, for i < select_k.
__device__ __forceinline__ void dsa_tr_sort_emit(float* cv, int* ci, unsigned int take,
                                                 unsigned int select_k, int* __restrict__ out) {
    const unsigned int tid = threadIdx.x;
    unsigned int n2 = 2u;
    while (n2 < take) n2 <<= 1;
    for (unsigned int i = take + tid; i < n2; i += blockDim.x) {
        cv[i] = -FLT_MAX;
        ci[i] = INT_MAX;
    }
    __syncthreads();
#define DSA_TR_GT(a, b) ((cv[(a)] > cv[(b)]) || (cv[(a)] == cv[(b)] && ci[(a)] < ci[(b)]))
    for (unsigned int kk = 2; kk <= n2; kk <<= 1) {
        for (unsigned int j = kk >> 1; j > 0; j >>= 1) {
            for (unsigned int i = tid; i < n2; i += blockDim.x) {
                const unsigned int l = i ^ j;
                if (l > i) {
                    const bool gt = DSA_TR_GT(i, l);
                    const bool want_desc = (i & kk) == 0u;
                    if (want_desc != gt) {
                        const float tv = cv[i];
                        cv[i] = cv[l];
                        cv[l] = tv;
                        const int ti = ci[i];
                        ci[i] = ci[l];
                        ci[l] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }
#undef DSA_TR_GT
    for (unsigned int i = tid; i < select_k; i += blockDim.x)
        out[i] = (i < take && ci[i] != INT_MAX) ? ci[i] : DSA_INVALID;
}

// 2026-10-21: dsa_tr_sort_emit's slots by rank instead of a bitonic network (45 barrier stages
// at 512 entries): entry i's slot is the number of entries before it in the total order, the
// composite key (dsa_tk(score) << 32) | ~pool compared as one u64 (dsa_tk keeps the float
// order with -0.0 == +0.0, and ~pool puts the smaller pool first among equal scores). The keys
// are distinct (pools are), so the ranks are the bitonic sort's positions. `key` is shared
// memory for `take` u64; cv / ci are only read. All threads of the block.
__device__ __forceinline__ void dsa_tr_rank_emit(const float* cv, const int* ci,
                                                 unsigned long long* key, unsigned int take,
                                                 unsigned int select_k, int* __restrict__ out) {
    const unsigned int tid = threadIdx.x;
    for (unsigned int i = tid; i < take; i += blockDim.x)
        key[i] = ((unsigned long long)dsa_tk(cv[i]) << 32) | (unsigned int)~ci[i];
    __syncthreads();
    for (unsigned int i = tid; i < take; i += 2u * blockDim.x) {
        const unsigned int i2 = i + blockDim.x;
        const unsigned long long a = key[i], b = i2 < take ? key[i2] : ~0ull;
        unsigned int ra = 0u, rb = 0u;
#pragma unroll 8
        for (unsigned int j = 0; j < take; ++j) {
            const unsigned long long kj = key[j];
            ra += kj > a ? 1u : 0u;
            rb += kj > b ? 1u : 0u;
        }
        out[ra] = ci[i];
        if (i2 < take) out[rb] = ci[i2];
    }
    for (unsigned int i = take + tid; i < select_k; i += blockDim.x) out[i] = DSA_INVALID;
}
