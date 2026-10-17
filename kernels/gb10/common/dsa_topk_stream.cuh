// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-17: A streaming per-row top-k of the DSA pool scores, kept by one warp in shared memory
// while the scores are produced, so the [rows, pools] score row is never written or re-read
// (`METRALE_GLM_DSA_SELECT_FUSED`, dsa_select_fused_pipe in dsa_index_scores_pipe.cu).
//
// Owner: gb10 kernels.
// Invariants:
// - The order is dsa_topk_pools's total order: score descending, then pool index ascending, over
//   every pool of the row (a non-candidate's -FLT_MAX included). Keys are dsa_tk of the score
//   (order-preserving, -0.0 keyed as +0.0, so key equality is value equality), so for NaN-free
//   scores the result is dsa_topk_pools_radix's selection, slot for slot.
// - Pools are offered in ascending order, 64 per call (lane l holds pools 2 l and 2 l + 1 of
//   the tile), and the buffer keeps them in that order, so among equal keys the buffer order is
//   the pool order.
// - The buffer always holds every offered pool that can still be in the row's top select_k:
//   a pool is refused only when select_k buffered pools precede it in the total order (key below
//   the cut, or equal to it at a larger index); a compaction keeps exactly the top select_k of the
//   buffer (all keys above the cut, then the first `need` at the cut, in buffer order).
// - Warp-only: no __syncthreads, so a caller may run it inside a CTA-wide loop on any subset of
//   warps (a warp-uniform condition).

#pragma once

#include <float.h>
#include <limits.h>
#include <stdint.h>

// 2026-10-17: select_k <= DSA_STK_MAXK (the radix select's bound). A warp buffers CAP entries, 192
// past select_k, so the worst pool order (ascending scores) compacts once per 3 tiles; 8 warps x
// 704 x 8 B = 45,056 B beside the 52,992 B ring is under the 99 KB per-block opt-in.
#define DSA_STK_MAXK 512u
#define DSA_STK_CAP (DSA_STK_MAXK + 192u)
#define DSA_STK_FULL 0xFFFFFFFFu

// 2026-10-12: The radix select's order-preserving key (moved here from dsa_indexer_rows.cu on
// 2026-10-17 so both selections use the one definition).
__device__ __forceinline__ unsigned int dsa_tk(float f) {
    unsigned int u = __float_as_uint(f);
    if (u == 0x80000000u) u = 0u;
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// 2026-10-17: One warp's running selection: `cnt` buffered entries in key[] / idx[] (CAP each),
// and once a compaction has run (`have`), the cut `thr` a later pool must exceed to be kept.
struct DsaStk {
    unsigned int* key;
    int* idx;
    unsigned int cnt;
    unsigned int thr;
    bool have;
};

__device__ __forceinline__ unsigned int dsa_stk_sum(unsigned int v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(DSA_STK_FULL, v, o);
    return v;
}

// 2026-10-17: Keep the top k (k < cnt) of the buffer in the total order, in buffer order. The cut
// is the largest key t with at least k buffered keys >= t, found bit by bit from the top; then
// every key above it and the first k - #(key > t) keys equal to it are kept.
__device__ __forceinline__ void dsa_stk_compact(DsaStk& s, unsigned int k, unsigned int lane) {
    unsigned int t = 0u;
    for (int b = 31; b >= 0; --b) {
        const unsigned int c = t | (1u << b);
        unsigned int m = 0u;
        for (unsigned int i = lane; i < s.cnt; i += 32u) m += s.key[i] >= c ? 1u : 0u;
        if (dsa_stk_sum(m) >= k) t = c;
    }
    unsigned int g = 0u;
    for (unsigned int i = lane; i < s.cnt; i += 32u) g += s.key[i] > t ? 1u : 0u;
    const unsigned int need = k - dsa_stk_sum(g);
    const unsigned int lower = (1u << lane) - 1u;
    unsigned int out = 0u, eqs = 0u;
    for (unsigned int base = 0u; base < s.cnt; base += 32u) {
        const unsigned int i = base + lane;
        const bool in = i < s.cnt;
        const unsigned int kv = in ? s.key[i] : 0u;
        const int iv = in ? s.idx[i] : 0;
        const bool eq = in && kv == t;
        const unsigned int be = __ballot_sync(DSA_STK_FULL, eq);
        const bool keep = (in && kv > t) || (eq && eqs + __popc(be & lower) < need);
        const unsigned int bk = __ballot_sync(DSA_STK_FULL, keep);
        // 2026-10-17: every read of this group precedes its writes; writes land at or below
        // base + 31, and the next group reads from base + 32.
        __syncwarp();
        if (keep) {
            const unsigned int o = out + __popc(bk & lower);
            s.key[o] = kv;
            s.idx[o] = iv;
        }
        __syncwarp();
        out += __popc(bk);
        eqs += __popc(be);
    }
    s.cnt = out;
    s.thr = t;
    s.have = true;
}

// 2026-10-17: Offer this lane's two pools (key k0 at pool p0, k1 at p0 + 1; v0 / v1 false past
// the row's pools), in pool order across the warp. Compacts first when the accepted pools would
// overflow the buffer (then cnt > CAP - 64 > DSA_STK_MAXK >= k, so the compaction has work).
__device__ __forceinline__ void dsa_stk_offer(DsaStk& s, unsigned int k0, unsigned int k1, int p0,
                                              bool v0, bool v1, unsigned int k,
                                              unsigned int lane) {
    bool a0 = v0 && (!s.have || k0 > s.thr);
    bool a1 = v1 && (!s.have || k1 > s.thr);
    unsigned int b0 = __ballot_sync(DSA_STK_FULL, a0), b1 = __ballot_sync(DSA_STK_FULL, a1);
    unsigned int n = __popc(b0) + __popc(b1);
    if (n == 0u) return;
    if (s.cnt + n > DSA_STK_CAP) {
        dsa_stk_compact(s, k, lane);
        a0 = v0 && k0 > s.thr;
        a1 = v1 && k1 > s.thr;
        b0 = __ballot_sync(DSA_STK_FULL, a0);
        b1 = __ballot_sync(DSA_STK_FULL, a1);
        n = __popc(b0) + __popc(b1);
        if (n == 0u) return;
    }
    const unsigned int lower = (1u << lane) - 1u;
    const unsigned int pos = s.cnt + __popc(b0 & lower) + __popc(b1 & lower);
    if (a0) {
        s.key[pos] = k0;
        s.idx[pos] = p0;
    }
    if (a1) {
        s.key[pos + (a0 ? 1u : 0u)] = k1;
        s.idx[pos + (a0 ? 1u : 0u)] = p0 + 1;
    }
    s.cnt += n;
    __syncwarp();
}

// 2026-10-17: Offer pools [p_lo, P), all non-candidates (-FLT_MAX), as the tail the walk skipped;
// stops as soon as the cut refuses -FLT_MAX (every later one is refused too).
__device__ __forceinline__ void dsa_stk_offer_floor(DsaStk& s, unsigned int p_lo, unsigned int P,
                                                    unsigned int k, unsigned int lane) {
    const unsigned int kneg = dsa_tk(-FLT_MAX);
    for (unsigned int base = p_lo; base < P; base += 64u) {
        if (s.have && kneg <= s.thr) return;
        const unsigned int p = base + 2u * lane;
        dsa_stk_offer(s, kneg, kneg, (int)p, p < P, p + 1u < P, k, lane);
    }
}

// 2026-10-17: Reduce to the top k and sort them in the total order (bitonic, descending), as the
// reference leaves its best list. Returns the entries kept, min(k, offered pools).
__device__ __forceinline__ unsigned int dsa_stk_finish(DsaStk& s, unsigned int k,
                                                       unsigned int lane) {
    if (s.cnt > k) dsa_stk_compact(s, k, lane);
    const unsigned int n = s.cnt;
    unsigned int n2 = 2u;
    while (n2 < n) n2 <<= 1;
    for (unsigned int i = n + lane; i < n2; i += 32u) {
        s.key[i] = 0u;
        s.idx[i] = INT_MAX;
    }
    __syncwarp();
    for (unsigned int kk = 2u; kk <= n2; kk <<= 1) {
        for (unsigned int j = kk >> 1; j > 0u; j >>= 1) {
            for (unsigned int i = lane; i < n2; i += 32u) {
                const unsigned int l = i ^ j;
                if (l > i) {
                    const bool gt =
                        s.key[i] > s.key[l] || (s.key[i] == s.key[l] && s.idx[i] < s.idx[l]);
                    if (((i & kk) == 0u) != gt) {
                        const unsigned int tk = s.key[i];
                        s.key[i] = s.key[l];
                        s.key[l] = tk;
                        const int ti = s.idx[i];
                        s.idx[i] = s.idx[l];
                        s.idx[l] = ti;
                    }
                }
            }
            __syncwarp();
        }
    }
    return n;
}
