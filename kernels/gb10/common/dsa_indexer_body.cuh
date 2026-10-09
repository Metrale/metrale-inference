// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-09: The bodies of the DSA indexer's selection kernels, shared by the one-sequence
// entries (dsa_indexer.cu) and the entries that run one sequence per grid row
// (dsa_indexer_rows.cu). Moved here from dsa_indexer.cu: each body is the kernel's code with
// the block index it read passed in, so the entries in dsa_indexer.cu run what they ran.
//
// Owner: gb10 kernels.
// Invariants:
// - Each body computes, for its (pool, query row) or query row, exactly the expressions in
//   exactly the order its dsa_indexer.cu entry computed before the move.
// - dsa_pool_key and dsa_pool_slot are the per-channel and per-slot halves of
//   dsa_kpool_compress; a pool key computed through them is the value dsa_kpool_compress
//   stores, wherever the caller keeps it.

#pragma once

#include <cuda_bf16.h>
#include <math_constants.h>
#include <float.h>
#include <limits.h>

// 2026-10-09: Indexer row addressing. Flat (bt NULL): row `raw` of a [capacity, D] buffer at
// element raw * D. Paged: the row sits in physical block bt[raw / bs] of a pool whose blocks
// are blk_elems BF16 elements apart, at slot raw % bs. valid NULL means every row below S is
// valid (a paged cache writes every row below the sequence length).
__device__ __forceinline__ size_t dsa_row_elem(long long raw, unsigned int D, const int* bt,
                                               unsigned int bs, unsigned int blk_elems) {
    if (bt == nullptr) return (size_t)raw * D;
    return (size_t)bt[raw / bs] * blk_elems + (size_t)(raw % bs) * D;
}
__device__ __forceinline__ bool dsa_row_valid(const unsigned char* valid, long long raw) {
    return valid == nullptr || valid[raw] != 0;
}

#define DSA_INVALID (-1)

// 2026-09-25: Replay-safe geometry. A CUDA graph fixes every scalar argument at capture time,
// but S, the pool counts, the top-k tile and select_k grow with the context. Each selector
// kernel therefore takes `geom`, a 5-int device vector that dsa_write_geom fills once per
// step from seq_len. When non-null it overrides the scalar arguments; when null the scalars
// are used as passed. Pool-indexed blocks past the live pool count return at once, so a
// ceiling launch can fix the grid at the context ceiling. Slots: S (tokens in the cache),
// pools including the trailing partial one, complete pools, select_k, top-k tile width.

#define DSA_GEOM_S        0
#define DSA_GEOM_NPOOLS_F 1
#define DSA_GEOM_NPOOLS   2
#define DSA_GEOM_SELECT_K 3
#define DSA_GEOM_NP2      4
#define DSA_GEOM_SLOTS    5

// 2026-09-25: Fills the geom slots from seq_len[0]. `tile` is the top-k tile width (a power of
// two); np2 = min(max(2, next power of two >= complete pools), tile).
__device__ __forceinline__ void dsa_write_geom_body(
    const int* __restrict__ seq_len,
    int* __restrict__ geom,
    unsigned int KP,
    unsigned int topk,
    unsigned int tile
) {
    const int S = seq_len[0];
    const int np = S / (int)KP;


    int np2 = 2;
    while (np2 < np && np2 < (int)tile) np2 <<= 1;
    const int cap = (int)(topk / KP);
    geom[DSA_GEOM_S] = S;
    geom[DSA_GEOM_NPOOLS_F] = (S + (int)KP - 1) / (int)KP;
    geom[DSA_GEOM_NPOOLS] = np;
    geom[DSA_GEOM_SELECT_K] = np < cap ? np : cap;
    geom[DSA_GEOM_NP2] = np2;
}

// 2026-09-25: Copies one staged indexer row (k and gate) into the cache at the device-side
// row pos[0] and marks it valid, so a replayed graph writes the live row.
__device__ __forceinline__ void dsa_indexer_store_body(
    const __nv_bfloat16* __restrict__ stage_k,
    const __nv_bfloat16* __restrict__ stage_gate,
    const int* __restrict__ pos,
    __nv_bfloat16* __restrict__ k_normed,
    __nv_bfloat16* __restrict__ gate,
    unsigned char* __restrict__ valid,
    unsigned int D,
    const int* __restrict__ bt,
    unsigned int bs,
    unsigned int blk_elems
) {
    const size_t base = dsa_row_elem(pos[0], D, bt, bs, blk_elems);
    for (unsigned int d = threadIdx.x; d < D; d += blockDim.x) {
        k_normed[base + d] = stage_k[d];
        gate[base + d] = stage_gate[d];
    }
    if (threadIdx.x == 0 && valid != nullptr) valid[pos[0]] = 1;
}

// 2026-10-09: Slot s of pool p: its raw token index when the token is in range and valid,
// else DSA_INVALID. dsa_kpool_compress stores it as the pool's index for that slot.
__device__ __forceinline__ int dsa_pool_slot(int first_key, unsigned int p, unsigned int KP,
                                             unsigned int s, unsigned int S,
                                             const unsigned char* __restrict__ valid) {
    long long raw = (long long)first_key + (long long)p * KP + s;
    bool in_range = raw >= 0 && raw < (long long)S;
    bool ok = in_range && dsa_row_valid(valid, raw);
    return ok ? (int)raw : DSA_INVALID;
}

// 2026-09-25: Channel d of pool p's key: a softmax over the pool's slots of gate + ape,
// weighting the slots' keys. Invalid slots weigh 0; a pool with no valid slot has sum 0, so
// its key is 0. KP must be <= 8 (lg[8]).
__device__ __forceinline__ float dsa_pool_key(
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ gate,
    const unsigned char* __restrict__ valid,
    const float* __restrict__ ape,
    unsigned int S,
    unsigned int D,
    unsigned int KP,
    int first_key,
    const int* __restrict__ bt,
    unsigned int bs,
    unsigned int blk_elems,
    unsigned int p,
    unsigned int d
) {
    float mx = -CUDART_INF_F;
    float lg[8];
    for (unsigned int s = 0; s < KP && s < 8; ++s) {
        long long raw = (long long)first_key + (long long)p * KP + s;
        bool in_range = raw >= 0 && raw < (long long)S;
        bool ok = in_range && dsa_row_valid(valid, raw);
        lg[s] = ok ? (__bfloat162float(gate[dsa_row_elem(raw, D, bt, bs, blk_elems) + d])
                     + ape[s * D + d])
                   : -CUDART_INF_F;
        mx = fmaxf(mx, lg[s]);
    }
    float sum = 0.0f;
    for (unsigned int s = 0; s < KP && s < 8; ++s) {
        lg[s] = (lg[s] == -CUDART_INF_F) ? 0.0f : __expf(lg[s] - mx);
        sum += lg[s];
    }
    // 2026-09-25: A pool with no valid slot has sum 0, so its weights and key are 0.
    float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    float acc = 0.0f;
    for (unsigned int s = 0; s < KP && s < 8; ++s) {
        long long raw = (long long)first_key + (long long)p * KP + s;
        bool in_range = raw >= 0 && raw < (long long)S;
        bool ok = in_range && dsa_row_valid(valid, raw);
        if (ok)
            acc += lg[s] * inv
                   * __bfloat162float(k[dsa_row_elem(raw, D, bt, bs, blk_elems) + d]);
    }
    return acc;
}

// 2026-09-25: The score of one (query, pool): sum_h weights[h] * relu(scale * dot(q_h, key)).
// One warp per head: each lane accumulates every 32nd product, a shuffle tree reduces them,
// and the head's term lands in sh[h]. Thread 0 then sums sh[0..H) in head order and returns
// it (other threads return 0). `q` is the query row's [H, D], `weights` its [H]; sh holds H
// floats. Every thread of the block must call it (it synchronises).
__device__ __forceinline__ float dsa_score_heads(
    const float* __restrict__ q,
    const float* __restrict__ pk,
    const float* __restrict__ weights,
    unsigned int H,
    unsigned int D,
    float scale,
    float* sh,
    unsigned int tid
) {
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int nwarps = (blockDim.x + 31u) / 32u;
    for (unsigned int h = warp; h < H; h += nwarps) {
        const float* __restrict__ qh = q + (size_t)h * D;
        float dot = 0.0f;
        for (unsigned int d = lane; d < D; d += 32u) dot += qh[d] * pk[d];
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, off);
        if (lane == 0) sh[h] = weights[h] * fmaxf(scale * dot, 0.0f);
    }
    __syncthreads();
    float acc = 0.0f;
    if (tid == 0) {
        for (unsigned int h = 0; h < H; ++h) acc += sh[h];
    }
    return acc;
}

// 2026-09-25: Deterministic top-k over the pools of query row r: a tiled bitonic select (see
// dsa_topk_pools in dsa_indexer.cu). Every thread of the block must call it.
__device__ __forceinline__ void dsa_topk_pools_body(
    const float* __restrict__ scores,
    int* __restrict__ selected,
    unsigned int P,
    unsigned int NP2,
    unsigned int select_k,
    const int* __restrict__ geom,
    unsigned int r
) {
    if (geom) {
        P = (unsigned int)geom[DSA_GEOM_NPOOLS];
        NP2 = (unsigned int)geom[DSA_GEOM_NP2];
        select_k = (unsigned int)geom[DSA_GEOM_SELECT_K];
    }
    // 2026-09-25: Under a ceiling launch the dynamic shared memory is sized for the ceiling;
    // the walk still runs over this step's NP2 and P.
    extern __shared__ char raw_sh[];
    const unsigned int T = NP2;
    float* sv = (float*)raw_sh;
    int*   si = (int*)(raw_sh + (size_t)(2 * T) * sizeof(float));
    const unsigned int tid = threadIdx.x;

    // 2026-09-25: Running best list, descending; starts as padding that sorts last on both keys.
    for (unsigned int i = tid; i < T; i += blockDim.x) {
        sv[i] = -FLT_MAX;
        si[i] = INT_MAX;
    }
    __syncthreads();

#define DSA_TOPK_GT(a, b) ((sv[(a)] > sv[(b)]) || (sv[(a)] == sv[(b)] && si[(a)] < si[(b)]))
#define DSA_TOPK_SWAP(a, b)                                                                  \
    do {                                                                                     \
        float tv_ = sv[(a)]; sv[(a)] = sv[(b)]; sv[(b)] = tv_;                               \
        int   ti_ = si[(a)]; si[(a)] = si[(b)]; si[(b)] = ti_;                               \
    } while (0)

    for (unsigned int base = 0; base < P; base += T) {
        // 2026-09-25: Candidate tile into [T, 2T); a short tile pads with the same sentinel.
        for (unsigned int i = tid; i < T; i += blockDim.x) {
            unsigned int idx = base + i;
            sv[T + i] = (idx < P) ? scores[(size_t)r * P + idx] : -FLT_MAX;
            si[T + i] = (idx < P) ? (int)idx : INT_MAX;
        }
        __syncthreads();

        // 2026-09-25: Bitonic sort of the candidate tile, descending.
        for (unsigned int k = 2; k <= T; k <<= 1) {
            for (unsigned int j = k >> 1; j > 0; j >>= 1) {
                for (unsigned int i = tid; i < T; i += blockDim.x) {
                    unsigned int l = i ^ j;
                    if (l > i) {
                        bool gt = DSA_TOPK_GT(T + i, T + l);
                        bool want_desc = ((i & k) == 0);
                        if (want_desc != gt) DSA_TOPK_SWAP(T + i, T + l);
                    }
                }
                __syncthreads();
            }
        }

        // 2026-09-25: Half-cleaner across the two descending runs: pairing best[i] with
        // cand[T-1-i] leaves the top T of both in [0, T), bitonic but not yet sorted.
        for (unsigned int i = tid; i < T; i += blockDim.x) {
            unsigned int a = i, b = T + (T - 1 - i);
            if (!DSA_TOPK_GT(a, b)) DSA_TOPK_SWAP(a, b);
        }
        __syncthreads();

        // 2026-09-25: Bitonic merge restores descending order over [0, T).
        for (unsigned int j = T >> 1; j > 0; j >>= 1) {
            for (unsigned int i = tid; i < T; i += blockDim.x) {
                unsigned int l = i ^ j;
                if (l > i && !DSA_TOPK_GT(i, l)) DSA_TOPK_SWAP(i, l);
            }
            __syncthreads();
        }
    }

#undef DSA_TOPK_GT
#undef DSA_TOPK_SWAP

    for (unsigned int i = tid; i < select_k; i += blockDim.x)
        selected[(size_t)r * select_k + i] = (si[i] == INT_MAX) ? DSA_INVALID : si[i];
}

// 2026-09-25: Expand query row r's selected pools into raw token ids (see
// dsa_expand_selection in dsa_indexer.cu). 2026-10-09: with SLOTS_FROM_GEOM the pools' token
// ids are recomputed (dsa_pool_slot) instead of read from the pool_indices
// dsa_kpool_compress stored, which are the same values; pool_indices is then not read.
// Every thread of the block must call it.
template <bool SLOTS_FROM_GEOM>
__device__ __forceinline__ void dsa_expand_selection_body(
    const int* __restrict__ selected,
    const int* __restrict__ pool_indices,
    const unsigned char* __restrict__ valid_cand,
    const unsigned char* __restrict__ valid_keys,
    const int* __restrict__ q_pos,
    const unsigned char* __restrict__ q_mask,
    int* __restrict__ out,
    unsigned int P,
    unsigned int KP,
    unsigned int S,
    unsigned int select_k,
    unsigned int width,
    int first_key,
    int always_tail,
    const int* __restrict__ geom,
    unsigned int r
) {
    const unsigned int tid = threadIdx.x;
    if (geom) {
        S = (unsigned int)geom[DSA_GEOM_S];
        P = (unsigned int)geom[DSA_GEOM_NPOOLS];
        select_k = (unsigned int)geom[DSA_GEOM_SELECT_K];
    }
    int* row = out + (size_t)r * width;

    for (unsigned int i = tid; i < width; i += blockDim.x) row[i] = DSA_INVALID;
    __syncthreads();
    if (q_mask[r] == 0) return;   // 2026-09-25: a masked query selects nothing; the row stays all -1

    // 2026-09-25: Per-row select_k. The scalar select_k is the pass's, planned from the pass's
    // cache length, and a multi-row pass plans it from the group's final length. Clamping it
    // to this row's own pool count (q_pos[r] + 1) / KP gives the row the select_k, and so the
    // tail base below, that a single-row pass at that row's length plans. `selected` keeps
    // the pass stride; only the count is per row. The tail slot matters because
    // glm5next_dsa_mla_decode_fp8 splits the row into NUM_WARPS (16) slices merged across
    // warps. Measured 2026-09-06 on that kernel without the clamp: 14 of 18 configurations
    // where a tail token crossed a slice boundary differed, by up to 2 BF16 ulp.
    const unsigned int row_pools = (unsigned int)(q_pos[r] + 1) / KP;
    const unsigned int row_select_k = (row_pools < select_k) ? row_pools : select_k;

    for (unsigned int j = tid; j < row_select_k; j += blockDim.x) {
        int p = selected[(size_t)r * select_k + j];
        bool ok = (p >= 0) && (valid_cand[(size_t)r * P + p] != 0);
        for (unsigned int s = 0; s < KP; ++s) {
            unsigned int w = j * KP + s;
            if (w < width) {
                if (SLOTS_FROM_GEOM) {
                    row[w] = ok ? dsa_pool_slot(first_key, (unsigned int)p, KP, s, S, valid_keys)
                                : DSA_INVALID;
                } else {
                    row[w] = ok ? pool_indices[(size_t)p * KP + s] : DSA_INVALID;
                }
            }
        }
    }

    if (always_tail) {
        // 2026-09-25: The in-progress pool as raw indices. vis_count, the number of valid keys
        // at or before q_pos[r], is an integer count split across all threads, so it is exact.
        // It is not taken as q_pos[r] + 1 - first_key, because valid_keys may mark keys at or
        // below q_pos[r] invalid. always_tail is kernel-uniform and r is block-uniform, so
        // every thread reaches the barrier below; the q_mask return above is whole-block as
        // well.
        __shared__ int vis_warp[32];
        const int qp = q_pos[r];
        int local = 0;
        for (unsigned int t = tid; t < S; t += blockDim.x)
            if ((int)t <= qp && dsa_row_valid(valid_keys, t)) ++local;
        for (int off = 16; off > 0; off >>= 1)
            local += __shfl_down_sync(0xffffffffu, local, off);
        const unsigned int lane = tid & 31u;
        const unsigned int warp = tid >> 5;
        if (lane == 0) vis_warp[warp] = local;
        __syncthreads();
        if (tid != 0) return;
        const unsigned int nwarps = (blockDim.x + 31u) / 32u;
        int vis_count = 0;
        for (unsigned int w = 0; w < nwarps; ++w) vis_count += vis_warp[w];

        int tail_count = vis_count % (int)KP;
        int tail_start = first_key + vis_count - tail_count;
        unsigned int base = row_select_k * KP;
        for (unsigned int t = 0; t + 1 < KP; ++t) {
            long long idx = (long long)tail_start + t;
            bool ok = ((int)t < tail_count) && idx >= 0 && idx < (long long)S
                      && ((int)idx <= q_pos[r]) && dsa_row_valid(valid_keys, idx);
            if (base + t < width) row[base + t] = ok ? (int)idx : DSA_INVALID;
        }
    }
}
