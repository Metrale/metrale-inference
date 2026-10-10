// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-09: The DSA indexer store and selection for the decode rows of several sequences in
// one launch per stage: grid row `row` is one sequence's decode row, on a paged indexer cache,
// with device geometry (the replay-safe path a captured batched decode runs). Per row, each
// entry runs the body the one-sequence entry in dsa_indexer.cu runs for that row alone, so a
// row's selection is the one-sequence ceiling pass's.
//
// Owner: gb10 kernels.
// Invariants:
// - Paged caches only: every row's keys and gates live in the same pool (`k`, `gate`) and are
//   addressed through block-table row `bt + row * bt_stride`; `valid` is NULL (every row below
//   the length is valid). No two grid rows may address one sequence's cache (the launcher
//   passes one decode row per sequence).
// - Row `row` reads and writes: staging row `row` (D elements), positions/q_pos and seq_len
//   entry `row`, geometry slots `row * DSA_GEOM_SLOTS`, query `row * H * D`, head weights
//   `row * H`, scores and candidacy `row * sc_stride`, selected pools `row * sel_stride`,
//   token row `row * width`. sc_stride must be at least the live pool count and sel_stride at
//   least the live select_k of every row (the launcher passes the context ceiling's).
// - dsa_pool_scores_rows fuses dsa_kpool_compress and dsa_index_scores: it computes a
//   candidate pool's key channels with dsa_pool_key into shared memory and scores them with
//   dsa_score_heads, the same expressions on the same values the two-kernel pass stores and
//   reloads; non-candidate pools are scored -FLT_MAX without computing their key, as
//   dsa_index_scores does. dsa_expand_selection_rows recomputes the selected pools' token ids
//   (dsa_pool_slot) instead of reading the pool_indices dsa_kpool_compress would have stored.

#include "dsa_indexer_body.cuh"

// 2026-10-09: grid (1, rows), block >= 1: dsa_indexer_store per row.
extern "C" __global__ void dsa_indexer_store_rows(
    const __nv_bfloat16* __restrict__ stage_k,
    const __nv_bfloat16* __restrict__ stage_gate,
    const int* __restrict__ pos,
    __nv_bfloat16* __restrict__ k_pool,
    __nv_bfloat16* __restrict__ gate_pool,
    unsigned int D,
    const int* __restrict__ bt,
    unsigned int bt_stride,
    unsigned int bs,
    unsigned int blk_elems
) {
    const unsigned int row = blockIdx.y;
    dsa_indexer_store_body(stage_k + (size_t)row * D, stage_gate + (size_t)row * D, pos + row,
                           k_pool, gate_pool, nullptr, D, bt + (size_t)row * bt_stride, bs,
                           blk_elems);
}

// 2026-10-09: grid (rows), block 1: dsa_write_geom per row, from seq_len[row] into geometry
// slots row * DSA_GEOM_SLOTS.
extern "C" __global__ void dsa_write_geom_rows(
    const int* __restrict__ seq_len,
    int* __restrict__ geom,
    unsigned int KP,
    unsigned int topk,
    unsigned int tile
) {
    if (threadIdx.x != 0) return;
    const unsigned int row = blockIdx.x;
    dsa_write_geom_body(seq_len + row, geom + (size_t)row * DSA_GEOM_SLOTS, KP, topk, tile);
}

// 2026-10-09: grid (max_pools, 1, rows), block SCORES_BLOCK: the score and candidacy of pool
// blockIdx.x for row blockIdx.z. Dynamic shared memory: (D + H) floats.
extern "C" __global__ void dsa_pool_scores_rows(
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ gate,
    const float* __restrict__ ape,
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
    unsigned int bs,
    unsigned int blk_elems,
    unsigned int sc_stride
) {
    const unsigned int p = blockIdx.x;
    const unsigned int row = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const int* g = geom + (size_t)row * DSA_GEOM_SLOTS;
    const unsigned int S = (unsigned int)g[DSA_GEOM_S];
    const unsigned int P = (unsigned int)g[DSA_GEOM_NPOOLS];
    if (p >= P) return;
    const int* btr = bt + (size_t)row * bt_stride;

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
    float* out_r = out + (size_t)row * sc_stride;
    if (tid == 0) valid_cand[(size_t)row * sc_stride + p] = cand ? 1 : 0;
    if (!cand) {
        if (tid == 0) out_r[p] = -FLT_MAX;
        return;
    }

    extern __shared__ float sh_rows[];
    float* key = sh_rows;
    float* heads = sh_rows + D;
    for (unsigned int d = tid; d < D; d += blockDim.x)
        key[d] = dsa_pool_key(k, gate, nullptr, ape, S, D, KP, first_key, btr, bs, blk_elems,
                              p, d);
    __syncthreads();
    const float acc = dsa_score_heads(q + (size_t)row * H * D, key, weights + (size_t)row * H,
                                      H, D, scale, heads, tid);
    if (tid == 0) out_r[p] = acc;
}

// 2026-10-09: grid (1, rows), block ROW_BLOCK: dsa_topk_pools per row, over scores row
// row * sc_stride into selected row row * sel_stride. Dynamic shared memory as for
// dsa_topk_pools at the ceiling tile.
extern "C" __global__ void dsa_topk_pools_rows(
    const float* __restrict__ scores,
    int* __restrict__ selected,
    const int* __restrict__ geom,
    unsigned int sc_stride,
    unsigned int sel_stride
) {
    const unsigned int row = blockIdx.y;
    // 2026-10-09: P, NP2 and select_k come from the row's geometry.
    dsa_topk_pools_body(scores + (size_t)row * sc_stride, selected + (size_t)row * sel_stride,
                        0u, 0u, 0u, geom + (size_t)row * DSA_GEOM_SLOTS, 0u);
}

// 2026-10-09: grid (1, rows), block ROW_BLOCK: dsa_expand_selection per row into token row
// row * width. q_mask[0] masks every row (the decode's all-ones mask).
extern "C" __global__ void dsa_expand_selection_rows(
    const int* __restrict__ selected,
    const unsigned char* __restrict__ valid_cand,
    const int* __restrict__ q_pos,
    const unsigned char* __restrict__ q_mask,
    int* __restrict__ out,
    unsigned int KP,
    unsigned int width,
    int first_key,
    int always_tail,
    const int* __restrict__ geom,
    unsigned int sc_stride,
    unsigned int sel_stride
) {
    const unsigned int row = blockIdx.y;
    // 2026-10-09: P, S and select_k come from the row's geometry.
    dsa_expand_selection_body<true>(selected + (size_t)row * sel_stride, nullptr,
                                    valid_cand + (size_t)row * sc_stride, nullptr, q_pos + row,
                                    q_mask, out + (size_t)row * width, 0u, KP, 0u, 0u, width,
                                    first_key, always_tail, geom + (size_t)row * DSA_GEOM_SLOTS,
                                    0u);
}

// 2026-10-10: dsa_index_scores for many query rows (a prefill sub-chunk), tiled: a block scores
// DSA_ST_R query rows against DSA_ST_P pools, each thread 2 rows x 4 pools, with the pools' keys
// staged once per block and each head's query slice once per block and head (both transposed,
// d-major), instead of once per (pool, query) block. Every score is dsa_index_scores's bits:
// per head, partial l (lane l of dsa_score_heads) is ((0 + q[l] k[l]) + q[l + 32] k[l + 32]) + ..
// over d = l, l + 32, .. ; the 32 partials are added in the __shfl_down_sync tree's order
// (t_j[l] = t_{j-1}[l] + t_{j-1}[l + 32 >> j], j = 1..5, as lane 0 receives them), evaluated
// depth first (dsa_st_node); then weights[h] * fmaxf(scale * dot, 0) is summed over h in order
// from 0; non-candidates score -FLT_MAX, and valid_cand is written as there. Host geometry only
// (geom null; the ceiling launch stays on dsa_index_scores). Launch: grid
// (ceil(P / DSA_ST_P), ceil(Q / DSA_ST_R)), block 256, dynamic shared memory
// DSA_ST_D * (DSA_ST_P + DSA_ST_R) * 4 bytes; D == DSA_ST_D.
#define DSA_ST_P 64
#define DSA_ST_R 32
#define DSA_ST_D 128
#define DSA_ST_TP 4
#define DSA_ST_TR 2

// 2026-10-10: Leaf l of the shuffle tree for the thread's 2 x 4 (row, pool) pairs.
template <int L>
__device__ __forceinline__ void dsa_st_leaf(float (&o)[DSA_ST_TR * DSA_ST_TP],
                                            const float* __restrict__ sKt,
                                            const float* __restrict__ sQt,
                                            unsigned int tp, unsigned int tr) {
#pragma unroll
    for (int e = 0; e < DSA_ST_TR * DSA_ST_TP; ++e) o[e] = 0.0f;
#pragma unroll
    for (int d = L; d < DSA_ST_D; d += 32) {
        const float4 kv = *reinterpret_cast<const float4*>(sKt + d * DSA_ST_P + tp * DSA_ST_TP);
        const float2 qv = *reinterpret_cast<const float2*>(sQt + d * DSA_ST_R + tr * DSA_ST_TR);
        const float k4[4] = {kv.x, kv.y, kv.z, kv.w};
        const float q2[2] = {qv.x, qv.y};
#pragma unroll
        for (int a = 0; a < DSA_ST_TR; ++a)
#pragma unroll
            for (int b = 0; b < DSA_ST_TP; ++b) o[a * DSA_ST_TP + b] += q2[a] * k4[b];
    }
}

// 2026-10-10: Tree node (J, L) = node (J - 1, L) + node (J - 1, L + 32 >> J); (0, L) is leaf L.
template <int J, int L>
__device__ __forceinline__ void dsa_st_node(float (&o)[DSA_ST_TR * DSA_ST_TP],
                                            const float* __restrict__ sKt,
                                            const float* __restrict__ sQt,
                                            unsigned int tp, unsigned int tr) {
    if constexpr (J == 0) {
        dsa_st_leaf<L>(o, sKt, sQt, tp, tr);
    } else {
        float b[DSA_ST_TR * DSA_ST_TP];
        dsa_st_node<J - 1, L>(o, sKt, sQt, tp, tr);
        dsa_st_node<J - 1, L + (32 >> J)>(b, sKt, sQt, tp, tr);
#pragma unroll
        for (int e = 0; e < DSA_ST_TR * DSA_ST_TP; ++e) o[e] += b[e];
    }
}

extern "C" __global__ void __launch_bounds__(256) dsa_index_scores_tiled(
    const float* __restrict__ q,
    const float* __restrict__ pool_keys,
    const float* __restrict__ weights,
    const int* __restrict__ pool_indices,
    const unsigned char* __restrict__ pool_valid,
    const unsigned char* __restrict__ valid_keys,
    const int* __restrict__ q_pos,
    float* __restrict__ out,
    unsigned char* __restrict__ valid_cand,
    unsigned int Q,
    unsigned int P,
    unsigned int H,
    unsigned int D,
    unsigned int KP,
    unsigned int S,
    float scale
) {
    extern __shared__ __align__(16) float dsa_st_sh[];
    float* sKt = dsa_st_sh;                     // [D][DSA_ST_P]
    float* sQt = dsa_st_sh + DSA_ST_D * DSA_ST_P; // [D][DSA_ST_R]
    const unsigned int tp = threadIdx.x % (DSA_ST_P / DSA_ST_TP);
    const unsigned int tr = threadIdx.x / (DSA_ST_P / DSA_ST_TP);
    const unsigned int p0 = blockIdx.x * DSA_ST_P + tp * DSA_ST_TP;
    const unsigned int r0 = blockIdx.y * DSA_ST_R + tr * DSA_ST_TR;
    bool cand[DSA_ST_TR * DSA_ST_TP];
    bool any = false;
#pragma unroll
    for (int a = 0; a < DSA_ST_TR; ++a) {
#pragma unroll
        for (int b = 0; b < DSA_ST_TP; ++b) {
            const unsigned int r = r0 + a, p = p0 + b;
            bool c = false;
            if (r < Q && p < P) {
                const int end = pool_indices[p * KP + KP - 1];
                const int end_c = end < 0 ? 0 : (end >= (int)S ? (int)S - 1 : end);
                const bool vis = (end_c <= q_pos[r]) && dsa_row_valid(valid_keys, end_c);
                c = (pool_valid[p] != 0) && vis;
                valid_cand[(size_t)r * P + p] = c ? 1 : 0;
            }
            cand[a * DSA_ST_TP + b] = c;
            any |= c;
        }
    }
    float acc[DSA_ST_TR * DSA_ST_TP];
#pragma unroll
    for (int e = 0; e < DSA_ST_TR * DSA_ST_TP; ++e) acc[e] = 0.0f;
    // 2026-10-10: A tile without a candidate skips the scoring (block-uniform).
    if (__syncthreads_or(any) && D == DSA_ST_D) {
        for (unsigned int e = threadIdx.x; e < DSA_ST_P * DSA_ST_D; e += blockDim.x) {
            const unsigned int pp = e / DSA_ST_D, d = e % DSA_ST_D;
            const unsigned int gp = blockIdx.x * DSA_ST_P + pp;
            sKt[d * DSA_ST_P + pp] = gp < P ? pool_keys[(size_t)gp * DSA_ST_D + d] : 0.0f;
        }
        for (unsigned int h = 0; h < H; ++h) {
            __syncthreads();
            for (unsigned int e = threadIdx.x; e < DSA_ST_R * DSA_ST_D; e += blockDim.x) {
                const unsigned int rr = e / DSA_ST_D, d = e % DSA_ST_D;
                const unsigned int gr = blockIdx.y * DSA_ST_R + rr;
                sQt[d * DSA_ST_R + rr] = gr < Q ? q[((size_t)gr * H + h) * DSA_ST_D + d] : 0.0f;
            }
            __syncthreads();
            float dot[DSA_ST_TR * DSA_ST_TP];
            dsa_st_node<5, 0>(dot, sKt, sQt, tp, tr);
#pragma unroll
            for (int a = 0; a < DSA_ST_TR; ++a) {
                const unsigned int r = r0 + a;
                const float w = r < Q ? weights[(size_t)r * H + h] : 0.0f;
#pragma unroll
                for (int b = 0; b < DSA_ST_TP; ++b)
                    acc[a * DSA_ST_TP + b] += w * fmaxf(scale * dot[a * DSA_ST_TP + b], 0.0f);
            }
        }
    }
#pragma unroll
    for (int a = 0; a < DSA_ST_TR; ++a) {
#pragma unroll
        for (int b = 0; b < DSA_ST_TP; ++b) {
            const unsigned int r = r0 + a, p = p0 + b;
            if (r < Q && p < P) out[(size_t)r * P + p] = cand[a * DSA_ST_TP + b] ? acc[a * DSA_ST_TP + b] : -FLT_MAX;
        }
    }
}
