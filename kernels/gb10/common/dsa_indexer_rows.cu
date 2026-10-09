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
