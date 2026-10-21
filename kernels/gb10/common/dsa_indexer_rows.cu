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

#include "dsa_indexer_pools.cuh"
#include "dsa_scores_mma.cuh"
#include "dsa_topk_stream.cuh"
#include "dsa_topk_radix.cuh"
#include <stdint.h>

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
// 2026-10-13: grid (x, 1, rows) for any x >= 1: block x scores pools x, x + gridDim.x, ... below
// the row's pool count, so the host caps x instead of launching one block per pool of the
// context ceiling (`--max-seq-len` / KP blocks per row, almost all returning at once at a long
// ceiling). Each pool's writes are the one-block-per-pool kernel's.
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
    extern __shared__ float sh_rows[];
    // 2026-10-16: The body is in dsa_indexer_pools.cuh, shared with the FP8 cache's entry.
    const DsaBf16Pools src{k, gate, nullptr, ape, D, KP, first_key, bs, blk_elems};
    dsa_pool_scores_rows_body(src, q, weights, q_pos, out, valid_cand, H, D, KP, first_key, scale,
                              geom, bt, bt_stride, sc_stride, sh_rows, sh_rows + D);
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

// 2026-10-12: dsa_index_scores on the tensor cores, for a prefill sub-chunk's query rows
// (`METRALE_GLM_DSA_SCORES_MMA`): score[r, p] = sum_h weights[r, h] * relu(scale * <q_{r,h}, k_p>)
// is a multi-query GEMM, M = query rows x 32 heads, N = pools, K = 128, with the ReLU, the head
// weights and the head sum in the epilogue (the [Q, H, P] products are never stored). q and the
// pool keys are rounded to BF16 (RN) and multiplied on mma.sync.m16n8k16 with FP32 accumulation,
// so a score is NOT dsa_index_scores's bits: another precision and summation order (vLLM runs
// this as FP8 MQA logits). Candidacy and the -FLT_MAX of a non-candidate are dsa_index_scores's.
//
// One CTA of 8 warps per 8 query rows, warp w on row 8 * blockIdx.x + w: its 32 heads as two
// m16 tiles whose A fragments (BF16) stay in registers for the whole pool walk. The CTA walks
// the pools in tiles of 64: candidacy of its 8 x 64 (row, pool) pairs first (a tile with none
// is written -FLT_MAX without a key load or an MMA), then the tile's keys to shared memory as
// BF16 ([pool][k], rows padded to 136 halves so ldmatrix is conflict-free), 2 x 8 x 8 MMAs per
// warp, and the epilogue: each lane folds its four heads (g, g + 8, 16 + g, 24 + g) of columns
// 2 * tig + {0, 1}, then three xor-shuffles over g add the 8 lane groups; lane tig writes. The
// order is fixed, so a score depends only on its own row and pool (row- and batch-invariant).
// Launch: grid (ceil(Q / 8), 1, 1), block 256; H = 32 and D = 128 only (else nothing is
// written; the launcher never sends another shape).
// 2026-10-14: The shapes and the per-score arithmetic are in dsa_scores_mma.cuh.

extern "C" __global__ void __launch_bounds__(256) dsa_index_scores_mma(
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
    __shared__ __align__(16) __nv_bfloat16 sB[DSA_SM_PT * DSA_SM_LD];
    __shared__ unsigned char sCand[DSA_SM_QPC][DSA_SM_PT];
    if (H != DSA_SM_H || D != DSA_SM_D) return;
    const unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned int g = lane >> 2, tig = lane & 3u;
    const unsigned int r = blockIdx.x * DSA_SM_QPC + warp;
    const bool live = r < Q;

    uint32_t a[2][8][4];
    float wv[2][2];
    dsa_sm_load_a(q, weights, r, live, g, tig, a, wv);
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(sB);
    const unsigned int ntiles = (P + DSA_SM_PT - 1u) / DSA_SM_PT;
    for (unsigned int t = 0; t < ntiles; ++t) {
        const unsigned int p0 = t * DSA_SM_PT;
        // 2026-10-12: Candidacy, dsa_index_scores's rule, for the CTA's 8 x 64 pairs.
        bool any = false;
        for (unsigned int e = threadIdx.x; e < DSA_SM_QPC * DSA_SM_PT; e += blockDim.x) {
            const unsigned int rr = e / DSA_SM_PT, pp = e % DSA_SM_PT;
            const unsigned int gr = blockIdx.x * DSA_SM_QPC + rr, gp = p0 + pp;
            bool c = false;
            if (gr < Q && gp < P) {
                const int end = pool_indices[(size_t)gp * KP + KP - 1u];
                const int end_c = end < 0 ? 0 : (end >= (int)S ? (int)S - 1 : end);
                const bool vis = (end_c <= q_pos[gr]) && dsa_row_valid(valid_keys, end_c);
                c = (pool_valid[gp] != 0) && vis;
                valid_cand[(size_t)gr * P + gp] = c ? 1 : 0;
            }
            sCand[rr][pp] = c ? 1 : 0;
            any |= c;
        }
        if (!__syncthreads_or(any)) {
            for (unsigned int e = threadIdx.x; e < DSA_SM_QPC * DSA_SM_PT; e += blockDim.x) {
                const unsigned int gr = blockIdx.x * DSA_SM_QPC + e / DSA_SM_PT;
                const unsigned int gp = p0 + e % DSA_SM_PT;
                if (gr < Q && gp < P) out[(size_t)gr * P + gp] = -FLT_MAX;
            }
            continue;
        }
        // 2026-10-12: The tile's keys, rounded to BF16, [pool][k]; pools past P are zero.
        for (unsigned int e = threadIdx.x; e < DSA_SM_PT * (DSA_SM_D / 4u); e += blockDim.x) {
            const unsigned int pp = e / (DSA_SM_D / 4u), c4 = e % (DSA_SM_D / 4u);
            const unsigned int gp = p0 + pp;
            const float4 v = gp < P ? ((const float4*)(pool_keys + (size_t)gp * DSA_SM_D))[c4]
                                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            uint2 packed;
            packed.x = dsa_sm_pack(make_float2(v.x, v.y));
            packed.y = dsa_sm_pack(make_float2(v.z, v.w));
            *(uint2*)(sB + pp * DSA_SM_LD + c4 * 4u) = packed;
        }
        __syncthreads();
        float acc[2][8][4];
        dsa_sm_zero(acc);
        // 2026-10-12: Warp-uniform: a row that sees none of the tile's pools skips its MMAs.
        const bool mine = __any_sync(0xFFFFFFFFu, sCand[warp][lane] | sCand[warp][lane + 32u]);
        if (live && mine) dsa_sm_tile_mma(acc, a, sbase, lane);
        // 2026-10-12: Epilogue (dsa_sm_fold); lane tig of group g == 0 writes.
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const float s = dsa_sm_fold(acc, wv, scale, nt, j);
                const unsigned int pp = (unsigned int)nt * 8u + 2u * tig + (unsigned int)j;
                const unsigned int gp = p0 + pp;
                if (g == 0u && live && gp < P)
                    out[(size_t)r * P + gp] = sCand[warp][pp] ? s : -FLT_MAX;
            }
        }
        // 2026-10-12: sB and sCand are rewritten by the next tile.
        __syncthreads();
    }
}

// 2026-10-12: dsa_topk_pools by radix select (`METRALE_GLM_DSA_TOPK_RADIX`): the same
// selected pools in the same order, without the tiled bitonic walk over every pool. One block
// per query row r. The order is dsa_topk_pools's total order, score descending then pool index
// ascending, over every pool of the row (a non-candidate's -FLT_MAX included):
// 1. Four 8-bit radix passes over the scores' order-preserving keys (dsa_tk; -0.0 keyed as
//    +0.0, so key equality is value equality) find the select_k-th largest key K and how many
//    of the select_k winners equal it (`need`).
// 2. One pass in pool order collects every pool keyed above K (select_k - need of them) and the
//    first `need` pools keyed at K, in ascending pool order: exactly the top select_k of the
//    total order.
// 3. A bitonic sort of those select_k (score, pool) pairs in the total order, as the reference
//    leaves its best list; slot i of the row is the i-th, -1 past the pools of a short row.
// Launch: grid (Q, 1, 1), block 256; select_k <= DSA_TR_MAX.
// 2026-10-21: With `geom` (a ceiling launch: captured decode or verify rows) row r's P and
// select_k are the live ones in geometry slots r * DSA_GEOM_SLOTS, as dsa_topk_pools_body reads
// them, so a replayed graph selects over this step's pools; the static shared memory does not
// depend on them. Rows read scores at stride P and write selected pools at stride select_k, the
// arguments as passed (a ceiling's max_pools and its select_k).
extern "C" __global__ void __launch_bounds__(256) dsa_topk_pools_radix(
    const float* __restrict__ scores,
    int* __restrict__ selected,
    unsigned int Q,
    unsigned int P,
    unsigned int select_k,
    const int* __restrict__ geom
) {
    // 2026-10-21: Row strides are the P and select_k arguments (a ceiling launch's), fixed
    // whatever the rows' live extents.
    const unsigned int sc_stride = P, sel_stride = select_k;
    if (geom) {
        geom += (size_t)blockIdx.x * DSA_GEOM_SLOTS;
        P = (unsigned int)geom[DSA_GEOM_NPOOLS];
        select_k = (unsigned int)geom[DSA_GEOM_SELECT_K];
    }
    __shared__ unsigned int hist[DSA_TR_WARPS][256];
    __shared__ float cv[DSA_TR_MAX];
    __shared__ int ci[DSA_TR_MAX];
    __shared__ unsigned int s_need, s_gt, s_eq, s_wsum[DSA_TR_WARPS], s_digit, s_cnt;
    const unsigned int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    if (r >= Q || select_k == 0u || select_k > DSA_TR_MAX || blockDim.x != 256u) return;
    const float* row = scores + (size_t)r * sc_stride;
    const unsigned int take = select_k < P ? select_k : P;

    if (P > select_k) {
        // 2026-10-12: 1. Radix select of the select_k-th largest key.
        unsigned int prefix = 0u, mask = 0u, k = select_k;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (unsigned int i = tid; i < DSA_TR_WARPS * 256u; i += blockDim.x)
                (&hist[0][0])[i] = 0u;
            __syncthreads();
            for (unsigned int i = tid; i < P; i += blockDim.x) {
                const unsigned int key = dsa_tk(row[i]);
                if ((key & mask) == prefix) atomicAdd(&hist[warp][(key >> shift) & 255u], 1u);
            }
            __syncthreads();
            if (warp == 0u)
                dsa_tr_pick([&](unsigned int d) {
                    unsigned int v = 0u;
#pragma unroll
                    for (unsigned int w = 0; w < DSA_TR_WARPS; ++w) v += hist[w][d];
                    return v;
                }, k, lane, &s_digit, &s_need, &s_cnt);
            __syncthreads();
            prefix |= s_digit << shift;
            k = s_need;
            mask |= 255u << shift;
        }
        // 2026-10-12: 2. Collect: above K anywhere in [0, select_k - need), the first `need`
        // at K, in pool order, after them.
        const unsigned int kstar = prefix, need = k, n_gt = select_k - need;
        if (tid == 0u) {
            s_gt = 0u;
            s_eq = 0u;
        }
        __syncthreads();
        for (unsigned int base = 0; base < P; base += blockDim.x) {
            const unsigned int i = base + tid;
            const float v = i < P ? row[i] : 0.0f;
            const unsigned int key = i < P ? dsa_tk(v) : 0u;
            const bool gt = i < P && key > kstar;
            const bool eq = i < P && key == kstar;
            if (gt) {
                const unsigned int pos = atomicAdd(&s_gt, 1u);
                cv[pos] = v;
                ci[pos] = (int)i;
            }
            const unsigned int bal = __ballot_sync(0xFFFFFFFFu, eq);
            if (lane == 0u) s_wsum[warp] = (unsigned int)__popc(bal);
            __syncthreads();
            unsigned int before = s_eq;
            for (unsigned int w = 0; w < warp; ++w) before += s_wsum[w];
            const unsigned int pos = before + (unsigned int)__popc(bal & ((1u << lane) - 1u));
            if (eq && pos < need) {
                cv[n_gt + pos] = v;
                ci[n_gt + pos] = (int)i;
            }
            __syncthreads();
            if (tid == 0u) {
                unsigned int t = 0u;
                for (unsigned int w = 0; w < DSA_TR_WARPS; ++w) t += s_wsum[w];
                s_eq += t;
            }
            __syncthreads();
        }
    } else {
        for (unsigned int i = tid; i < P; i += blockDim.x) {
            cv[i] = row[i];
            ci[i] = (int)i;
        }
    }
    // 2026-10-12: 3. Bitonic sort of the `take` winners, descending in the total order
    // (dsa_tr_sort_emit since 2026-10-21).
    __syncthreads();
    dsa_tr_sort_emit(cv, ci, take, select_k, selected + (size_t)r * sel_stride);
}
