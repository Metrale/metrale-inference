// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-21: The GLM DSA decode selection's warp-per-pool indexer scores and the persisted
// pool keys of a paged BF16 indexer cache (IndexerRowFormat::Bf16Pooled):
// - dsa_index_scores_warp: dsa_index_scores's bytes, one warp per (pool, row);
// - dsa_index_scores_warp_paged: the same over the persisted keys, read through the block table,
//   candidacy from the geometry (no dsa_kpool_compress);
// - dsa_pool_keys_store_rows: a pool's key, written once its last row is in the cache.
//
// Owner: gb10 kernels.
// Invariants: every score, candidacy and pool-key byte is the dsa_index_scores /
// dsa_kpool_compress path's (gate examples/glm5next_dsa_scores_mma_bench.rs).

#include "dsa_scores_heads.cuh"

// 2026-10-21: dsa_index_scores with one warp per (pool, query row) instead of one block
// (`METRALE_GLM_DSA_SCORES_WARP`, default on), for the launches dsa_index_scores serves (the
// captured decode's ceiling launch among them). The block stages its query row's [H, D] q and
// [H] weights in shared memory once, and each warp walks pools blockIdx.x * 8 + warp, + 8 *
// gridDim.x, ... with no barrier per pool. Every byte is dsa_index_scores's:
// - candidacy and the -FLT_MAX of a non-candidate are the same expressions;
// - a head's dot is lane l's `dot += q * k` over d = l, l + 32, ... from 0 (the reference's own
//   expressions, so both follow the family's --fmad flag alike), then the same shfl_down tree
//   (16, 8, 4, 2, 1), so lane 0 holds dsa_score_heads's `dot`;
// - its term is the rounded product weights[h] * relu(scale * dot), and lane 0 adds the terms
//   in head order from 0, as thread 0 sums sh[0..H) there (explicit _rn intrinsics where the
//   reference rounds through shared memory, so no FMA can be contracted under any --fmad).
// With `geom` (a ceiling launch) S and the live pools come from it as dsa_index_scores reads
// them; 2026-10-21: row r's from slots r * DSA_GEOM_SLOTS (dsa_write_geom_rows' layout), its
// scores and candidacy at row stride P (the ceiling's), so one launch scores a verify's rows. Launch: grid (any x, Q, 1), block 256; H <= DSA_SW_HMAX, D <= DSA_SW_DMAX and D a
// multiple of 32 (else nothing is written; the launcher never sends another shape).
// 2026-10-21: The body is generic over where a pool's key and candidacy come from (`Keys`):
// dsa_kpool_compress's scratch (DsaSwScratchKeys) or the persisted pool keys of a paged cache
// (DsaSwPersistedKeys, dsa_index_scores_warp_paged). 2026-10-22: DSA_SW_WARPS, DSA_SW_HMAX,
// DSA_SW_DMAX and the persisted keys' addressing live in dsa_scores_heads.cuh.

// 2026-10-21: Keys and candidacy from dsa_kpool_compress's pool_keys / pool_indices /
// pool_valid: dsa_index_scores's expressions.
struct DsaSwScratchKeys {
    const float* __restrict__ pool_keys;
    const int* __restrict__ pool_indices;
    const unsigned char* __restrict__ pool_valid;
    const unsigned char* __restrict__ valid_keys;
    unsigned int KP;
    __device__ __forceinline__ bool cand(unsigned int p, unsigned int S, int qp) const {
        const int end = pool_indices[p * KP + KP - 1];
        const int end_c = end < 0 ? 0 : (end >= (int)S ? (int)S - 1 : end);
        const bool vis = (end_c <= qp) && dsa_row_valid(valid_keys, end_c);
        return (pool_valid[p] != 0) && vis;
    }
    __device__ __forceinline__ const float* key(unsigned int p, unsigned int D) const {
        return pool_keys + (size_t)p * D;
    }
};

template <class Keys>
__device__ __forceinline__ void dsa_index_scores_warp_body(
    const Keys& keys,
    const float* __restrict__ q,
    const float* __restrict__ weights,
    const int* __restrict__ q_pos,
    float* __restrict__ out,
    unsigned char* __restrict__ valid_cand,
    unsigned int Q,
    unsigned int P,
    unsigned int H,
    unsigned int D,
    unsigned int S,
    float scale,
    const int* __restrict__ geom
) {
    __shared__ float sq[DSA_SW_HMAX * DSA_SW_DMAX];
    __shared__ float sw[DSA_SW_HMAX];
    const unsigned int r = blockIdx.y, tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    if (r >= Q || H > DSA_SW_HMAX || D > DSA_SW_DMAX || (D & 31u) != 0u || blockDim.x != 256u)
        return;
    // 2026-10-21: Row r's scores and candidacy start at r * stride: the P argument (a ceiling
    // launch's max_pools), so rows whose live pool counts differ keep fixed strides.
    const unsigned int stride = P;
    unsigned int n = P;
    if (geom) {
        geom += (size_t)r * DSA_GEOM_SLOTS;
        S = (unsigned int)geom[DSA_GEOM_S];
        P = (unsigned int)geom[DSA_GEOM_NPOOLS];
        n = min(P, n);
    }
    for (unsigned int i = tid; i < H * D; i += blockDim.x) sq[i] = q[(size_t)r * H * D + i];
    for (unsigned int i = tid; i < H; i += blockDim.x) sw[i] = weights[(size_t)r * H + i];
    __syncthreads();
    const unsigned int nd = D >> 5;
    const int qp = q_pos[r];
    for (unsigned int p = blockIdx.x * DSA_SW_WARPS + warp; p < n;
         p += gridDim.x * DSA_SW_WARPS) {
        if (!keys.cand(p, S, qp)) {
            if (lane == 0u) {
                valid_cand[(size_t)r * stride + p] = 0;
                out[(size_t)r * stride + p] = -FLT_MAX;
            }
            continue;
        }
        const float* pk = keys.key(p, D);
        float kr[DSA_SW_DMAX / 32u];
#pragma unroll
        for (unsigned int j = 0; j < DSA_SW_DMAX / 32u; ++j)
            kr[j] = j < nd ? pk[lane + 32u * j] : 0.0f;
        float acc = 0.0f;
        for (unsigned int h = 0; h < H; ++h) {
            const float* qh = sq + h * D;
            float dot = 0.0f;
#pragma unroll
            for (unsigned int j = 0; j < DSA_SW_DMAX / 32u; ++j)
                if (j < nd) dot += qh[lane + 32u * j] * kr[j];
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, off);
            acc = __fadd_rn(acc, __fmul_rn(sw[h], fmaxf(scale * dot, 0.0f)));
        }
        if (lane == 0u) {
            valid_cand[(size_t)r * stride + p] = 1;
            out[(size_t)r * stride + p] = acc;
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) dsa_index_scores_warp(
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
    float scale,
    const int* __restrict__ geom
) {
    const DsaSwScratchKeys keys{pool_keys, pool_indices, pool_valid, valid_keys, KP};
    dsa_index_scores_warp_body(keys, q, weights, q_pos, out, valid_cand, Q, P, H, D, S, scale,
                               geom);
}

// 2026-10-21: dsa_index_scores_warp over a paged cache's persisted pool keys (no
// dsa_kpool_compress): `pk` is the V pool's base plus the pool-key region's offset, `bt` the
// sequence's block table. Same launch as dsa_index_scores_warp.
extern "C" __global__ void __launch_bounds__(256) dsa_index_scores_warp_paged(
    const float* __restrict__ q,
    const float* __restrict__ pk,
    const float* __restrict__ weights,
    const int* __restrict__ bt,
    const int* __restrict__ q_pos,
    float* __restrict__ out,
    unsigned char* __restrict__ valid_cand,
    unsigned int Q,
    unsigned int P,
    unsigned int H,
    unsigned int D,
    unsigned int KP,
    unsigned int S,
    unsigned int bs,
    unsigned int pk_blk,
    float scale,
    const int* __restrict__ geom
) {
    const DsaSwPersistedKeys keys{pk, bt, bs, pk_blk, KP};
    dsa_index_scores_warp_body(keys, q, weights, q_pos, out, valid_cand, Q, P, H, D, S, scale,
                               geom);
}

// 2026-10-21: grid (1, rows), block >= 1, after the rows' dsa_indexer_store(_rows) or GEMM
// placement: row r sits at x = pos[r] (or pos0 + r with pos NULL) through block table
// bt + r * bt_stride; when it completes pool p = x / KP ((x + 1) % KP == 0) the pool's key,
// dsa_pool_key over its KP rows (the expressions dsa_kpool_compress computes it with on a paged
// cache: first_key 0, no validity bytes, every slot below S), goes to its persisted slot
// (dsa_pk_ptr). A separate launch, so every row of the pool is in the cache first.
// 2026-10-21: A flat cache (the MTP drafter's) runs the same bt-NULL addressing over its whole
// buffers: x is the row's position, pk the [pools, D] key buffer.
// With bt NULL (a host-placed row) k_pool, gate_pool and pk point at the row's own block and
// x is its slot there: a pool never spans blocks (bs is a multiple of KP), so the in-block
// rows and entry (x / KP) are the same bytes.
extern "C" __global__ void dsa_pool_keys_store_rows(
    const int* __restrict__ pos,
    int pos0,
    const __nv_bfloat16* __restrict__ k_pool,
    const __nv_bfloat16* __restrict__ gate_pool,
    float* __restrict__ pk,
    const float* __restrict__ ape,
    unsigned int D,
    unsigned int KP,
    const int* __restrict__ bt,
    unsigned int bt_stride,
    unsigned int bs,
    unsigned int blk_elems,
    unsigned int pk_blk
) {
    const unsigned int row = blockIdx.y;
    const int x = pos ? pos[row] : pos0 + (int)row;
    if (x < 0 || ((unsigned int)x + 1u) % KP != 0u) return;
    const unsigned int p = (unsigned int)x / KP;
    const int* btr = bt ? bt + (size_t)row * bt_stride : nullptr;
    float* dst = btr ? dsa_pk_ptr(pk, btr, bs, pk_blk, KP, D, p) : pk + (size_t)p * D;
    for (unsigned int d = threadIdx.x; d < D; d += blockDim.x)
        dst[d] = dsa_pool_key(k_pool, gate_pool, nullptr, ape, (unsigned int)x + 1u, D, KP, 0,
                              btr, bs, blk_elems, p, d);
}

// 2026-10-21: dsa_index_scores_warp_paged's bytes with one lane per index head
// (`METRALE_GLM_DSA_SCORES_HEADS`, default on), for the captured decode's ceiling launch over
// the persisted pool keys. Lane h keeps its row's [D] query of head h and its head weight in
// registers for the whole walk (staged through shared memory once per block), so a pool costs
// no shuffle and no shared-memory query read:
// - the warp copies the pool's 512 B key into a 4-deep cp.async ring (one 16 B piece per lane,
//   coalesced; the block-table reads for a window of 32 pools are made a window ahead), and
//   every lane reads it back as broadcast float4s;
// - lane h's head term is dsa_sh_pool_term (dsa_scores_heads.cuh: the warp kernel's partials,
//   tree and rounded product), staged in shared memory, and lane j of the warp adds pool j's
//   terms in head order from 0 with _rn intrinsics (dsa_sh_flush), every 16 pools.
// The rows of a verify are on warps: warp w scores row w % Q, and the Q warps of a group walk
// the same pools, so the second row's key read is an L1/L2 hit. Launch: grid (any x, 1, 1),
// block 256, Q <= 8; H <= 32, D == 128 (else nothing is written; the launcher sends no other
// shape), pk 16-byte aligned and pk_blk a multiple of 4 (the 16 B copies). Rows, strides and
// candidacy are dsa_index_scores_warp_paged's.
extern "C" __global__ void __launch_bounds__(256, 1) dsa_index_scores_heads_paged(
    const float* __restrict__ q,
    const float* __restrict__ pk,
    const float* __restrict__ weights,
    const int* __restrict__ bt,
    const int* __restrict__ q_pos,
    float* __restrict__ out,
    unsigned char* __restrict__ valid_cand,
    unsigned int Q,
    unsigned int P,
    unsigned int H,
    unsigned int D,
    unsigned int KP,
    unsigned int S,
    unsigned int bs,
    unsigned int pk_blk,
    float scale,
    const int* __restrict__ geom
) {
    // 2026-10-21: The walk's key ring, head terms and pool ids; before the walk the same bytes
    // stage each row's [H, D] query once per block (row stride DSA_SH_QLD, so the 8 lanes of a
    // float4 phase read distinct banks).
    __shared__ __align__(16) unsigned char smem[DSA_SH_SMEM];
    auto sring = reinterpret_cast<float4(*)[DSA_SH_RING][32]>(smem);
    auto sterm = reinterpret_cast<float(*)[DSA_SH_BATCH][33]>(smem + DSA_SH_RING_BYTES);
    auto spool = reinterpret_cast<unsigned int(*)[DSA_SH_BATCH]>(
        smem + DSA_SH_RING_BYTES + DSA_SH_TERM_BYTES);
    float* const sq = reinterpret_cast<float*>(smem);
    const unsigned int tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    if (Q == 0u || Q > DSA_SW_WARPS || H > 32u || D != DSA_SH_D || blockDim.x != 256u) return;
    const unsigned int groups = DSA_SW_WARPS / Q;
    const bool active = warp < groups * Q;
    const unsigned int r = warp % Q;
    const bool head = lane < H;
    float qr[DSA_SH_D];
    float wh = 0.0f;
    dsa_sh_stage_q(sq, q, weights, Q, H, r, active && head, qr, wh);
    if (!active) return;
    const DsaSwPersistedKeys keys{pk, bt, bs, pk_blk, KP};
    const unsigned int stride = P;
    unsigned int n = P;
    if (geom) {
        const int* g = geom + (size_t)r * DSA_GEOM_SLOTS;
        S = (unsigned int)g[DSA_GEOM_S];
        n = min((unsigned int)g[DSA_GEOM_NPOOLS], n);
    }
    const int qp = q_pos[r];
    const unsigned int first = blockIdx.x * groups + warp / Q, step = gridDim.x * groups;
    float* const orow = out + (size_t)r * stride;
    unsigned char* const crow = valid_cand + (size_t)r * stride;
    // 2026-10-21: Pool j of this warp's walk is first + j * step; its key goes to ring slot
    // j % RING as one cp.async group (empty past the walk or for a non-candidate). Lane l holds
    // the key address of pool 32 w + l of window w (NULL past the walk or for a non-candidate)
    // for the current and the next window, so the block-table reads are made a window ahead of
    // the copies that need them.
    auto win = [&](unsigned int j0) -> const float* {
        const unsigned int p = first + (j0 + lane) * step;
        return (p < n && keys.cand(p, S, qp)) ? keys.key(p, DSA_SH_D) : nullptr;
    };
    const float* cur = win(0u);
    const float* nxt = win(32u);
    auto issue = [&](unsigned int j) {
        if (j % 32u == 0u && j != 0u) {
            cur = nxt;
            nxt = win(j + 32u);
        }
        const float* from = (const float*)__shfl_sync(
            0xffffffffu, (unsigned long long)cur, (int)(j % 32u));
        if (from)
            e4m3g::cp16(dsa_sh_smem(&sring[warp][j % DSA_SH_RING][lane]), from + 4u * lane, true);
        e4m3g::cp_commit();
    };
#pragma unroll
    for (unsigned int j = 0; j + 1u < DSA_SH_RING; ++j) issue(j);
    unsigned int nb = 0;
    for (unsigned int i = 0;; ++i) {
        const unsigned int p = first + i * step;
        const bool live = p < n;
        const bool c = live && keys.cand(p, S, qp);
        issue(i + DSA_SH_RING - 1u);
        e4m3g::cp_wait<DSA_SH_RING - 1u>();
        __syncwarp();
        if (c && head)
            sterm[warp][nb][lane] = dsa_sh_pool_term(qr, sring[warp][i % DSA_SH_RING], wh, scale);
        if (c && lane == 0u) spool[warp][nb] = p;
        if (live && !c && lane == 0u) {
            crow[p] = 0;
            orow[p] = -FLT_MAX;
        }
        nb += c ? 1u : 0u;
        // 2026-10-21: Every 16 scored pools (and at the end) lane j adds pool j's head terms.
        if (nb == DSA_SH_BATCH || (!live && nb != 0u)) {
            __syncwarp();
            dsa_sh_flush(sterm[warp], spool[warp], nb, H, lane, crow, orow);
            nb = 0u;
        }
        // 2026-10-21: Every lane is past slot i % RING (and the batch) before it is rewritten.
        __syncwarp();
        if (!live) break;
    }
    e4m3g::cp_wait<0>();
}
