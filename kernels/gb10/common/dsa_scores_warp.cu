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

#include "dsa_indexer_body.cuh"

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
// them. Launch: grid (any x, Q, 1), block 256; H <= DSA_SW_HMAX, D <= DSA_SW_DMAX and D a
// multiple of 32 (else nothing is written; the launcher never sends another shape).
// 2026-10-21: The body is generic over where a pool's key and candidacy come from (`Keys`):
// dsa_kpool_compress's scratch (DsaSwScratchKeys) or the persisted pool keys of a paged cache
// (DsaSwPersistedKeys, dsa_index_scores_warp_paged).
#define DSA_SW_WARPS 8u
#define DSA_SW_HMAX 32u
#define DSA_SW_DMAX 128u

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

// 2026-10-21: Persisted pool keys (IndexerRowFormat::Bf16Pooled): pool p's f32 key is entry
// (p * KP % bs) / KP of the [bs / KP, D] region at `pk` in V block bt[p * KP / bs], `pk_blk`
// f32 apart. dsa_pool_keys_store_rows writes it when the pool's last row is stored. On a paged
// cache (first_key 0, no validity bytes) dsa_kpool_compress's pool p is valid iff
// (p + 1) * KP <= S and its last slot is p * KP + KP - 1, so the candidacy above is
// (p + 1) * KP <= S && p * KP + KP - 1 <= q_pos.
__device__ __forceinline__ float* dsa_pk_ptr(float* pk, const int* bt, unsigned int bs,
                                             unsigned int pk_blk, unsigned int KP,
                                             unsigned int D, unsigned int p) {
    const unsigned long long t = (unsigned long long)p * KP;
    return pk + (size_t)bt[t / bs] * pk_blk + (size_t)((t % bs) / KP) * D;
}

struct DsaSwPersistedKeys {
    const float* __restrict__ pk;
    const int* __restrict__ bt;
    unsigned int bs;
    unsigned int pk_blk;
    unsigned int KP;
    __device__ __forceinline__ bool cand(unsigned int p, unsigned int S, int qp) const {
        return (unsigned long long)(p + 1u) * KP <= S && (long long)p * KP + KP - 1 <= qp;
    }
    __device__ __forceinline__ const float* key(unsigned int p, unsigned int D) const {
        return dsa_pk_ptr(const_cast<float*>(pk), bt, bs, pk_blk, KP, D, p);
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
    unsigned int n = P;
    if (geom) {
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
                valid_cand[(size_t)r * P + p] = 0;
                out[(size_t)r * P + p] = -FLT_MAX;
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
            valid_cand[(size_t)r * P + p] = 1;
            out[(size_t)r * P + p] = acc;
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
