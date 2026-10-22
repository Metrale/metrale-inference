// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-22: The pieces the persisted-pool-key scores kernels share, moved here from
// dsa_scores_warp.cu so dsa_scores_heads_window.cu runs the same expressions:
// - the warp kernels' shape bounds and the persisted keys' addressing and candidacy;
// - the one-lane-per-head kernels' shared-memory layout, query staging, per-pool head term and
//   head-order sum.
//
// Owner: gb10 kernels.
// Invariants: every helper is the code dsa_index_scores_heads_paged ran inline before the move,
// in the same order, so a kernel built from them writes dsa_index_scores_warp_paged's bytes
// (gate examples/glm5next_dsa_ceiling_scores_gate.rs).

#pragma once

#include "dsa_indexer_body.cuh"
#include "e4m3_mma_pipe.cuh"

// 2026-10-21: dsa_index_scores_warp's launch: warps per block and the largest head count and
// head dimension it stages.
#define DSA_SW_WARPS 8u
#define DSA_SW_HMAX 32u
#define DSA_SW_DMAX 128u

// 2026-10-21: Persisted pool keys (IndexerRowFormat::Bf16Pooled): pool p's f32 key is entry
// (p * KP % bs) / KP of the [bs / KP, D] region at `pk` in V block bt[p * KP / bs], `pk_blk`
// f32 apart. dsa_pool_keys_store_rows writes it when the pool's last row is stored. On a paged
// cache (first_key 0, no validity bytes) dsa_kpool_compress's pool p is valid iff
// (p + 1) * KP <= S and its last slot is p * KP + KP - 1, so the candidacy above is
// (p + 1) * KP <= S && p * KP + KP - 1 <= q_pos.
// 2026-10-21: With bt NULL (a flat cache's persisted keys, the MTP drafter's) pool p's key is
// row p of a [pools, D] buffer at `pk`, where dsa_pool_keys_store_rows writes it with bt NULL;
// the candidacy is the same, every row below S of that cache being valid.
// 2026-10-22: Split in two (the block-table read, then the address in that block) so a walk can
// read a window's block ids ahead of the copies that need them (dsa_scores_heads_window.cu).
__device__ __forceinline__ int dsa_pk_block(const int* bt, unsigned int bs, unsigned int KP,
                                            unsigned int p) {
    return bt[(unsigned long long)p * KP / bs];
}
__device__ __forceinline__ float* dsa_pk_in_block(float* pk, int blk, unsigned int bs,
                                                  unsigned int pk_blk, unsigned int KP,
                                                  unsigned int D, unsigned int p) {
    const unsigned long long t = (unsigned long long)p * KP;
    return pk + (size_t)blk * pk_blk + (size_t)((t % bs) / KP) * D;
}
__device__ __forceinline__ float* dsa_pk_ptr(float* pk, const int* bt, unsigned int bs,
                                             unsigned int pk_blk, unsigned int KP,
                                             unsigned int D, unsigned int p) {
    return dsa_pk_in_block(pk, dsa_pk_block(bt, bs, KP, p), bs, pk_blk, KP, D, p);
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
        return bt ? dsa_pk_ptr(const_cast<float*>(pk), bt, bs, pk_blk, KP, D, p)
                  : pk + (size_t)p * D;
    }
};

// 2026-10-21: The one-lane-per-head kernels' shapes and shared memory: a 128-wide head, a ring
// of RING 512 B keys per warp, the head terms of BATCH pools per warp (33 apart, so lane j's
// column reads are conflict-free) and their pool ids; the query staging (row stride QLD, so the
// 8 lanes of a float4 phase read distinct banks) reuses the same bytes before the walk.
#define DSA_SH_D 128u
#define DSA_SH_RING 4u
#define DSA_SH_BATCH 16u
#define DSA_SH_QLD (DSA_SH_D + 4u)
#define DSA_SH_RING_BYTES (DSA_SW_WARPS * DSA_SH_RING * 32u * 16u)
#define DSA_SH_TERM_BYTES (DSA_SW_WARPS * DSA_SH_BATCH * 33u * 4u)
#define DSA_SH_WALK_BYTES (DSA_SH_RING_BYTES + DSA_SH_TERM_BYTES + DSA_SW_WARPS * DSA_SH_BATCH * 4u)
#define DSA_SH_Q_BYTES (DSA_SW_HMAX * DSA_SH_QLD * 4u)
#define DSA_SH_SMEM (DSA_SH_WALK_BYTES > DSA_SH_Q_BYTES ? DSA_SH_WALK_BYTES : DSA_SH_Q_BYTES)

__device__ __forceinline__ unsigned int dsa_sh_smem(const void* p) {
    return (unsigned int)__cvta_generic_to_shared(p);
}

// 2026-10-22: Each row's [H, D] query staged through `sq` once per block (every thread of the
// block calls it); lane h of a warp scoring row r (`mine`: an active warp's head lane) keeps
// row r's head h query in `qr` and its head weight in `wh`.
__device__ __forceinline__ void dsa_sh_stage_q(float* sq, const float* __restrict__ q,
                                               const float* __restrict__ weights,
                                               unsigned int Q, unsigned int H, unsigned int r,
                                               bool mine, float (&qr)[DSA_SH_D], float& wh) {
    const unsigned int tid = threadIdx.x, lane = tid & 31u;
    for (unsigned int rr = 0; rr < Q; ++rr) {
        for (unsigned int e = tid; e < H * (DSA_SH_D / 4u); e += blockDim.x) {
            const unsigned int h = e / (DSA_SH_D / 4u), c4 = e % (DSA_SH_D / 4u);
            *(float4*)(sq + h * DSA_SH_QLD + 4u * c4) =
                ((const float4*)(q + ((size_t)rr * H + h) * DSA_SH_D))[c4];
        }
        __syncthreads();
        if (mine && r == rr) {
            const float4* qv = (const float4*)(sq + lane * DSA_SH_QLD);
#pragma unroll
            for (unsigned int i = 0; i < DSA_SH_D / 4u; ++i) {
                const float4 v = qv[i];
                qr[4u * i] = v.x;
                qr[4u * i + 1u] = v.y;
                qr[4u * i + 2u] = v.z;
                qr[4u * i + 3u] = v.w;
            }
            wh = weights[(size_t)r * H + lane];
        }
        __syncthreads();
    }
}

// 2026-10-21: Lane h's head term of the key at `K` (32 float4s in shared memory, read back as
// broadcasts): the 32 lane partials p_l over d = l, l + 32, l + 64, l + 96 in that order (the
// warp kernel's per-lane `dot += q * k`), then the shfl_down tree's sums in its pairing:
// p_i + p_(i+16), then + 8, + 4, + 2, + 1 (IEEE addition commutes). It starts each p_l at its
// first product instead of adding it to 0.0f: those values equal the warp kernel's except that
// a -0.0 there is +0.0 (by induction over the adds: x + y is exact-equal under that map for
// every x, y), so one `+ 0.0f` (_rn) on the tree's result gives lane 0's `dot` exactly; the term
// is the warp kernel's rounded product.
__device__ __forceinline__ float dsa_sh_pool_term(const float (&qr)[DSA_SH_D],
                                                  const float4* K, float wh, float scale) {
    float a[16];
#pragma unroll
    for (unsigned int half = 0; half < 2u; ++half) {
#pragma unroll
        for (unsigned int i4 = 0; i4 < 4u; ++i4) {
            const unsigned int m = half * 4u + i4;
            const float4 k0 = K[m], k1 = K[8u + m], k2 = K[16u + m], k3 = K[24u + m];
            const float k0v[4] = {k0.x, k0.y, k0.z, k0.w};
            const float k1v[4] = {k1.x, k1.y, k1.z, k1.w};
            const float k2v[4] = {k2.x, k2.y, k2.z, k2.w};
            const float k3v[4] = {k3.x, k3.y, k3.z, k3.w};
#pragma unroll
            for (unsigned int e = 0; e < 4u; ++e) {
                const unsigned int l = 4u * m + e;
                float dot = qr[l] * k0v[e];
                dot += qr[32u + l] * k1v[e];
                dot += qr[64u + l] * k2v[e];
                dot += qr[96u + l] * k3v[e];
                if (half == 0u) a[l] = dot;
                else a[l - 16u] = a[l - 16u] + dot;
            }
        }
    }
#pragma unroll
    for (unsigned int off = 8u; off > 0u; off >>= 1)
#pragma unroll
        for (unsigned int j = 0; j < off; ++j) a[j] = a[j] + a[j + off];
    return __fmul_rn(wh, fmaxf(scale * __fadd_rn(a[0], 0.0f), 0.0f));
}

// 2026-10-21: Lane j < nb adds pool pools[j]'s head terms in head order from 0 with _rn
// intrinsics (as dsa_index_scores_warp's lane 0 does) and writes its score and candidacy.
__device__ __forceinline__ void dsa_sh_flush(const float (*terms)[33],
                                             const unsigned int* pools, unsigned int nb,
                                             unsigned int H, unsigned int lane,
                                             unsigned char* crow, float* orow) {
    if (lane < nb) {
        float acc = 0.0f;
        for (unsigned int h = 0; h < H; ++h) acc = __fadd_rn(acc, terms[lane][h]);
        const unsigned int pj = pools[lane];
        crow[pj] = 1;
        orow[pj] = acc;
    }
}
