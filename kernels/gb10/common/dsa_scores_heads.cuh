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

// 2026-10-23: The persisted keys' element type (METRALE_GLM_DSA_POOL_KEYS_BF16): FP32, or the
// FP32 key rounded to BF16 (RN) in the first 256 B of the same 512 B slot, so the addressing and
// pk_blk are unchanged. A BF16 key widens to FP32 exactly, so a kernel over BF16 keys writes the
// bytes its FP32 instance writes for the keys rounded to BF16. A key's 16 B pieces are copied by
// the warp's first dsa_sh_key_lanes lanes, each at its FP32 piece's byte offset.
template <bool BF16>
__device__ __forceinline__ unsigned int dsa_sh_key_lanes() {
    return BF16 ? 16u : 32u;
}
// 2026-10-23: Elements 4 m .. 4 m + 3 of the key at K (a ring slot).
template <bool BF16>
__device__ __forceinline__ float4 dsa_sh_key4(const float4* K, unsigned int m) {
    if constexpr (BF16) {
        const uint2 u = reinterpret_cast<const uint2*>(K)[m];
        return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xFFFF0000u),
                           __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xFFFF0000u));
    } else {
        return K[m];
    }
}

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
template <bool BF16>
__device__ __forceinline__ float dsa_sh_pool_term(const float (&qr)[DSA_SH_D],
                                                  const float4* K, float wh, float scale) {
    float a[16];
#pragma unroll
    for (unsigned int half = 0; half < 2u; ++half) {
#pragma unroll
        for (unsigned int i4 = 0; i4 < 4u; ++i4) {
            const unsigned int m = half * 4u + i4;
            const float4 k0 = dsa_sh_key4<BF16>(K, m), k1 = dsa_sh_key4<BF16>(K, 8u + m),
                         k2 = dsa_sh_key4<BF16>(K, 16u + m), k3 = dsa_sh_key4<BF16>(K, 24u + m);
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

// 2026-10-23: A walk's key ring per warp: DSA_SH_RING slots of a 512 B FP32 key, or twice as
// many 256 B BF16 keys in the same bytes (DSA_SH_RING_BYTES), so the BF16 walk keeps twice as
// many copies in flight. Slots are dsa_sh_slot4 float4s.
template <bool BF16>
__device__ constexpr unsigned int dsa_sh_ring() {
    return BF16 ? 2u * DSA_SH_RING : DSA_SH_RING;
}
template <bool BF16>
__device__ constexpr unsigned int dsa_sh_slot4() {
    return BF16 ? 16u : 32u;
}

// 2026-10-23: dsa_pool_keys_store_rows (dsa_scores_warp.cu holds its contract), writing the
// FP32 key or its BF16 rounding (dsa_sh_key4's layout).
template <bool BF16>
__device__ __forceinline__ void dsa_pool_keys_store_body(
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
) {    const unsigned int row = blockIdx.y;
    const int x = pos ? pos[row] : pos0 + (int)row;
    if (x < 0 || ((unsigned int)x + 1u) % KP != 0u) return;
    const unsigned int p = (unsigned int)x / KP;
    const int* btr = bt ? bt + (size_t)row * bt_stride : nullptr;
    float* dst = btr ? dsa_pk_ptr(pk, btr, bs, pk_blk, KP, D, p) : pk + (size_t)p * D;
    for (unsigned int d = threadIdx.x; d < D; d += blockDim.x) {
        const float v = dsa_pool_key(k_pool, gate_pool, nullptr, ape, (unsigned int)x + 1u, D,
                                     KP, 0, btr, bs, blk_elems, p, d);
        if constexpr (BF16) {
            reinterpret_cast<__nv_bfloat16*>(dst)[d] = __float2bfloat16_rn(v);
        } else {
            dst[d] = v;
        }
    }
}

// 2026-10-23: dsa_index_scores_heads_paged's walk (dsa_scores_warp.cu holds its method and
// launch contract), over FP32 or BF16 persisted keys (dsa_sh_key_lanes, dsa_sh_key4).
template <bool BF16>
__device__ __forceinline__ void dsa_sh_paged_walk(
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
) {    // 2026-10-21: The walk's key ring, head terms and pool ids; before the walk the same bytes
    // stage each row's [H, D] query once per block (row stride DSA_SH_QLD, so the 8 lanes of a
    // float4 phase read distinct banks).
    __shared__ __align__(16) unsigned char smem[DSA_SH_SMEM];
    constexpr unsigned int R = dsa_sh_ring<BF16>();
    auto sring = reinterpret_cast<float4(*)[R][dsa_sh_slot4<BF16>()]>(smem);
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
        if (from && lane < dsa_sh_key_lanes<BF16>())
            e4m3g::cp16(dsa_sh_smem(&sring[warp][j % R][lane]), from + 4u * lane, true);
        e4m3g::cp_commit();
    };
#pragma unroll
    for (unsigned int j = 0; j + 1u < R; ++j) issue(j);
    unsigned int nb = 0;
    for (unsigned int i = 0;; ++i) {
        const unsigned int p = first + i * step;
        const bool live = p < n;
        const bool c = live && keys.cand(p, S, qp);
        issue(i + R - 1u);
        e4m3g::cp_wait<R - 1u>();
        __syncwarp();
        if (c && head)
            sterm[warp][nb][lane] = dsa_sh_pool_term<BF16>(qr, sring[warp][i % R], wh, scale);
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

// 2026-10-23: dsa_index_scores_heads_window's walk (dsa_scores_heads_window.cu holds its method
// and launch contract), over FP32 or BF16 persisted keys (dsa_sh_key_lanes, dsa_sh_key4).
template <bool BF16>
__device__ __forceinline__ void dsa_sh_window_walk(
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
) {    __shared__ __align__(16) unsigned char smem[DSA_SH_SMEM];
    constexpr unsigned int R = dsa_sh_ring<BF16>();
    auto sring = reinterpret_cast<float4(*)[R][dsa_sh_slot4<BF16>()]>(smem);
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
    // 2026-10-22: Zero for a lane past H, which runs the math it never stores.
    float qr[DSA_SH_D];
#pragma unroll
    for (unsigned int i = 0; i < DSA_SH_D; ++i) qr[i] = 0.0f;
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
    // 2026-10-22: Lane l's pool of window w, its candidacy, its block id (read only for a
    // candidate of a paged cache) and its key address (`pk` itself for a non-candidate: the
    // copy from it reads no byte).
    auto wpool = [&](unsigned int w) { return first + (32u * w + lane) * step; };
    auto wcand = [&](unsigned int w) {
        const unsigned int p = wpool(w);
        return p < n && keys.cand(p, S, qp);
    };
    auto wblock = [&](unsigned int w, bool c) {
        return (c && bt) ? dsa_pk_block(bt, bs, KP, wpool(w)) : 0;
    };
    auto waddr = [&](unsigned int w, bool c, int blk) -> const float* {
        if (!c) return pk;
        return bt ? dsa_pk_in_block(const_cast<float*>(pk), blk, bs, pk_blk, KP, DSA_SH_D,
                                    wpool(w))
                  : pk + (size_t)wpool(w) * DSA_SH_D;
    };
    const bool c0 = wcand(0u), c1 = wcand(1u);
    const float* cur = waddr(0u, c0, wblock(0u, c0));
    const float* nxt = waddr(1u, c1, wblock(1u, c1));
    unsigned int cm = __ballot_sync(0xffffffffu, c0), cmn = __ballot_sync(0xffffffffu, c1);
    bool c2 = wcand(2u);
    int blk2 = wblock(2u, c2);
    // 2026-10-22: Pool j of the walk (first + j * step) goes to ring slot j % RING as one
    // cp.async group; the prologue copies pools 0 .. RING - 1, all in window 0.
    // 2026-10-23: Only a key's lanes copy (a BF16 slot is 256 B: a zero-fill from lane 16 up
    // would land in the next slot).
    auto copy = [&](unsigned int j, const float* from, bool on) {
        if (lane < dsa_sh_key_lanes<BF16>())
            e4m3g::cp16(dsa_sh_smem(&sring[warp][j % R][lane]), from + 4u * lane, on);
        e4m3g::cp_commit();
    };
#pragma unroll
    for (unsigned int t = 0; t < R; ++t)
        copy(t, (const float*)__shfl_sync(0xffffffffu, (unsigned long long)cur, (int)t),
             (cm >> t) & 1u);
    unsigned int nb = 0, j = 0;
    for (unsigned int w = 0;; ++w) {
        const unsigned int live = __popc(__ballot_sync(0xffffffffu, wpool(w) < n));
        for (unsigned int t = 0; t < live; ++t, ++j) {
            const unsigned int p = first + (32u * w + t) * step;
            const bool c = (cm >> t) & 1u;
            const unsigned int ta = t + R;
            const bool in_cur = ta < 32u;
            const float* ahead = (const float*)__shfl_sync(
                0xffffffffu, (unsigned long long)(in_cur ? cur : nxt), (int)(ta & 31u));
            const bool ahead_on = ((in_cur ? cm : cmn) >> (ta & 31u)) & 1u;
            e4m3g::cp_wait<R - 1u>();
            __syncwarp();
            const float term = dsa_sh_pool_term<BF16>(qr, sring[warp][j % R], wh, scale);
            if (c && head) sterm[warp][nb][lane] = term;
            if (c && lane == 0u) spool[warp][nb] = p;
            if (!c && lane == 0u) {
                crow[p] = 0;
                orow[p] = -FLT_MAX;
            }
            nb += c ? 1u : 0u;
            if (nb == DSA_SH_BATCH) {
                __syncwarp();
                dsa_sh_flush(sterm[warp], spool[warp], nb, H, lane, crow, orow);
                nb = 0u;
            }
            // 2026-10-22: Every lane is past slot j % RING (and the batch) before it is
            // rewritten with pool j + RING.
            __syncwarp();
            copy(j + R, ahead, ahead_on);
        }
        if (live < 32u) break;
        cur = nxt;
        cm = cmn;
        nxt = waddr(w + 2u, c2, blk2);
        cmn = __ballot_sync(0xffffffffu, c2);
        c2 = wcand(w + 3u);
        blk2 = wblock(w + 3u, c2);
    }
    __syncwarp();
    dsa_sh_flush(sterm[warp], spool[warp], nb, H, lane, crow, orow);
    e4m3g::cp_wait<0>();
}
