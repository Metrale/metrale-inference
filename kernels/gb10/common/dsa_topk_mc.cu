// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-21: dsa_topk_pools_radix's selection over C CTAs per row (`METRALE_GLM_DSA_TOPK_MC`,
// default on) for the captured decode's ceiling launch: the one-CTA radix walks a 32768-pool
// row five times from one SM (~96 us at a 128k context); here each CTA of a row owns a
// contiguous chunk of its pools.
//
// Launch sequence, all grid (C, Q), block 256, 1 <= C <= DSA_MC_CMAX fixed at capture (the row's live pools P
// and select_k come from the device geometry, as dsa_topk_pools_radix reads them):
//   dsa_topk_mc_pass(pass 0), (pass 1), (pass 2), (pass 3), dsa_topk_mc_collect.
// Pass j histograms digit j (bits 31 - 8 j .. 24 - 8 j of dsa_tk) of the keys that match the
// digits chosen so far; each CTA re-derives those digits from the earlier passes' histograms
// with dsa_tr_pick (the same pick as dsa_topk_pools_radix, so the same K and `need`), one
// pick per kernel from the state the previous pass stored. A pass
// stops early once the chosen digit's bin holds exactly the keys still needed. The collect
// writes every pool keyed above the prefix (any order) and each CTA's first `need` pools at it
// (pool order); the last CTA of the row (atomic arrival count) gathers the gt pools, then the
// prefix pools in CTA order up to `need`, and ranks them with dsa_tr_rank_emit. The selected
// set is the top select_k of the total order, so the slots are dsa_topk_pools_radix's for
// NaN-free scores.
//
// Workspace (u32 words, `ws` + r * DSA_MC_WS_WORDS(C)): h0 [C][256] (pass 0's per-CTA counts,
// fully written, no zeroing needed), h [3][256] (passes 1..3, atomic), ctr [2] (gt slots,
// arrivals), gtv [512] f32, gti [512], eqc [C], eqi [C][512], st [4][4] (dsa_mc_state; written
// before it is read in every sequence). Pass 0 zeroes h and ctr before
// any later kernel of the sequence reads or adds to them, so nothing carries over between
// steps (replay-safe). Rows read scores at stride P and write selected at stride select_k, the
// arguments as passed.
//
// Owner: gb10 kernels.

#include "dsa_indexer_body.cuh"
#include "dsa_topk_stream.cuh"
#include "dsa_topk_radix.cuh"

#define DSA_MC_CMAX 64u
#define DSA_MC_WS_WORDS(C) ((C) * 256u + 768u + 2u + 2u * DSA_TR_MAX + (C) + (C) * DSA_TR_MAX + 16u)

struct DsaMcRow {
    const float* row;
    unsigned int* h0;
    unsigned int* h;
    unsigned int* ctr;
    float* gtv;
    int* gti;
    unsigned int* eqc;
    int* eqi;
    unsigned int* st;
    unsigned int P, select_k, lo, hi;
};

// 2026-10-21: Row blockIdx.y's views, live extents and this CTA's pools [lo, hi): thread t
// owns E consecutive pools from lo + t E. False (nothing to do) on the launch refusals
// dsa_topk_pools_radix makes.
__device__ __forceinline__ bool dsa_mc_row(const float* scores, unsigned int* ws, unsigned int Q,
                                           unsigned int P, unsigned int select_k,
                                           const int* geom, DsaMcRow& m) {
    const unsigned int r = blockIdx.y, C = gridDim.x;
    const unsigned int sc_stride = P;
    if (geom) {
        geom += (size_t)r * DSA_GEOM_SLOTS;
        P = (unsigned int)geom[DSA_GEOM_NPOOLS];
        select_k = (unsigned int)geom[DSA_GEOM_SELECT_K];
    }
    if (r >= Q || select_k == 0u || select_k > DSA_TR_MAX || blockDim.x != 256u ||
        C > DSA_MC_CMAX)
        return false;
    unsigned int* w = ws + (size_t)r * DSA_MC_WS_WORDS(C);
    m.row = scores + (size_t)r * sc_stride;
    m.h0 = w;
    m.h = w + C * 256u;
    m.ctr = m.h + 768u;
    m.gtv = (float*)(m.ctr + 2u);
    m.gti = (int*)(m.ctr + 2u + DSA_TR_MAX);
    m.eqc = m.ctr + 2u + 2u * DSA_TR_MAX;
    m.eqi = (int*)(m.eqc + C);
    m.st = (unsigned int*)(m.eqi + C * DSA_TR_MAX);
    m.P = P;
    m.select_k = select_k;
    const unsigned int chunk = (P + C - 1u) / C;
    m.lo = min(P, blockIdx.x * chunk);
    m.hi = min(P, m.lo + chunk);
    return true;
}

// 2026-10-21: S_j, the digits passes 0 .. j - 1 chose: prefix / mask over their bits, k the
// winners still needed among keys matching it, done once a chosen bin held exactly k (later
// digits are then not chosen). S_0 is (0, 0, select_k, no); S_j for j >= 1 is S_(j-1) (as
// pass j - 1 stored it, st[j - 1]) advanced by one dsa_tr_pick over pass j - 1's histogram.
// CTA 0 of a row stores S_j at st[j] for j <= 3 (passes j + 1 read it). All threads; uses
// s_cnt[256] and three shared words.
__device__ __forceinline__ void dsa_mc_state(const DsaMcRow& m, unsigned int j,
                                             unsigned int* s_cnt, unsigned int* s_w,
                                             unsigned int& prefix, unsigned int& mask,
                                             unsigned int& k, bool& done) {
    const unsigned int tid = threadIdx.x, C = gridDim.x;
    prefix = 0u;
    mask = 0u;
    k = m.select_k;
    done = false;
    if (j == 0u) return;
    if (j >= 2u) {
        const unsigned int* st = m.st + 4u * (j - 1u);
        prefix = __ldcg(st);
        mask = __ldcg(st + 1);
        k = __ldcg(st + 2);
        done = __ldcg(st + 3) != 0u;
    }
    if (!done) {
        unsigned int v = 0u;
        if (j == 1u) {
            // 2026-10-21: Independent loads, 16 in flight per thread.
            unsigned int part[16] = {};
            for (unsigned int c0 = 0; c0 < C; c0 += 16u) {
#pragma unroll
                for (unsigned int u = 0; u < 16u; ++u)
                    if (c0 + u < C) part[u] += __ldcg(m.h0 + (c0 + u) * 256u + tid);
            }
#pragma unroll
            for (unsigned int u = 0; u < 16u; ++u) v += part[u];
        } else {
            v = __ldcg(m.h + (j - 2u) * 256u + tid);
        }
        s_cnt[tid] = v;
        __syncthreads();
        if (tid < 32u)
            dsa_tr_pick([&](unsigned int d) { return s_cnt[d]; }, k, tid, &s_w[0], &s_w[1],
                        &s_w[2]);
        __syncthreads();
        const unsigned int shift = 32u - 8u * j;
        prefix |= s_w[0] << shift;
        mask |= 255u << shift;
        k = s_w[1];
        done = s_w[2] == k;
    }
    if (j <= 3u && blockIdx.x == 0u && tid == 0u) {
        unsigned int* st = m.st + 4u * j;
        st[0] = prefix;
        st[1] = mask;
        st[2] = k;
        st[3] = done ? 1u : 0u;
    }
}

extern "C" __global__ void __launch_bounds__(256) dsa_topk_mc_pass(
    const float* __restrict__ scores,
    unsigned int* __restrict__ ws,
    unsigned int Q,
    unsigned int P,
    unsigned int select_k,
    const int* __restrict__ geom,
    unsigned int pass
) {
    __shared__ unsigned int hist[DSA_TR_WARPS][256];
    __shared__ unsigned int s_cnt[256], s_w[3];
    DsaMcRow m;
    if (pass > 3u || !dsa_mc_row(scores, ws, Q, P, select_k, geom, m)) return;
    const unsigned int tid = threadIdx.x, warp = tid >> 5;
    if (pass == 0u && blockIdx.x == 0u)
        for (unsigned int i = tid; i < 770u; i += blockDim.x) m.h[i] = 0u;
    if (m.P <= m.select_k) return;
    unsigned int prefix, mask, k;
    bool done;
    dsa_mc_state(m, pass, s_cnt, s_w, prefix, mask, k, done);
    if (done) return;
    for (unsigned int i = tid; i < DSA_TR_WARPS * 256u; i += blockDim.x) (&hist[0][0])[i] = 0u;
    __syncthreads();
    const unsigned int shift = 24u - 8u * pass;
    const unsigned int E = (m.hi - m.lo + blockDim.x - 1u) / blockDim.x;
    const unsigned int b = m.lo + tid * E, e = min(m.hi, b + E);
#pragma unroll 4
    for (unsigned int i = b; i < e; ++i) {
        const unsigned int key = dsa_tk(m.row[i]);
        if ((key & mask) == prefix) atomicAdd(&hist[warp][(key >> shift) & 255u], 1u);
    }
    __syncthreads();
    unsigned int v = 0u;
#pragma unroll
    for (unsigned int w = 0; w < DSA_TR_WARPS; ++w) v += hist[w][tid];
    if (pass == 0u) m.h0[blockIdx.x * 256u + tid] = v;
    else if (v != 0u) atomicAdd(m.h + (pass - 1u) * 256u + tid, v);
}

extern "C" __global__ void __launch_bounds__(256) dsa_topk_mc_collect(
    const float* __restrict__ scores,
    unsigned int* __restrict__ ws,
    int* __restrict__ selected,
    unsigned int Q,
    unsigned int P,
    unsigned int select_k,
    const int* __restrict__ geom
) {
    __shared__ float cv[DSA_TR_MAX];
    __shared__ int ci[DSA_TR_MAX];
    __shared__ unsigned long long skey[DSA_TR_MAX];
    __shared__ unsigned int s_cnt[256], s_w[3], s_wsum[DSA_TR_WARPS], s_wgt[DSA_TR_WARPS], s_last, s_off[DSA_MC_CMAX];
    const unsigned int sel_stride = select_k;
    DsaMcRow m;
    if (!dsa_mc_row(scores, ws, Q, P, select_k, geom, m)) return;
    const unsigned int tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, C = gridDim.x;
    int* const out = selected + (size_t)blockIdx.y * sel_stride;
    if (m.P <= m.select_k) {
        if (blockIdx.x != 0u) return;
        for (unsigned int i = tid; i < m.P; i += blockDim.x) {
            cv[i] = m.row[i];
            ci[i] = (int)i;
        }
        __syncthreads();
        dsa_tr_rank_emit(cv, ci, skey, m.P, m.select_k, out);
        return;
    }
    unsigned int prefix, mask, need;
    bool done;
    dsa_mc_state(m, 4u, s_cnt, s_w, prefix, mask, need, done);
    const unsigned int n_gt = m.select_k - need;
    // 2026-10-21: Pools above the prefix to the gt slots (a range per CTA; any order, the sort
    // orders them); this CTA's pools at it, in pool order, to its first `need` eq slots.
    const unsigned int E = (m.hi - m.lo + blockDim.x - 1u) / blockDim.x;
    const unsigned int b = m.lo + tid * E, e = min(m.hi, b + E);
    unsigned int ngt = 0u, neq = 0u;
    for (unsigned int i = b; i < e; ++i) {
        const unsigned int key = dsa_tk(m.row[i]) & mask;
        ngt += key > prefix ? 1u : 0u;
        neq += key == prefix ? 1u : 0u;
    }
    unsigned int igt = ngt, ieq = neq;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const unsigned int a = __shfl_up_sync(0xFFFFFFFFu, igt, o);
        const unsigned int c = __shfl_up_sync(0xFFFFFFFFu, ieq, o);
        if (lane >= (unsigned int)o) {
            igt += a;
            ieq += c;
        }
    }
    if (lane == 31u) {
        s_wsum[warp] = ieq;
        s_wgt[warp] = igt;
    }
    __syncthreads();
    unsigned int pe = ieq - neq, pg = igt - ngt;
    for (unsigned int w = 0; w < warp; ++w) {
        pe += s_wsum[w];
        pg += s_wgt[w];
    }
    if (tid == 0u) {
        unsigned int te = 0u, tg = 0u;
        for (unsigned int w = 0; w < DSA_TR_WARPS; ++w) {
            te += s_wsum[w];
            tg += s_wgt[w];
        }
        m.eqc[blockIdx.x] = te;
        s_w[0] = tg != 0u ? atomicAdd(m.ctr, tg) : 0u;
    }
    __syncthreads();
    pg += s_w[0];
    if (ngt != 0u || (neq != 0u && pe < need))
        for (unsigned int i = b; i < e; ++i) {
            const float v = m.row[i];
            const unsigned int key = dsa_tk(v) & mask;
            if (key > prefix) {
                m.gtv[pg] = v;
                m.gti[pg++] = (int)i;
            } else if (key == prefix && pe < need) {
                m.eqi[blockIdx.x * DSA_TR_MAX + pe++] = (int)i;
            }
        }
    // 2026-10-21: The last CTA of the row to arrive sorts the row's winners.
    __threadfence();
    __syncthreads();
    if (tid == 0u) s_last = atomicAdd(m.ctr + 1u, 1u) == C - 1u;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    // 2026-10-21: CTA c's eq pools go after those of CTAs 0 .. c - 1, up to `need` in all.
    if (tid < C) s_cnt[tid] = __ldcg(m.eqc + tid);
    for (unsigned int i = tid; i < n_gt; i += blockDim.x) {
        cv[i] = __ldcg(m.gtv + i);
        ci[i] = __ldcg(m.gti + i);
    }
    __syncthreads();
    if (tid == 0u) {
        unsigned int off = 0u;
        for (unsigned int c = 0; c < C; ++c) {
            s_off[c] = off;
            const unsigned int t = min(s_cnt[c], need - off);
            s_cnt[c] = t;
            off += t;
        }
    }
    __syncthreads();
    for (unsigned int c = warp; c < C; c += DSA_TR_WARPS)
        for (unsigned int i = lane; i < s_cnt[c]; i += 32u) {
            const int p = __ldcg(m.eqi + c * DSA_TR_MAX + i);
            cv[n_gt + s_off[c] + i] = m.row[p];
            ci[n_gt + s_off[c] + i] = p;
        }
    __syncthreads();
    dsa_tr_rank_emit(cv, ci, skey, m.select_k, m.select_k, out);
}
