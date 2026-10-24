// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-23: The captured verify / decode selection's scores over the persisted pool keys stored
// as BF16 (METRALE_GLM_DSA_POOL_KEYS_BF16) on BF16 tensor cores
// (METRALE_GLM_DSA_DECODE_SCORES_MMA): dsa_index_scores_mma's arithmetic (dsa_scores_mma.cuh)
// with dsa_index_scores_heads_window's candidacy, outputs and launch.
//
// Why (GB10, gate examples/glm5next_dsa_ceiling_scores_gate.rs, 2 rows at 32k pools): the
// window walk takes 82-86 us on FP32 keys and 92 us on BF16 keys: one lane per head runs 128
// FMAs per pool and row, so a 2-row pass is bound by the head math, not the key bytes. Here
// the MMAs take the head math and every CTA streams 64-pool BF16 tiles (16 KB) straight from
// the persisted slots, double-buffered.
//
// Method. Grid any x, block 256: CTA b walks tiles b, b + gridDim.x, ... of 64 pools. Tile t's
// keys go to stage k % 2 (k the CTA's walk index) as one cp.async group, issued a tile ahead;
// a pool no row may score is zero-filled, its slot never read. The 8 warps split the rows:
// 8 / Q warps per row (Q <= 8), each taking ceil(8 / (8 / Q)) of the tile's 8 column tiles,
// its row's A fragments (dsa_sm_load_a) in registers for the walk.
//
// Owner: gb10 kernels.
// Invariants:
// - A score is a function of its row's query and weights and its pool's BF16 key only: the
//   bits dsa_index_scores_mma (and dsa_index_scores_mma_pipe) writes for that key (the same A
//   fragments, the same per-accumulator MMA order, dsa_sm_fold_col), since the persisted BF16
//   key is dsa_sm_pack's RN rounding of the FP32 key.
// - Candidacy (row r, pool p): p < n_r && DsaSwPersistedKeys::cand(p, S_r, q_pos[r]) with S_r,
//   n_r from the geometry when given, as dsa_index_scores_heads_window; a non-candidate below
//   n_r gets -FLT_MAX / 0; pools at or past n_r are not written.
// - Launch: dsa_index_scores_heads_window's (any grid x, block 256, no dynamic shared memory);
//   Q <= 8, H == 32, D == 128, pk 16-byte aligned (else nothing is written).

#pragma once

#include <float.h>

#include "dsa_scores_heads.cuh"
#include "dsa_scores_mma.cuh"

#define DSA_SMP_STAGES 2u
#define DSA_SMP_TILE_BYTES (DSA_SM_PT * DSA_SM_LD * 2u)
#define DSA_SMP_CHUNKS (DSA_SM_D / 8u)

__device__ __forceinline__ void dsa_smp_walk(
    const float* __restrict__ q, const DsaSwPersistedKeys keys, const float* __restrict__ weights,
    const int* __restrict__ q_pos, float* __restrict__ out, unsigned char* __restrict__ valid_cand,
    unsigned int Q, unsigned int P, unsigned int S, float scale, const int* __restrict__ geom) {
    __shared__ __align__(16) unsigned char dsa_smp_smem[DSA_SMP_STAGES * DSA_SMP_TILE_BYTES];
    __shared__ unsigned int s_n[DSA_SM_QPC], s_s[DSA_SM_QPC];
    __shared__ int s_qp[DSA_SM_QPC];
    __shared__ unsigned int s_nmax;
    const unsigned int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    const unsigned int g = lane >> 2, tig = lane & 3u;
    if (tid == 0u) s_nmax = 0u;
    __syncthreads();
    if (tid < Q) {
        unsigned int n = P, s = S;
        if (geom) {
            const int* gr = geom + (size_t)tid * DSA_GEOM_SLOTS;
            s = (unsigned int)gr[DSA_GEOM_S];
            n = min((unsigned int)gr[DSA_GEOM_NPOOLS], n);
        }
        s_n[tid] = n;
        s_s[tid] = s;
        s_qp[tid] = q_pos[tid];
        atomicMax(&s_nmax, n);
    }
    const unsigned int wpr = DSA_SM_QPC / Q, per = (8u + wpr - 1u) / wpr;
    const unsigned int r = warp / wpr, nt0 = (warp % wpr) * per;
    const bool live = r < Q;
    uint32_t a[2][8][4];
    float wv[2][2];
    dsa_sm_load_a(q, weights, r, live, g, tig, a, wv);
    __syncthreads();
    auto cand = [&](unsigned int rr, unsigned int p) {
        return p < s_n[rr] && keys.cand(p, s_s[rr], s_qp[rr]);
    };
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(dsa_smp_smem);
    const unsigned int ntiles = (s_nmax + DSA_SM_PT - 1u) / DSA_SM_PT;
    // 2026-10-23: Walk step k's tile into stage k % 2: 16 B chunk c of pool p's BF16 key (the
    // first 256 B of its slot) to row p of the tile, or zeros for a pool no row may score.
    auto issue = [&](unsigned int k) {
        const unsigned int t = blockIdx.x + k * gridDim.x;
        if (t < ntiles) {
            const uint32_t st = sbase + (k % DSA_SMP_STAGES) * DSA_SMP_TILE_BYTES;
#pragma unroll
            for (unsigned int i = 0; i < DSA_SM_PT * DSA_SMP_CHUNKS / 256u; ++i) {
                const unsigned int e = tid + 256u * i, pp = e / DSA_SMP_CHUNKS;
                const unsigned int c = e % DSA_SMP_CHUNKS, p = t * DSA_SM_PT + pp;
                bool need = false;
                for (unsigned int rr = 0; rr < Q; ++rr) need = need || cand(rr, p);
                const char* src = need ? (const char*)keys.key(p, DSA_SM_D) + 16u * c
                                       : (const char*)keys.pk;
                e4m3g::cp16(st + (pp * DSA_SM_LD + 8u * c) * 2u, src, need);
            }
        }
        e4m3g::cp_commit();
    };
    issue(0u);
    for (unsigned int k = 0;; ++k) {
        const unsigned int t = blockIdx.x + k * gridDim.x;
        if (t >= ntiles) break;
        issue(k + 1u);
        e4m3g::cp_wait<1>();
        __syncthreads();
        const uint32_t st = sbase + (k % DSA_SMP_STAGES) * DSA_SMP_TILE_BYTES;
        if (live) {
            for (unsigned int nt = nt0; nt < nt0 + per && nt < 8u; ++nt) {
                const unsigned int pc = t * DSA_SM_PT + 8u * nt + 2u * tig;
                const bool c0 = cand(r, pc), c1 = cand(r, pc + 1u);
                float k0 = 0.0f, k1 = 0.0f;
                if (__any_sync(0xFFFFFFFFu, c0 || c1)) {
                    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    dsa_sm_ntile_mma(acc0, acc1, a, st, lane, nt);
                    k0 = dsa_sm_fold_col(acc0, acc1, wv, scale, 0);
                    k1 = dsa_sm_fold_col(acc0, acc1, wv, scale, 1);
                }
                const unsigned int n = s_n[r];
                if (g == 0u && pc < n) {
                    out[(size_t)r * P + pc] = c0 ? k0 : -FLT_MAX;
                    valid_cand[(size_t)r * P + pc] = c0 ? 1 : 0;
                }
                if (g == 0u && pc + 1u < n) {
                    out[(size_t)r * P + pc + 1u] = c1 ? k1 : -FLT_MAX;
                    valid_cand[(size_t)r * P + pc + 1u] = c1 ? 1 : 0;
                }
            }
        }
        // 2026-10-23: Every warp has read stage k % 2 before step k + 2's copies overwrite it.
        __syncthreads();
    }
    e4m3g::cp_wait<0>();
}
