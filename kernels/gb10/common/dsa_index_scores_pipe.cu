// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-14: dsa_index_scores_mma with its key tiles streamed (`METRALE_GLM_DSA_SCORES_PIPE`).
// dsa_index_scores_mma converts each 64-pool tile's FP32 keys to BF16 and evaluates the
// candidacy rule (four gathers per pool) inside every CTA, between two barriers, so a CTA's MMAs
// wait on that work for each tile. Here the conversion and the candidacy's pool-side half run
// once per pool (dsa_pool_prep_bf16), and the CTA streams the prepared tiles through a 3-stage
// cp.async ring while it multiplies the previous one.
//
// Owner: gb10 kernels.
// Invariants:
// - keys_bf16[p][k] is dsa_sm_pack of pool_keys[p][k] (the conversion dsa_index_scores_mma
//   applies in shared memory), so the B tiles hold the same bits.
// - vis[p] = end_c when pool_valid[p] and dsa_row_valid(valid_keys, end_c), else -1, where end_c
//   is the pool's last index clamped to [0, S - 1]; (row r, pool p) is a candidate iff
//   vis[p] >= 0 && vis[p] <= q_pos[r], which is dsa_index_scores_mma's rule (end_c >= 0).
// - Every score and candidacy byte is dsa_index_scores_mma's: the same A fragments, the same
//   tile MMAs skipped under the same warp-uniform condition (no candidate in the row's 64 pools),
//   the same dsa_sm_fold; non-candidates are -FLT_MAX; every (row < Q, pool < P) pair is written.
// - 2026-10-14 (I2 tail skip): a CTA streams only tiles 0 .. n_live - 1, where n_live covers the
//   last pool that is a candidate of ANY of its rows (scanned from vis, no ordering assumed). Every
//   pool past it is a non-candidate of every row of the CTA, so its -FLT_MAX / 0 bytes are written
//   directly, which are the bytes the MMA path writes for a non-candidate.

#include "dsa_indexer_body.cuh"
#include "dsa_scores_mma.cuh"
#include "e4m3_mma_pipe.cuh"
#include <float.h>
#include <stdint.h>

#define DSA_SP_STAGES 3u
#define DSA_SP_TILE_BYTES (DSA_SM_PT * DSA_SM_LD * 2u)
#define DSA_SP_CHUNKS (DSA_SM_D / 8u)

// 2026-10-14: The keys of pools 0 .. P - 1 as BF16 rows and each pool's visibility position.
// Launch: grid (ceil(P / 8), 1, 1), block 256 (32 threads of 4 channels per pool); D = 128 only
// (else nothing is written; the launcher never sends another shape).
extern "C" __global__ void __launch_bounds__(256) dsa_pool_prep_bf16(
    const float* __restrict__ pool_keys,
    const int* __restrict__ pool_indices,
    const unsigned char* __restrict__ pool_valid,
    const unsigned char* __restrict__ valid_keys,
    __nv_bfloat16* __restrict__ keys_bf16,
    int* __restrict__ vis,
    unsigned int P,
    unsigned int D,
    unsigned int KP,
    unsigned int S
) {
    if (D != DSA_SM_D) return;
    const unsigned int p = blockIdx.x * 8u + (threadIdx.x >> 5), c4 = threadIdx.x & 31u;
    if (p >= P) return;
    const float4 v = ((const float4*)(pool_keys + (size_t)p * DSA_SM_D))[c4];
    uint2 packed;
    packed.x = dsa_sm_pack(make_float2(v.x, v.y));
    packed.y = dsa_sm_pack(make_float2(v.z, v.w));
    *(uint2*)(keys_bf16 + (size_t)p * DSA_SM_D + c4 * 4u) = packed;
    if (c4 == 0u) {
        const int end = pool_indices[(size_t)p * KP + KP - 1u];
        const int end_c = end < 0 ? 0 : (end >= (int)S ? (int)S - 1 : end);
        vis[p] = (pool_valid[p] != 0 && dsa_row_valid(valid_keys, end_c)) ? end_c : -1;
    }
}

// 2026-10-14: Tile t's keys and visibility into ring stage t % 3, as one cp.async group; pools
// past P are zero-filled. A t past the last tile commits an empty group, so the wait counts of
// the main loop hold to the end.
__device__ __forceinline__ void dsa_sp_issue(const __nv_bfloat16* __restrict__ keys_bf16,
                                             const int* __restrict__ vis, unsigned int P,
                                             unsigned int t, unsigned int ntiles, uint32_t sbase,
                                             uint32_t vbase) {
    if (t < ntiles) {
        const unsigned int s = t % DSA_SP_STAGES, p0 = t * DSA_SM_PT;
        for (unsigned int e = threadIdx.x; e < DSA_SM_PT * DSA_SP_CHUNKS; e += blockDim.x) {
            const unsigned int pp = e / DSA_SP_CHUNKS, c = e % DSA_SP_CHUNKS, gp = p0 + pp;
            const __nv_bfloat16* src = keys_bf16 + (size_t)(gp < P ? gp : 0u) * DSA_SM_D + c * 8u;
            e4m3g::cp16(sbase + s * DSA_SP_TILE_BYTES + (pp * DSA_SM_LD + c * 8u) * 2u, src, gp < P);
        }
        if (threadIdx.x < DSA_SM_PT) {
            const unsigned int gp = p0 + threadIdx.x;
            e4m3g::cp4(vbase + (s * DSA_SM_PT + threadIdx.x) * 4u, vis + (gp < P ? gp : 0u), gp < P);
        }
    }
    e4m3g::cp_commit();
}

// 2026-10-14: dsa_index_scores_mma's scores and candidacy from the prepared keys_bf16 / vis.
// One CTA of 8 warps per 8 query rows, warp w on row 8 * blockIdx.x + w, its A fragments in
// registers for the whole pool walk. Tile t: wait for its cp.async group, one barrier (which
// also retires tile t - 1's reads of the stage tile t + 2 then overwrites), issue tile t + 2,
// then the MMAs and the epilogue. Lane (g, tig) keeps the folded score of columns
// 8 g + 2 tig + {0, 1} (every lane of a column holds its bits after dsa_sm_fold) and writes
// those two pools' score and candidacy, so a warp's 64 writes per tile are contiguous.
// Before the walk the CTA takes the largest q_pos of its rows and the last pool any of them can
// see (vis scanned in full), so the ring stops at that pool's tile (I2 tail skip, see Invariants).
// Launch: grid (ceil(Q / 8), 1, 1), block 256, dynamic shared memory DSA_SP_STAGES x
// (DSA_SP_TILE_BYTES + 256) = 52,992 B; H = 32 and D = 128 only.
extern "C" __global__ void __launch_bounds__(256) dsa_index_scores_mma_pipe(
    const float* __restrict__ q,
    const __nv_bfloat16* __restrict__ keys_bf16,
    const float* __restrict__ weights,
    const int* __restrict__ vis,
    const int* __restrict__ q_pos,
    float* __restrict__ out,
    unsigned char* __restrict__ valid_cand,
    unsigned int Q,
    unsigned int P,
    unsigned int H,
    unsigned int D,
    float scale
) {
    extern __shared__ __align__(16) unsigned char dsa_sp_smem[];
    if (H != DSA_SM_H || D != DSA_SM_D) return;
    const unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned int g = lane >> 2, tig = lane & 3u;
    const unsigned int r = blockIdx.x * DSA_SM_QPC + warp;
    const bool live = r < Q;
    const int qp = live ? q_pos[r] : -1;
    const int* sV = (const int*)(dsa_sp_smem + DSA_SP_STAGES * DSA_SP_TILE_BYTES);
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(dsa_sp_smem);
    const uint32_t vbase = sbase + DSA_SP_STAGES * DSA_SP_TILE_BYTES;

    __shared__ int s_qmax, s_pmax;
    if (threadIdx.x == 0) {
        s_qmax = -1;
        s_pmax = -1;
    }
    __syncthreads();
    if (lane == 0 && live) atomicMax(&s_qmax, qp);
    __syncthreads();
    const int cq = s_qmax;
    int pm = -1;
    for (unsigned int p = threadIdx.x; p < P; p += blockDim.x) {
        const int v = vis[p];
        if (v >= 0 && v <= cq) pm = (int)p;
    }
    if (pm >= 0) atomicMax(&s_pmax, pm);

    uint32_t a[2][8][4];
    float wv[2][2];
    dsa_sm_load_a(q, weights, r, live, g, tig, a, wv);
    __syncthreads();
    const unsigned int ntiles = ((unsigned int)(s_pmax + 1) + DSA_SM_PT - 1u) / DSA_SM_PT;
    dsa_sp_issue(keys_bf16, vis, P, 0u, ntiles, sbase, vbase);
    dsa_sp_issue(keys_bf16, vis, P, 1u, ntiles, sbase, vbase);
    const unsigned int pl = 8u * g + 2u * tig;
    for (unsigned int t = 0; t < ntiles; ++t) {
        e4m3g::cp_wait<1>();
        __syncthreads();
        dsa_sp_issue(keys_bf16, vis, P, t + 2u, ntiles, sbase, vbase);
        const unsigned int s = t % DSA_SP_STAGES, gp = t * DSA_SM_PT + pl;
        const int v0 = sV[s * DSA_SM_PT + pl], v1 = sV[s * DSA_SM_PT + pl + 1u];
        const bool c0 = gp < P && v0 >= 0 && v0 <= qp;
        const bool c1 = gp + 1u < P && v1 >= 0 && v1 <= qp;
        // 2026-10-14: Warp-uniform, as in dsa_index_scores_mma: the row's 64 pools of the tile.
        const bool mine = __any_sync(0xFFFFFFFFu, c0 || c1);
        float k0 = 0.0f, k1 = 0.0f;
        if (live && mine) {
            float acc[2][8][4];
            dsa_sm_zero(acc);
            dsa_sm_tile_mma(acc, a, sbase + s * DSA_SP_TILE_BYTES, lane);
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const float f0 = dsa_sm_fold(acc, wv, scale, nt, 0);
                const float f1 = dsa_sm_fold(acc, wv, scale, nt, 1);
                k0 = nt == (int)g ? f0 : k0;
                k1 = nt == (int)g ? f1 : k1;
            }
        }
        if (live && gp < P) {
            out[(size_t)r * P + gp] = c0 ? k0 : -FLT_MAX;
            valid_cand[(size_t)r * P + gp] = c0 ? 1 : 0;
        }
        if (live && gp + 1u < P) {
            out[(size_t)r * P + gp + 1u] = c1 ? k1 : -FLT_MAX;
            valid_cand[(size_t)r * P + gp + 1u] = c1 ? 1 : 0;
        }
    }
    e4m3g::cp_wait<0>();
    if (live) {
        for (unsigned int p = ntiles * DSA_SM_PT + lane; p < P; p += 32u) {
            out[(size_t)r * P + p] = -FLT_MAX;
            valid_cand[(size_t)r * P + p] = 0;
        }
    }
}
