// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-22: dsa_index_scores_heads_paged's bytes with a straight-line pool step
// (`METRALE_GLM_DSA_SCORES_HEADS_WINDOW`, default on): the captured decode/verify's ceiling
// scores over the persisted pool keys.
//
// Owner: gb10 kernels.
// Invariants: every score and candidacy byte is dsa_index_scores_heads_paged's, hence
// dsa_index_scores_warp_paged's (gate examples/glm5next_dsa_ceiling_scores_gate.rs).

#include "dsa_scores_heads.cuh"

// 2026-10-22: dsa_index_scores_heads_paged spends about as long per pool on its walk's
// bookkeeping as on the head math (a 2-row verify at 32k pools is compute-bound there): a
// branch per pool around the math and the copy, the next copy's address shuffled right before
// the copy that needs it, and the block-table reads consumed by the address arithmetic as they
// are made. Here the walk is per window of 32 pools:
// - at a window's start the warp holds, per lane, the key address of pool 32 w + l of windows
//   w and w + 1 (block ids read one more window ahead, the address formed a window later), the
//   windows' candidacy as ballot masks, and the window's live count (live pools are a prefix);
// - a pool's step has no branch but the 16-pool flush: the head term is computed for every live
//   pool and stored only for a candidate (a non-candidate's ring slot is a zero-filled copy),
//   the copy RING pools ahead is predicated, and its address is shuffled before the step's
//   math and used after it, once every lane is past the slot it overwrites.
// The head math, the head-order sum, the rows on warps, the strides, the candidacy and the
// launch are dsa_index_scores_heads_paged's (dsa_scores_heads.cuh). Launch: grid (any x, 1, 1),
// block 256, Q <= 8; H <= 32, D == 128 (else nothing is written; the launcher sends no other
// shape), pk 16-byte aligned and pk_blk a multiple of 4.
extern "C" __global__ void __launch_bounds__(256, 1) dsa_index_scores_heads_window(
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
    auto copy = [&](unsigned int j, const float* from, bool on) {
        e4m3g::cp16(dsa_sh_smem(&sring[warp][j % DSA_SH_RING][lane]), from + 4u * lane, on);
        e4m3g::cp_commit();
    };
#pragma unroll
    for (unsigned int t = 0; t < DSA_SH_RING; ++t)
        copy(t, (const float*)__shfl_sync(0xffffffffu, (unsigned long long)cur, (int)t),
             (cm >> t) & 1u);
    unsigned int nb = 0, j = 0;
    for (unsigned int w = 0;; ++w) {
        const unsigned int live = __popc(__ballot_sync(0xffffffffu, wpool(w) < n));
        for (unsigned int t = 0; t < live; ++t, ++j) {
            const unsigned int p = first + (32u * w + t) * step;
            const bool c = (cm >> t) & 1u;
            const unsigned int ta = t + DSA_SH_RING;
            const bool in_cur = ta < 32u;
            const float* ahead = (const float*)__shfl_sync(
                0xffffffffu, (unsigned long long)(in_cur ? cur : nxt), (int)(ta & 31u));
            const bool ahead_on = ((in_cur ? cm : cmn) >> (ta & 31u)) & 1u;
            e4m3g::cp_wait<DSA_SH_RING - 1u>();
            __syncwarp();
            const float term = dsa_sh_pool_term(qr, sring[warp][j % DSA_SH_RING], wh, scale);
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
            copy(j + DSA_SH_RING, ahead, ahead_on);
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
