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
    dsa_sh_window_walk<false>(q, pk, weights, bt, q_pos, out, valid_cand, Q, P, H, D, KP, S, bs, pk_blk, scale, geom);
}
