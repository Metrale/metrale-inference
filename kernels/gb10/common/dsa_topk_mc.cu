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
// 2026-10-24: The bodies live in dsa_topk_mc.cuh (shared with dsa_topk_mc_seg.cu).
//
// Owner: gb10 kernels.

#include "dsa_topk_mc.cuh"

extern "C" __global__ void __launch_bounds__(256) dsa_topk_mc_pass(
    const float* __restrict__ scores,
    unsigned int* __restrict__ ws,
    unsigned int Q,
    unsigned int P,
    unsigned int select_k,
    const int* __restrict__ geom,
    unsigned int pass
) {
    dsa_mc_pass_body(scores, ws, Q, P, select_k, geom, pass, blockIdx.x, blockIdx.y, gridDim.x);
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
    dsa_mc_collect_body(scores, ws, selected, Q, P, select_k, geom, blockIdx.x, blockIdx.y,
                        gridDim.x);
}
