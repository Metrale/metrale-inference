// SPDX-License-Identifier: MIT OR Apache-2.0
// metrale: instantiates dsa_indexer (dsa_topk_mc.cuh)

// 2026-10-24: dsa_topk_mc_seg (`--glm-dsa-topk-seg`, R4): the multi-CTA radix top-k's five
// launches (dsa_topk_mc_pass x 4, dsa_topk_mc_collect; dsa_topk_mc.cu) as one persistent segment
// (segment.cuh): 5 phases of C x Q tasks, task t of a phase being CTA (t % C, t / C) of the
// five-launch grid (C, Q). Any grid size completes (tickets in order, no co-residency needed);
// the launcher uses C x Q blocks of 256.
//
// Owner: gb10 kernels.
// Invariants:
// - Every task runs the five-launch path's body on the same (c, r, C), so `selected` holds the
//   five launches' bytes at the same C (examples/glm5next_dsa_topk_mc_gate.rs).
// - `ctl` (SegCtlN<5>, 32 bytes) is zero at launch; the last block resets it (replay-safe).

#include "dsa_topk_mc.cuh"
#include "segment.cuh"

extern "C" __global__ void __launch_bounds__(256) dsa_topk_mc_seg(
    const float* __restrict__ scores,
    unsigned int* __restrict__ ws,
    int* __restrict__ selected,
    unsigned int Q,
    unsigned int P,
    unsigned int select_k,
    const int* __restrict__ geom,
    unsigned int C,
    SegCtlN<DSA_MC_SEG_PHASES>* __restrict__ ctl
) {
    const unsigned int n = C * Q;
    const unsigned int tasks[DSA_MC_SEG_PHASES] = {n, n, n, n, n};
    segment_run(ctl, tasks, 0xFFFFFFFFu, DsaMcSegRun{scores, ws, selected, Q, P, select_k, geom, C});
}
