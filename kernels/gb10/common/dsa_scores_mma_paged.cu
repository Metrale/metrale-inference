// SPDX-License-Identifier: MIT OR Apache-2.0
// metrale: instantiates dsa_indexer (dsa_scores_mma_paged.cuh)

// 2026-10-23: dsa_index_scores_mma_paged_bf16: the decode / verify selection's scores over the
// BF16 persisted pool keys on BF16 tensor cores (dsa_scores_mma_paged.cuh holds the method, the
// numerics and the launch contract; METRALE_GLM_DSA_DECODE_SCORES_MMA).
//
// Owner: gb10 kernels.
// Invariants: as dsa_scores_mma_paged.cuh; the arguments are dsa_index_scores_heads_window's.

#include "dsa_scores_mma_paged.cuh"

extern "C" __global__ void __launch_bounds__(256, 1) dsa_index_scores_mma_paged_bf16(
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
    if (Q == 0u || Q > DSA_SM_QPC || H != DSA_SM_H || D != DSA_SM_D || blockDim.x != 256u) return;
    const DsaSwPersistedKeys keys{pk, bt, bs, pk_blk, KP};
    dsa_smp_walk(q, keys, weights, q_pos, out, valid_cand, Q, P, S, scale, geom);
}
